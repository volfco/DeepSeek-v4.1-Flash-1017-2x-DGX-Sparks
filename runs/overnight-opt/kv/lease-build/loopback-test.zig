
test "benchmark metadata lease allows local decisions while a real loopback follower is busy" {
    const pr = try Pair.start("retentionlease", 0, 93);
    defer pr.deinit();
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = pr.r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 5);
    defer ld.destroy();
    const t = ld.tier();
    var ids: [5 * bt]u32 = undefined;
    testIds(&ids, 78);
    var made: usize = 0;
    var snaps: [5]*Snap = undefined;
    defer for (snaps[0..made]) |sn| gpa.destroy(sn);
    for (1..5) |turn| {
        const tokens = ids[0 .. turn * bt];
        snaps[made] = try keepBoth(w, t, pr.r0.sim, turn, 0, tokens, tokens);
        made += 1;
        try pumpUntil(ld, turn);
        try w.waitPresent(format.pointKey(tokens, bt, &.{}), true);
    }
    const first = format.pointKey(ids[0..bt], bt, &.{});
    pr.r0.sim.delay = std.math.maxInt(u32);
    try w.send(T.hold, &.{&.{1}});
    snaps[made] = try keepBoth(w, t, pr.r0.sim, 5, 0, &ids, &ids);
    made += 1;
    const point: spill.store.LeasePoint = .{ .key = first, .at = bt, .rank = 1, .compat = ld.side.store.cfg.compat, .part = null };
    const sent = std.Io.Clock.awake.now(io).toNanoseconds();
    try w.send(32, &.{ &first, &le(u32, bt), &le(u32, 500), &le(u64, 71) });
    var buffer: [64]u8 = undefined;
    const grant_frame = try w.l.recvWithin(&buffer, std.time.ns_per_s);
    try testing.expectEqual(@as(usize, 45), grant_frame.len);
    try testing.expectEqual(@as(u8, 1), grant_frame[0]);
    const grant: spill.store.LeaseGrant = .{ .point = point, .incarnation = grant_frame[1..17].*,
        .serial = std.mem.readInt(u64, grant_frame[17..25], .little), .bytes = std.mem.readInt(u64, grant_frame[25..33], .little),
        .ttl_ms = std.mem.readInt(u32, grant_frame[33..37], .little), .request_nonce = std.mem.readInt(u64, grant_frame[37..45], .little) };
    const view = spill.store.LeaseView.fromReply(grant, 71, sent, std.Io.Clock.awake.now(io).toNanoseconds()) orelse return error.ExpectedView;
    try w.send(T.hold, &.{&.{0}});
    try w.waitPresent(format.pointKey(&ids, bt, &.{}), true);
    try w.waitPresent(first, true);
    try w.send(33, &.{&le(u32, 80)});
    const check_start = std.Io.Clock.awake.now(io).toNanoseconds();
    for (0..1000) |_| {
        try testing.expect(view.permits(grant.incarnation, point, std.Io.Clock.awake.now(io).toNanoseconds()));
        try testing.expect(ld.side.store.presence(first));
    }
    const check_ns = std.Io.Clock.awake.now(io).toNanoseconds() - check_start;
    try testing.expect(check_ns < 40 * std.time.ns_per_ms);
    const reply = try w.l.recvWithin(&buffer, std.time.ns_per_s);
    const service_ms = @divTrunc(std.Io.Clock.awake.now(io).toNanoseconds() - check_start, std.time.ns_per_ms);
    try testing.expectEqualSlices(u8, &.{1}, reply);
    try testing.expect(service_ms >= 70);
    try w.waitPresent(first, false);
    try testing.expect(!view.permits(grant.incarnation, point, std.Io.Clock.awake.now(io).toNanoseconds()));
    try testing.expect(ld.side.store.presence(first));
    try testing.expectEqual(@as(usize, 0), fam.reserved);
    try testing.expectEqual(@as(u8, 0), (try w.query(first))[1]);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().loaded);
    try testing.expect(ld.broken == null);
    std.debug.print("PAIRED_FOLLOWER_TRIM=deferred THEN_EXPIRED=true FOLLOWER_BUSY_MS={d} LOCAL_1000_CHECKS_US={d} READS=0 DESTINATION_RESERVATIONS=0 PAIR_HEALTHY=true\n", .{service_ms, @divTrunc(check_ns, std.time.ns_per_us)});
    pr.r0.sim.delay = 0;
    @memset(&pr.r0.sim.pending, 0);
    try pumpUntil(ld, 5);
    try pr.finish();
}
