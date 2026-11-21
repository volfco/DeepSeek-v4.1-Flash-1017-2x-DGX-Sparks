// Added to the existing CPU spill fixture. Test-only frame 32 sleeps on the follower before its next command.
test "benchmark proof under delayed metadata replies reports a whole-pair break after the real timeout" {
    const pr = try Pair.start("pointproofdelay", 0, 90);
    defer pr.deinit();
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = pr.r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 5);
    defer ld.destroy();
    ld.bench_check = true;
    const t = ld.tier();
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 75);
    const sn = try keepBoth(w, t, pr.r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(sn);
    try pumpUntil(ld, 1);
    const key = format.pointKey(&ids, bt, &.{});
    try w.waitPresent(key, true);
    try testing.expect(ld.benchPresence(sn.id, ids.len) == .present);
    var wait: [4]u8 = undefined;
    std.mem.writeInt(u32, &wait, 80, .little);
    try w.send(32, &.{&wait});
    const slow_start = std.Io.Clock.awake.now(io).toNanoseconds();
    try testing.expect(ld.benchPresence(sn.id, ids.len) == .present);
    const slow_ms = @divFloor(std.Io.Clock.awake.now(io).toNanoseconds() - slow_start, std.time.ns_per_ms);
    try testing.expect(slow_ms >= 70);
    const observer = struct {
        fn broken(ctx: ?*anyopaque, err: anyerror) void {
            const reported: *bool = @ptrCast(@alignCast(ctx.?));
            reported.* = err == error.PairBroken;
        }
    };
    var reported = false;
    ld.on_broken = observer.broken;
    ld.broken_ctx = &reported;
    try testing.expectEqual(@as(u64, 10 * std.time.ns_per_s), ld.lead.?.timeouts.has_ns);
    std.mem.writeInt(u32, &wait, 10600, .little);
    try w.send(32, &.{&wait});
    const began = std.Io.Clock.awake.now(io).toNanoseconds();
    try testing.expect(ld.benchPresence(sn.id, ids.len) == .busy);
    const elapsed = @divFloor(std.Io.Clock.awake.now(io).toNanoseconds() - began, std.time.ns_per_ms);
    try testing.expect(elapsed >= 9800 and elapsed < 10600);
    try testing.expectEqual(@as(anyerror, error.PairBroken), ld.broken.?);
    try testing.expectEqual(@as(anyerror, error.ReplyTimeout), ld.lead.?.broken.?.err);
    try testing.expectEqual(@as(u16, 1), ld.lead.?.broken.?.rank);
    try testing.expect(reported);
    try testing.expect(ld.benchPresence(sn.id, ids.len) == .unmapped);
    try testing.expectEqual(@as(usize, 0), fam.reserved);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().loaded);
    std.debug.print("RESPONSIVE_MS={d} TIMEOUT_MS={d} FOLLOWER=1 CAUSE=ReplyTimeout PAIR_CALLBACK=true READS=0 RESERVATIONS=0\n", .{ slow_ms, elapsed });
    try pr.finish();
}
