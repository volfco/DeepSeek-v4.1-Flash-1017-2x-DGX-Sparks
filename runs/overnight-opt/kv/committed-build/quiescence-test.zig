test "benchmark aggregate quiescence does not identify committed paired points" {
    const pr = try Pair.start("pendingpointproof", 0, 92);
    defer pr.deinit();
    var fam: Fam0 = .{ .sim = pr.r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 5);
    defer ld.destroy();
    ld.bench_check = true;
    const t = ld.tier();
    const w = &pr.w;
    var ids: [4 * bt]u32 = undefined;
    testIds(&ids, 81);
    const key = format.pointKey(&ids, bt, &.{});
    // Baseline from the existing once-per-second stats path. No pressure lookup in flight.
    ld.stats_at = 0;
    ld.askStats();
    try testing.expectEqual(@as(usize, 1), ld.peers_n);
    try testing.expectEqual(@as(u64, 0), ld.peers[0].writes_pending);
    try w.send(T.hold, &.{&.{1}});
    const first = try keepBoth(w, t, pr.r0.sim, 1, 0, &ids, &ids);
    defer gpa.destroy(first);
    try pumpUntil(ld, 1);
    try testing.expect(ld.side.store.presence(key));
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().pending_jobs);
    // The leader has finished. Its last real peer sample still says zero.
    try testing.expectEqual(@as(u64, 0), ld.peers[0].writes_pending);
    ld.stats_at = 0;
    ld.askStats();
    try testing.expect(ld.peers[0].writes_pending > 0);
    try testing.expect(ld.benchPresence(first.id, ids.len) != .present);
    try w.send(T.hold, &.{&.{0}});
    try w.waitPresent(key, true);
    try testing.expect(ld.benchPresence(first.id, ids.len) == .present);

    // An ordinary follower may decline a keep whose slot IDs differ. No pending write is left there.
    var second_ids: [4 * bt]u32 = undefined;
    testIds(&second_ids, 83);
    const second = try keepBoth(w, t, pr.r0.sim, 2, 4 * bt, &second_ids, &ids);
    defer gpa.destroy(second);
    try pumpUntil(ld, 2);
    const next_key = format.pointKey(&second_ids, bt, &.{});
    try w.waitPresent(next_key, false);
    ld.stats_at = 0;
    ld.askStats();
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().pending_jobs);
    try testing.expectEqual(@as(u64, 0), ld.peers[0].writes_pending);
    try testing.expect(ld.side.store.presence(next_key));
    try testing.expect(ld.benchPresence(first.id, ids.len) == .present);
    try testing.expect(ld.benchPresence(second.id, second_ids.len) == .peer_missing);
    try testing.expect(ld.broken == null);
    try testing.expectEqual(@as(usize, 0), fam.reserved);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().loaded);
    std.debug.print("CACHED_PENDING=0 FRESH_PENDING=positive FRESH_BOTH_IDLE=true FIRST=present SECOND=peer_missing READS=0 RESERVATIONS=0\n", .{});
    try pr.finish();
}
