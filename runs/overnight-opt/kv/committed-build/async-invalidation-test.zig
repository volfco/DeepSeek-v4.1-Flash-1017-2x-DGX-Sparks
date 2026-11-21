test "benchmark cached affirmative is invalidated by follower publication without another keep or forget" {
    const pr = try Pair.start("asynctrimproof", 0, 91);
    defer pr.deinit();
    const w = &pr.w;
    var fam: Fam0 = .{ .sim = pr.r0.sim };
    defer fam.deinit();
    const ld = try pr.leader(&fam, 5);
    defer ld.destroy();
    ld.bench_check = true;
    const t = ld.tier();
    var ids: [6 * bt]u32 = undefined;
    testIds(&ids, 76);
    var snaps: [5]*Snap = undefined;
    var made: usize = 0;
    defer for (snaps[0..made]) |sn| gpa.destroy(sn);
    const first = format.pointKey(ids[0..bt], bt, &.{});
    for (1..5) |turn| {
        const tokens = ids[0 .. turn * bt];
        snaps[made] = try keepBoth(w, t, pr.r0.sim, turn, 0, tokens, tokens);
        made += 1;
        try pumpUntil(ld, turn);
        try w.waitPresent(format.pointKey(tokens, bt, &.{}), true);
    }
    try testing.expect(ld.benchPresence(snaps[0].id, bt) == .present);
    // Both ranks have the same normal four-turn retention. Hold only simulated device-copy
    // completion, so the fifth write is pending before the final paired positive observation.
    pr.r0.sim.delay = std.math.maxInt(u32);
    try w.send(T.hold, &.{&.{1}});
    const fifth = ids[0 .. 5 * bt];
    snaps[made] = try keepBoth(w, t, pr.r0.sim, 5, 0, fifth, fifth);
    made += 1;
    const affirmative = ld.benchPresence(snaps[0].id, bt);
    try testing.expect(affirmative == .present);
    try testing.expect(ld.side.store.presence(first));
    // No further serving keep, forget, restore, or cap change. The test-device release lets
    // the follower's already-queued fifth manifest publish and trim its oldest ancestor.
    try w.send(T.hold, &.{&.{0}});
    try w.waitPresent(format.pointKey(fifth, bt, &.{}), true);
    try w.waitPresent(first, false);
    try testing.expect(ld.side.store.presence(first));
    try testing.expect(ld.benchPresence(snaps[0].id, bt) == .peer_missing);
    try testing.expect(affirmative == .present);
    try testing.expect(ld.broken == null);
    try testing.expectEqual(@as(usize, 0), fam.reserved);
    try testing.expectEqual(@as(u64, 0), ld.side.store.counts().loaded);
    std.debug.print("CACHED=present CURRENT=peer_missing HEAD=present BOTH_TURNS=4 NEW_KEEPS_AFTER_PROOF=0 EXPLICIT_FORGETS=0\n", .{});
    pr.r0.sim.delay = 0;
    @memset(&pr.r0.sim.pending, 0);
    try pumpUntil(ld, 5);
    try pr.finish();
}
