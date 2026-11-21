test "metadata guard checks an expiring victim once and selects the confirmed alternative" {
    const p = 512 * C;
    var t: T = .{ .gpa = std.testing.allocator, .m = .none, .verbose = true };
    var f = try Fx.init(&t, 2, p);
    defer f.deinit();
    try f.kept(41, 0, 145 * C, 145 * C);
    try f.kept(42, 161 * C, 146 * C, 307 * C);
    try f.kept(43, 330 * C, 146 * C, 476 * C);
    try f.kept(44, p, 146 * C, p + 148 * C);
    try f.live(1, p, 148 * C);
    try f.live(2, p + 150 * C, 148 * C);
    try f.live(3, p + 300 * C, 148 * C);
    const request = rq(4, 148 * C, 44);
    var plain = try f.plan(request);
    defer plain.deinit(t.gpa);
    try std.testing.expect(plain.mode == .carry and staledAre(&plain, &.{41}));
    const Proof = struct {
        fn confirmed(ctx: *anyopaque, id: u64, at: usize) bool {
            const calls: *usize = @ptrCast(@alignCast(ctx));
            calls.* += 1;
            return (id == 42 or id == 43) and at == 146 * C;
        }
    };
    var calls: usize = 0;
    var guard: BenchGuard = .{ .ctx = &calls, .check = Proof.confirmed };
    f.pl.bench_guard = &guard;
    try std.testing.expect(f.pl.benchRisk(&plain));
    guard.ranked = true;
    var guarded = try f.plan(request);
    defer guarded.deinit(t.gpa);
    try std.testing.expect(guarded.mode == .carry and staledAre(&guarded, &.{42}));
    try std.testing.expectEqual(plain.source, guarded.source);
    try std.testing.expect(!f.pl.benchRisk(&guarded));
    try std.testing.expectEqual(@as(usize, 2), calls);
    try f.apply(&guarded);
    try std.testing.expect(!f.kp(41).stale and f.kp(42).stale);
}
