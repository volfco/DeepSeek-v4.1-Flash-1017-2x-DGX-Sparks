// Appended to the same committed planner source as planner-test.zig.
const AlwaysMissing = struct {
    fn probe(ctx: *anyopaque, _: u64, _: usize) bool {
        const calls: *usize = @ptrCast(@alignCast(ctx));
        calls.* += 1;
        return false;
    }
};

test "a full partition with no recoverable candidate retains its histories until the patient deadline" {
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
    var calls: usize = 0;
    var guard: BenchGuard = .{ .ctx = &calls, .check = AlwaysMissing.probe };
    f.pl.bench_guard = &guard;
    const early = rq(4, 148 * C, 44);
    var plain = try f.plan(early);
    defer plain.deinit(t.gpa);
    try std.testing.expect(f.pl.benchRisk(&plain));
    guard.ranked = true;
    var alternate = try f.plan(early);
    defer alternate.deinit(t.gpa);
    try std.testing.expect(f.pl.benchRisk(&alternate));
    guard.ranked = false;
    var fallback = try f.plan(rqp(4, 148 * C, 44));
    defer fallback.deinit(t.gpa);
    try std.testing.expectEqual(Status.full, fallback.status);
    try std.testing.expect(staledAre(&fallback, &.{}));
    try std.testing.expect(calls > 0 and calls <= 8);
    f.pl.bench_guard = null;
    var expired = try f.plan(early);
    defer expired.deinit(t.gpa);
    try std.testing.expect(staledAre(&expired, &.{41}));
}

test "one admission consults at most eight distinct kept IDs and reuses negative answers" {
    var t: T = .{ .gpa = std.testing.allocator, .m = .none, .verbose = true };
    var f = try Fx.init(&t, 2, 512 * C);
    defer f.deinit();
    var calls: usize = 0;
    var guard: BenchGuard = .{ .ctx = &calls, .check = AlwaysMissing.probe };
    f.pl.bench_guard = &guard;
    for (0..12) |i| try f.kept(i, 2 * C * i, C, 2 * C * i + C);
    for (0..2) |_| {
        for (0..12) |i| try std.testing.expect(!f.pl.benchVerified(i));
    }
    try std.testing.expectEqual(@as(usize, 8), calls);
}
