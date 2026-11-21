const std = @import("std");
const cuda = @import("cuda");
const tri = @import("tri.zig");
const markov = @import("tri_markov.zig");
const shape = @import("prompt_rows.zig");

// No CUDA context or addresses are used. Exercise the production wrapper and dispatch's
// exact-or-fallback predicate against the deployed kit for both output types.
test "deployed TP2 wo_b kernel coverage" {
    const parsed = try std.json.parseFromSlice(cuda.aot.SetJson, std.testing.allocator, @embedFile("kit-bf16-diagnostic.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 198), parsed.value.kernels.len);
    try std.testing.expect(!shape.denseOnGemm(128, shape.default_gemm));
    for ([_]usize{ 129, 131, 1024, 2048 }) |rows| {
        try std.testing.expect(shape.denseOnGemm(rows, shape.default_gemm));
        for ([_]markov.OutType{ .bf16, .fp32 }) |out_type| {
            var log = tri.Log{ .gpa = std.testing.allocator };
            defer log.deinit();
            try markov.matmul(.{ .log = &log }, 0x1000, 0x2000, 0x3000, 0x4000, null, 0x5000, out_type, 5120, rows, 4096, 5120);
            const launch = log.items.items[0];
            const resolved = cuda.aot.specsHave(parsed.value.kernels, launch.name, launch.args, launch.consts);
            std.debug.print(" {d} rows, {s} output, kit resolves = {}\n", .{ rows, @tagName(out_type), resolved });
            try std.testing.expectEqual(out_type == .fp32, resolved);
        }
    }
}
