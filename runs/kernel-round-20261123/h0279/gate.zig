//! h0279's host gate against the WORKTREE's real modules: TF_DS_IDX_PICK_WIDE's arm decisions at BRIEF-0279's case
//! matrix, through fused_trace.idxArm (imported from the worktree path build.sh passes as -Dwt; the module's closure
//! owns the real idx_pick_wide.wideOk that idxArm consults). 16,385 / 32,768 / 65,536 blocks take the wide arm at
//! rows 1 and 7 with k = 2048 (decode); 65,537 refuses and keeps the unfused three-launch fallback; the switch off,
//! prompt and rows past 7 keep today's arms. Exit 0: all match.
const std = @import("std");
const fused_trace = @import("fused_trace");

var fails: usize = 0;

fn arm(name: []const u8, fused: bool, wide: bool, allow_small: bool, rows: usize, width: usize, nbk: usize, k: usize, want: fused_trace.Arm) void {
    const got = fused_trace.idxArm(fused, wide, allow_small, rows, width, 8, nbk, k);
    const ok = got == want;
    if (!ok) fails += 1;
    std.debug.print("{s}: idxArm(fused={any}, wide={any}, rows={d}, nbk={d}, k={d}) = {s} (want {s}) {s}\n", .{ name, fused, wide, rows, nbk, k, @tagName(got), @tagName(want), if (ok) "ok" else "FAIL" });
}

pub fn main() !void {
    std.debug.print("h0279 gate: the worktree's fused_trace.idxArm (over idx_pick_wide.wideOk) at the brief's matrix\n", .{});
    // the wide window (the served widths: a short last block, width = 8 * nbk - 3, and whole blocks)
    for ([_]usize{ 1, 7 }) |rows| {
        arm("16,385 blocks", true, true, true, rows, 8 * 16385 - 3, 16385, 2048, .idx_wide);
        arm("32,768 blocks", true, true, true, rows, 8 * 32768, 32768, 2048, .idx_wide);
        arm("65,536 blocks", true, true, true, rows, 8 * 65536, 65536, 2048, .idx_wide);
        // 65,537 refuses: the unfused fallback (block_max + topk_indices + pool_pick) stays
        arm("65,537 blocks", true, true, true, rows, 8 * 65537, 65537, 2048, .idx_unfused);
    }
    // the refusal's neighbours and the off path
    arm("switch off", true, false, true, 7, 8 * 32768, 32768, 2048, .idx_unfused);
    arm("prompt arm", true, true, false, 7, 8 * 32768, 32768, 2048, .idx_unfused);
    arm("rows 8", true, true, true, 8, 8 * 32768, 32768, 2048, .idx_unfused);
    arm("k past the cap", true, true, true, 7, 8 * 16385, 16385, 2049, .idx_unfused);
    // today's regimes unchanged (the off path is byte- and launch-identical)
    arm("staging bound (small)", true, true, true, 7, 8 * 16384, 16384, 2048, .idx_small);
    arm("small pool", true, true, true, 7, 8 * 6144, 6144, 2048, .idx_small);
    arm("many rows (fused)", true, true, true, 32, 8 * 16384, 16384, 2048, .idx_fused);
    arm("prompt pooled", true, true, false, 1, 65533, 8192, 2048, .idx_fused);
    arm("everything off", false, false, true, 7, 65533, 8192, 2048, .idx_unfused);
    std.debug.print("h0279 gate: {s} ({d} mismatches)\n", .{ if (fails == 0) "all arms as expected" else "FAIL", fails });
    if (fails != 0) std.process.exit(1);
}
