//! h0268: TF_DS_DRAFT_QKV_NORM's host check (BRIEF-0268): the arguments the drafter's joint _q_kv_norm call receives
//! against the two separate calls' (_rmsnorm_rot + _kv_norm_rope), over N 1..4 streams x n 5..7 block rows, with and
//! without LoRA, through the worktree's real tri_norm wrappers. Kernel arithmetic parity (_q_kv_norm's cubin against
//! the two separate cubins) is GPU-gate only; this pins the launch construction: widths, strides, slot lists, the -1
//! sentinel, the const sets and the kit variant class (fixtures/launches.json's recorded _q_kv_norm launch).
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const tri = @import("tri.zig");
const tri_norm = @import("tri_norm.zig");
const dq = @import("draft_qkv_norm.zig");

test {
    // the copied worktree modules' own tests run here too (every recorded launch's conformance, the switch's forms)
    _ = tri;
    _ = tri_norm;
    _ = dq;
}

// the served drafter's widths (fixtures/launches.json: _q_kv_norm's round:r16:b1024 record and the draft_batch
// captures' _rmsnorm_rot / _kv_norm_rope): query width 1,280, KV width 512, RoPE width 64, one input rotation
const D: usize = 1280; // lay.wq_a.n: the query width (the q norm's)
const HD: usize = 512; // c.head_dim: the KV width
const RD: usize = 64; // c.rope_dim
const RING: usize = 144; // pool.ring (window 128 + ring_extra 16)
const SLOTS: usize = 4; // pool.slots (the served --parallel 4)
const STAGES: usize = 3; // the served checkpoint's DSpark blocks (one norm pair a block)
const MAX_ROWS = 28; // 4 streams x 7 block rows
const eps: f32 = 1e-20; // c.eps (the fixtures' scalar)

/// A modeled stage's buffers and weights: the values Drafter.attention passes (fake 16-aligned, distinct addresses).
const Stage = struct { qa: u64, ykv: u64, qr: u64, xhq: u64, kvb: u64, ring: u64, pos: u64, sent: u64, q_norm: u64, kv_norm: u64, suh: u64, cos: u64, sin: u64 };

fn stage(j: u64) Stage {
    var s: Stage = undefined;
    var i: u64 = 0;
    inline for (@typeInfo(Stage).@"struct".field_names) |name| {
        @field(s, name) = 0x7f00_0000_0000 + j * 0x100_0000_0000 + i * 0x1_0000;
        i += 1;
    }
    return s;
}

/// The fallback arm's two launches, exactly as Drafter.attention makes them without the switch (draft.zig).
fn separate(t: tri.Tri, s: Stage, R: usize) !void {
    const rot = [_]tri_norm.Rot{.{ .suh = s.suh, .h = s.xhq }};
    try tri_norm.rmsnormRot(t, s.qa, D, s.q_norm, eps, &rot, s.qr, D, R, D);
    try tri_norm.kvNormRope(t, s.ykv, s.kv_norm, s.cos, s.sin, s.pos, s.ring, SLOTS * RING, s.sent, eps, true, RD, s.kvb, R, HD);
}

/// The switch arm's one launch, exactly as Drafter.attention makes it (draft.zig).
fn joint(t: tri.Tri, s: Stage, R: usize) !void {
    const rot = [_]tri_norm.Rot{.{ .suh = s.suh, .h = s.xhq }};
    try tri_norm.qKvNorm(t, s.qa, D, s.q_norm, eps, rot[0..1], s.ykv, s.kv_norm, s.cos, s.sin, s.pos, s.ring, SLOTS * RING, s.sent, true, RD, s.qr, s.kvb, R, D, HD);
}

fn arg(k: tri.Log.Kept, name: []const u8) aot.Arg {
    for (k.args) |a| if (std.mem.eql(u8, a.name, name)) return a;
    std.debug.panic("no argument {s} in {s}", .{ name, k.name });
}

fn argEq(a: aot.Arg, b: aot.Arg) bool {
    return switch (a.value) {
        .ptr => |x| switch (b.value) {
            .ptr => |y| x.addr == y.addr and std.mem.eql(u8, x.ty, y.ty),
            else => false,
        },
        .i32 => |x| switch (b.value) { .i32 => |y| x == y, else => false },
        .f32 => |x| switch (b.value) {
            .f32 => |y| @as(u32, @bitCast(x)) == @as(u32, @bitCast(y)),
            else => false,
        },
        .u64 => |x| switch (b.value) { .u64 => |y| x == y, else => false },
    };
}

fn constOf(k: tri.Log.Kept, name: []const u8) ?aot.Const {
    for (k.consts) |c| if (std.mem.eql(u8, c.name, name)) return c;
    return null;
}

test "the joint call receives the two separate calls' arguments (N 1..4, n 5..7, LoRA on and off)" {
    for (1..5) |N| for (5..8) |n| {
        const R = N * n;
        for ([_]bool{ false, true }) |lora| {
            // The LoRA dimension is asserted-independent by construction: the adapter's terms land in dr.qa / dr.ykv
            // before the norms and read dr.qr at wq_b, but the norm launches' arguments do not mention it - so both
            // iterations must capture identical launches (and this model must track draft.zig:751-762 exactly). The
            // LoRA-dependent property (every reader after the joint) is the order test below.
            var log_sep: tri.Log = .{ .gpa = std.testing.allocator };
            defer log_sep.deinit();
            var log_joint: tri.Log = .{ .gpa = std.testing.allocator };
            defer log_joint.deinit();
            for (0..STAGES) |j| {
                try separate(.{ .log = &log_sep }, stage(j), R);
                try joint(.{ .log = &log_joint }, stage(j), R);
            }
            try std.testing.expectEqual(@as(usize, 2 * STAGES), log_sep.items.items.len);
            try std.testing.expectEqual(@as(usize, STAGES), log_joint.items.items.len);
            for (0..STAGES) |j| {
                const q = log_sep.items.items[2 * j];
                const kv = log_sep.items.items[2 * j + 1];
                const jn = log_joint.items.items[j];
                try std.testing.expectEqualStrings("_rmsnorm_rot", q.name);
                try std.testing.expectEqualStrings("_kv_norm_rope", kv.name);
                try std.testing.expectEqualStrings("_q_kv_norm", jn.name);
                // the grid: one program a row and role (progs(1) = 3: OUT, kv, the one rotation)
                try std.testing.expectEqual(@as(u32, @intCast(R)), jn.grid[0]);
                try std.testing.expectEqual(@as(u32, @intCast(dq.progs(1))), jn.grid[1]);
                try std.testing.expectEqual(@as(u32, 1), jn.grid[2]);
                try std.testing.expectEqual(@as(u32, @intCast(R)), q.grid[0]);
                try std.testing.expectEqual(@as(u32, @intCast(R)), kv.grid[0]);
                // every joint argument is the separate launch's it sources from (draft_qkv_norm.arg_source); the
                // one both launches pass, eps, is compared on the kv side too
                for (dq.arg_source) |src| {
                    const want = arg(if (src.arm == .q) q else kv, src.name);
                    try std.testing.expect(argEq(arg(jn, src.joint), want));
                }
                try std.testing.expect(argEq(arg(jn, "eps"), arg(kv, "eps")));
                // the consts: q's and kv's, kv's width as DK, and SPLIT on (one rotation a program)
                for ([_][]const u8{ "BLOCK", "D", "NROT", "PDL" }) |name| {
                    try std.testing.expectEqual(constOf(q, name).?.int, constOf(jn, name).?.int);
                }
                for ([_][]const u8{ "QUANT", "RD", "PDL" }) |name| {
                    try std.testing.expectEqual(constOf(kv, name).?.int, constOf(jn, name).?.int);
                }
                try std.testing.expectEqual(constOf(kv, "D").?.int, constOf(jn, "DK").?.int);
                try std.testing.expectEqual(@as(?i64, 1), constOf(jn, "SPLIT").?.int);
                try std.testing.expectEqual(@as(?i64, 1), constOf(jn, "NROT").?.int);
                try std.testing.expectEqual(@as(?i64, D), constOf(jn, "D").?.int);
                try std.testing.expectEqual(@as(?i64, HD), constOf(jn, "DK").?.int);
                try std.testing.expectEqual(@as(?i64, RD), constOf(jn, "RD").?.int);
                // widths and strides: the q rows' stride (xs, os_) and the KV width both arms pass
                try std.testing.expectEqual(@as(i32, D), arg(jn, "xs").value.i32);
                try std.testing.expectEqual(@as(i32, D), arg(jn, "os_").value.i32);
                try std.testing.expectEqual(@as(i32, SLOTS * RING), arg(jn, "ring_size").value.i32);
            }
            std.debug.print("N={d} n={d} rows={d} lora={}: {d} joint launches, arguments equal the {d} separate launches'\n", .{ N, n, R, lora, STAGES, 2 * STAGES });
        }
    };
}

test "the -1 sentinel's slot list is the one the joint call receives and writes no ring row" {
    const sent: [MAX_ROWS]i64 = @splat(-1); // draft.zig's row(5): hv[5 * max_rows + r] = -1, every block row
    for (1..5) |N| for (5..8) |n| {
        const R = N * n;
        var log_sep: tri.Log = .{ .gpa = std.testing.allocator };
        defer log_sep.deinit();
        var log_joint: tri.Log = .{ .gpa = std.testing.allocator };
        defer log_joint.deinit();
        for (0..STAGES) |j| {
            try separate(.{ .log = &log_sep }, stage(j), R);
            try joint(.{ .log = &log_joint }, stage(j), R);
        }
        for (0..STAGES) |j| {
            const kv = log_sep.items.items[2 * j + 1];
            const jn = log_joint.items.items[j];
            // the same slot list (the same buffer) in both arms
            try std.testing.expectEqual(arg(kv, "SLOT_OF").value.ptr.addr, arg(jn, "SLOT_OF").value.ptr.addr);
            try std.testing.expectEqual(arg(kv, "RING").value.ptr.addr, arg(jn, "RING").value.ptr.addr);
            try std.testing.expectEqual(arg(kv, "ring_size").value.i32, arg(jn, "ring_size").value.i32);
        }
        var writes: usize = 0;
        for (sent[0..R]) |slot| {
            writes += @intFromBool(dq.ringWrite(slot));
        }
        try std.testing.expectEqual(@as(usize, 0), writes);
    };
}

test "the joint launch is the recorded _q_kv_norm variant class (kit dispatch at draft row counts)" {
    // The kit variant resolves on constexprs and the arguments' specialization (cuda/aot.zig matches/fits take no
    // grid), so the drafter's row counts dispatch whenever its launch is the recorded one's class. At the served
    // shape its scalars are exactly the fixture's (xs / os_ / ring_size 1280 / 1280 / 576).
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, tri.fixtures, .{});
    defer parsed.deinit();
    const cases = parsed.value.object.get("_q_kv_norm").?.array.items;
    var seen: usize = 0;
    for (cases) |case| {
        const consts = case.object.get("consts").?.object;
        if (consts.get("NROT").?.integer != 1) continue; // the drafter's: wq_b's rotation alone
        seen += 1;
        var log: tri.Log = .{ .gpa = std.testing.allocator };
        defer log.deinit();
        const s = stage(0);
        const R = 4 * 7;
        try joint(.{ .log = &log }, s, R);
        const jn = log.items.items[0];
        // the constexprs: the same names and values as recorded
        try std.testing.expectEqual(consts.count(), jn.consts.len);
        var it = consts.iterator();
        while (it.next()) |e| {
            const want: i64 = switch (e.value_ptr.*) {
                .integer => |x| x,
                .bool => |b| @intFromBool(b),
                else => unreachable,
            };
            try std.testing.expectEqual(want, constOf(jn, e.key_ptr.*).?.int);
        }
        // the arguments: the recorded tensors' and scalars' names and dtypes, and the scalars' values
        const rec = case.object.get("tensors").?.object;
        const scal = case.object.get("scalars").?.object;
        try std.testing.expectEqual(rec.count() + scal.count(), jn.args.len);
        var nit = rec.iterator();
        while (nit.next()) |e| {
            const dtype = e.value_ptr.array.items[0].string;
            const ty = if (std.mem.eql(u8, dtype, "bfloat16"))
                "*bf16"
            else if (std.mem.eql(u8, dtype, "float16"))
                "*fp16"
            else if (std.mem.eql(u8, dtype, "float32"))
                "*fp32"
            else
                "*i64";
            try std.testing.expectEqualStrings(ty, arg(jn, e.key_ptr.*).value.ptr.ty);
        }
        for ([_][]const u8{ "xs", "os_", "ring_size" }) |name| {
            try std.testing.expectEqual(@as(i32, @intCast(scal.get(name).?.integer)), arg(jn, name).value.i32);
        }
        try std.testing.expectEqual(@as(i64, 1280), scal.get("xs").?.integer); // X's row stride (query width)
        try std.testing.expectEqual(@as(i64, 1280), scal.get("os_").?.integer); // OUT's row stride
        try std.testing.expectEqual(@as(i64, SLOTS * RING), scal.get("ring_size").?.integer); // the served pool plane
        // the grid: the recorded programs a row (its rows are the decoder's window; the drafter's are N * n)
        const grid = case.object.get("grid").?.array.items;
        try std.testing.expectEqual(@as(u32, @intCast(R)), jn.grid[0]);
        try std.testing.expectEqual(@as(u32, @intCast(grid[1].integer)), jn.grid[1]);
        try std.testing.expectEqual(@as(u32, @intCast(grid[2].integer)), jn.grid[2]);
        std.debug.print("_q_kv_norm record {s}: the drafter's joint launch at {d} rows matches its consts, dtypes and scalars\n", .{ case.object.get("phase").?.string, R });
    }
    try std.testing.expect(seen > 0); // the fixture must hold the one-rotation record or this passes vacuously
}

/// Drafter.attention's launch order as each arm runs it (draft.zig's calls) as buffers: every read must follow that
/// buffer's write. x and ring (the absorbed window rows) are the stage's inputs.
const Step = struct { name: []const u8, reads: []const []const u8, writes: []const []const u8 };

fn armSteps(out: []Step, joint_arm: bool, lora: bool) []Step {
    var n: usize = 0;
    out[n] = .{ .name = "attn_in", .reads = &.{"x"}, .writes = &.{ "qa", "ykv" } };
    n += 1;
    if (lora) {
        out[n] = .{ .name = "lora_wq_a", .reads = &.{"x"}, .writes = &.{"qa"} };
        n += 1;
        out[n] = .{ .name = "lora_wkv", .reads = &.{"x"}, .writes = &.{"ykv"} };
        n += 1;
    }
    if (joint_arm) {
        out[n] = .{ .name = "q_kv_norm", .reads = &.{ "qa", "ykv" }, .writes = &.{ "qr", "xhq", "kvb" } };
        n += 1;
    } else {
        out[n] = .{ .name = "rmsnorm_rot", .reads = &.{"qa"}, .writes = &.{ "qr", "xhq" } };
        n += 1;
    }
    out[n] = .{ .name = "wq_b", .reads = &.{"xhq"}, .writes = &.{"q"} };
    n += 1;
    if (lora) {
        out[n] = .{ .name = "lora_wq_b", .reads = &.{"qr"}, .writes = &.{"q"} };
        n += 1;
    }
    if (!joint_arm) {
        out[n] = .{ .name = "kv_norm_rope", .reads = &.{"ykv"}, .writes = &.{"kvb"} };
        n += 1;
    }
    out[n] = .{ .name = "sparse_attn", .reads = &.{ "q", "kvb", "ring" }, .writes = &.{"o"} };
    n += 1;
    return out[0..n];
}

test "the swap keeps every read after its write, with and without LoRA" {
    for ([_]bool{ false, true }) |joint_arm| for ([_]bool{ false, true }) |lora| {
        var buf: [8]Step = undefined;
        const steps = armSteps(&buf, joint_arm, lora);
        for (steps, 0..) |s, i| {
            for (s.reads) |r| {
                if (std.mem.eql(u8, r, "x") or std.mem.eql(u8, r, "ring")) continue; // the stage's inputs
                var written = false;
                for (steps[0..i]) |prev| {
                    for (prev.writes) |w| {
                        if (std.mem.eql(u8, w, r)) written = true;
                    }
                }
                try std.testing.expect(written); // else a read before its write: the swap broke the order
            }
        }
        var norms: usize = 0;
        for (steps) |s| {
            if (std.mem.indexOf(u8, s.name, "norm") != null) norms += 1;
        }
        try std.testing.expectEqual(if (joint_arm) @as(usize, 1) else 2, norms);
    };
}

test "launches per pass at N 1..4, n 5..7: two a block before, one after" {
    for (1..5) |N| for (5..8) |n| {
        const R = N * n;
        var log_sep: tri.Log = .{ .gpa = std.testing.allocator };
        defer log_sep.deinit();
        var log_joint: tri.Log = .{ .gpa = std.testing.allocator };
        defer log_joint.deinit();
        for (0..STAGES) |j| {
            try separate(.{ .log = &log_sep }, stage(j), R);
            try joint(.{ .log = &log_joint }, stage(j), R);
        }
        try std.testing.expectEqual(@as(usize, 2 * STAGES), log_sep.items.items.len);
        try std.testing.expectEqual(@as(usize, STAGES), log_joint.items.items.len);
        std.debug.print("N={d} n={d} rows={d}: norm launches a pass {d} -> {d} ({d} DSpark blocks)\n", .{ N, n, R, log_sep.items.items.len, log_joint.items.items.len, STAGES });
    };
}
