//! TF_DS_DRAFT_QKV_NORM (off by default): drafter attention's query and KV normalization as the decoder's one
//! qKvNorm launch - each DSpark block's query RMSNorm + wq_b input rotation (_rmsnorm_rot) and KV RMSNorm + RoPE +
//! FP8 round trip (_kv_norm_rope) run as the _q_kv_norm kit launch the decoder makes (round.zig, tri_norm.qKvNorm):
//! one launch a block instead of two, three a served pass. Pure: no cuda import; the launcher is tri_norm.qKvNorm and
//! the call site Drafter.attention (draft.zig), whose two separate launches stay the fallback arm. The block rows
//! pass the -1 slot sentinel (draft.zig's row(5)); _q_kv_norm's kv program guards its ring write exactly as
//! _kv_norm_rope does (kernels.py's `if slot >= 0`), so the ring is untouched. A malformed switch refuses startup.
const std = @import("std");

pub const env_name: [:0]const u8 = "TF_DS_DRAFT_QKV_NORM";

/// The switch's value: unset, "" or "0" off; "1" on; anything else is a refusal to start (the message is on stderr).
pub fn parse(text: ?[]const u8) ?bool {
    const t = text orelse return false;
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "1")) return true;
    std.debug.print("{s}: expected 1 or 0\n", .{env_name});
    return null;
}

pub fn fromText(text: ?[]const u8) error{BadDraftQkvNorm}!bool {
    return parse(text) orelse error.BadDraftQkvNorm;
}

/// tri_norm.qKvNorm's width bound (its RowTooWide); wider query rows keep the two separate launches.
pub const max_d = 2048;

/// _q_kv_norm's grid.y at the served split (qkv_split on in the served build): the norm's OUT row, the kv program,
/// and one program a input rotation (its norm recomputed, the same bits).
pub fn progs(nrot: usize) usize {
    return 2 + nrot;
}

/// _q_kv_norm's kv program's ring rule (_kv_norm_rope's own): ring row `slot` is written only at slot >= 0. The
/// drafter's block rows pass -1 (draft.zig's row(5)), so their launch leaves the ring untouched.
pub fn ringWrite(slot: i64) bool {
    return slot >= 0;
}

/// The argument equivalence the joint call must hold (h0268's host check): every argument of each separate launch
/// (q = _rmsnorm_rot's, kv = _kv_norm_rope's) and the name it takes in _q_kv_norm's launch. The two renames are kv's
/// weight (W -> WK) and output (OUT -> KOUT); eps is the one argument both launches pass.
pub const Arm = enum { q, kv };

pub const arg_source = [_]struct { arm: Arm, name: []const u8, joint: []const u8 }{
    .{ .arm = .q, .name = "X", .joint = "X" },
    .{ .arm = .q, .name = "xs", .joint = "xs" },
    .{ .arm = .q, .name = "W", .joint = "W" },
    .{ .arm = .q, .name = "OUT", .joint = "OUT" },
    .{ .arm = .q, .name = "os_", .joint = "os_" },
    .{ .arm = .q, .name = "eps", .joint = "eps" },
    .{ .arm = .q, .name = "S0", .joint = "S0" },
    .{ .arm = .q, .name = "H0", .joint = "H0" },
    .{ .arm = .q, .name = "S1", .joint = "S1" },
    .{ .arm = .q, .name = "H1", .joint = "H1" },
    .{ .arm = .kv, .name = "Y", .joint = "Y" },
    .{ .arm = .kv, .name = "W", .joint = "WK" },
    .{ .arm = .kv, .name = "COS", .joint = "COS" },
    .{ .arm = .kv, .name = "SIN", .joint = "SIN" },
    .{ .arm = .kv, .name = "POS", .joint = "POS" },
    .{ .arm = .kv, .name = "OUT", .joint = "KOUT" },
    .{ .arm = .kv, .name = "RING", .joint = "RING" },
    .{ .arm = .kv, .name = "SLOT_OF", .joint = "SLOT_OF" },
    .{ .arm = .kv, .name = "ring_size", .joint = "ring_size" },
};

test "the switch parses its forms and nothing else" {
    try std.testing.expectEqual(false, parse(null).?);
    try std.testing.expectEqual(false, parse("").?);
    try std.testing.expectEqual(false, parse("0").?);
    try std.testing.expectEqual(true, parse("1").?);
    try std.testing.expectEqual(@as(?bool, null), parse("2"));
    try std.testing.expectEqual(@as(?bool, null), parse("yes"));
    try std.testing.expectEqual(@as(?bool, null), parse("01"));
    try std.testing.expectEqual(@as(bool, true), try fromText("1"));
    try std.testing.expectError(error.BadDraftQkvNorm, fromText("2"));
}

test "the joint launch's arguments are the two separate launches' (its spec)" {
    // _rmsnorm_rot's 10 and _kv_norm_rope's 10 arguments, WK / KOUT the renames and eps shared: the joint launch's
    // 19 names are exactly their targets
    const names = [_][]const u8{ "X", "xs", "W", "OUT", "os_", "eps", "S0", "H0", "S1", "H1", "Y", "WK", "COS", "SIN", "POS", "KOUT", "RING", "SLOT_OF", "ring_size" };
    try std.testing.expectEqual(names.len, arg_source.len);
    for (names) |n| {
        var hits: usize = 0;
        for (arg_source) |s| {
            if (std.mem.eql(u8, s.joint, n)) hits += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), hits);
    }
    // the kv side's weight and output are the only renames
    var renames: usize = 0;
    for (arg_source) |s| {
        if (std.mem.eql(u8, s.name, s.joint)) continue;
        try std.testing.expectEqual(Arm.kv, s.arm);
        try std.testing.expect(std.mem.eql(u8, s.joint, "WK") or std.mem.eql(u8, s.joint, "KOUT"));
        renames += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), renames);
}

test "_q_kv_norm's program count is the recorded grids'" {
    try std.testing.expectEqual(@as(usize, 3), progs(1)); // fixtures' _q_kv_norm NROT 1: grid [16, 3, 1]
    try std.testing.expectEqual(@as(usize, 4), progs(2)); // fixtures' NROT 2 (the indexer's wq_b): [16, 4, 1]
}

test "the -1 slot sentinel writes no ring row at any served pass shape" {
    for (1..5) |N| for (5..8) |n| {
        var writes: usize = 0;
        for (0..N * n) |r| {
            writes += @intFromBool(ringWrite(-1 - @as(i64, @intCast(r % 3))));
        }
        try std.testing.expectEqual(@as(usize, 0), writes);
    };
    // a non-negative slot list (the absorbs') writes every row
    for (0..4) |r| try std.testing.expect(ringWrite(@intCast(r)));
}
