//! TF_DS_PROMPT_PICKS_ONCE (off by default): the prompt attention's pick list padded once per selection - a layer
//! whose picks are the same selection as the previous padded list reuses it (no fill, no copy), and any host-known
//! event that changes the picks (an indexer's output, a replay cut's rows, a new chunk) re-pads. Pure: no cuda
//! import; the launcher calls are prompt.zig's attention (the cidx -> cidxp pad ahead of prompt sparse attention),
//! the events its indexer, replayCut and begin. The validity is decided from those host events alone, never from
//! comparing device buffers. A malformed switch refuses startup.
const std = @import("std");

pub const env_name: [:0]const u8 = "TF_DS_PROMPT_PICKS_ONCE";

/// The switch's value: unset, "" or "0" off; "1" on; anything else is a refusal to start (the message is on stderr).
pub fn parse(text: ?[]const u8) ?bool {
    const t = text orelse return false;
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return false;
    if (std.mem.eql(u8, t, "1")) return true;
    std.debug.print("{s}: expected 1 or 0\n", .{env_name});
    return null;
}

pub fn fromText(text: ?[]const u8) error{BadPromptPicksOnce}!bool {
    return parse(text) orelse error.BadPromptPicksOnce;
}

// -- host reference -------------------------------------------------------------------------------------------------

/// The padded row stride (sparse_attn's F.pad rule: the pick list padded with -1 to whole 16-column tiles).
pub fn padWidth(kk: usize) usize {
    return (kk + 15) / 16 * 16;
}

/// The padded list the attention reads (rows x padWidth(kk) int64): each row its kk picks, then -1 in the padding
/// columns (the fill's 0xff bytes). The picks' own list keeps its compact kk stride (the sharded fetch reads it).
pub fn padList(cidx: []const i64, rows: usize, kk: usize, out: []i64) void {
    const kp = padWidth(kk);
    std.debug.assert(cidx.len >= rows * kk and out.len >= rows * kp);
    for (0..rows) |r| {
        @memcpy(out[r * kp ..][0..kk], cidx[r * kk ..][0..kk]);
        @memset(out[r * kp + kk ..][0..kp - kk], -1);
    }
}

/// The host-known validity of the padded list: what its bytes were made from, and whether a layer may read them
/// again. `stale` at every event that changes the picks (an indexer writes cidx), the rows (a replay cut rewrites
/// the chunk) or the chunk (begin); `held` records what the pad then made.
pub const Tracker = struct {
    holds: bool = false, // cidxp holds a padded list of rows x kk's picks
    rows: usize = 0,
    kk: usize = 0,

    /// The picks or the rows changed: the padded list no longer describes this layer's read.
    pub fn stale(t: *Tracker) void {
        t.holds = false;
    }

    /// Whether the padded list made for this rows x kk still holds this layer's picks.
    pub fn reusable(t: *const Tracker, rows: usize, kk: usize) bool {
        return t.holds and t.rows == rows and t.kk == kk;
    }

    /// The padded list now holds rows x kk's picks.
    pub fn held(t: *Tracker, rows: usize, kk: usize) void {
        t.holds = true;
        t.rows = rows;
        t.kk = kk;
    }
};

// -- tests ----------------------------------------------------------------------------------------------------------

test "the switch parses its forms and nothing else" {
    try std.testing.expectEqual(@as(?bool, false), parse(null));
    try std.testing.expectEqual(@as(?bool, false), parse(""));
    try std.testing.expectEqual(@as(?bool, false), parse("0"));
    try std.testing.expectEqual(@as(?bool, true), parse("1"));
    try std.testing.expectEqual(@as(?bool, null), parse("2"));
    try std.testing.expectEqual(@as(?bool, null), parse("on"));
    try std.testing.expectEqual(@as(?bool, null), parse("01"));
    try std.testing.expectEqual(@as(bool, true), try fromText("1"));
    try std.testing.expectError(error.BadPromptPicksOnce, fromText("2"));
}

test "padWidth pads to whole 16-column tiles" {
    try std.testing.expectEqual(@as(usize, 16), padWidth(1));
    try std.testing.expectEqual(@as(usize, 16), padWidth(16));
    try std.testing.expectEqual(@as(usize, 32), padWidth(17));
    try std.testing.expectEqual(@as(usize, 64), padWidth(63)); // the served cold-prompt pick count
    try std.testing.expectEqual(@as(usize, 64), padWidth(64));
    try std.testing.expectEqual(@as(usize, 128), padWidth(127));
    try std.testing.expectEqual(@as(usize, 512), padWidth(512)); // index_topk, already whole
}

test "padList: the kk picks per row, then -1 to the tile" {
    const cidx = [_]i64{ 10, 11, 12, 20, 21, 22 }; // 2 rows x 3 picks
    var out: [32]i64 = @splat(0);
    padList(&cidx, 2, 3, &out);
    var want: [32]i64 = @splat(-1); // 2 rows x padWidth(3) = 16 columns of -1, the picks in front
    want[0] = 10;
    want[1] = 11;
    want[2] = 12;
    want[16] = 20;
    want[17] = 21;
    want[18] = 22;
    try std.testing.expectEqualSlices(i64, &want, &out);
}

test "the tracker reuses a pad until an event or a shape change" {
    var t: Tracker = .{};
    try std.testing.expect(!t.reusable(127, 63)); // nothing padded yet
    t.held(127, 63);
    try std.testing.expect(t.reusable(127, 63)); // layers 3-7 of a cold 127-token prompt read it again
    try std.testing.expect(!t.reusable(127, 64)); // a layer's own kk differs: re-pad
    try std.testing.expect(!t.reusable(63, 63)); // a replay cut's rows differ: re-pad
    t.stale(); // an indexer's output
    try std.testing.expect(!t.reusable(127, 63));
    t.held(127, 63);
    t.stale(); // a replay cut's rows (even when the row count happens to match)
    try std.testing.expect(!t.reusable(127, 63));
}

test "the served cold 127-token prompt's pad count" {
    // the layer map (config.zig's served ratios): layers 2-19 ratio 2, 20-39 ratio 1; the index sources' indexer
    // rewrites the picks, every other compressed layer shares the latest selection (prompt.zig's attention)
    const index_sources = [_]usize{ 2, 8, 14, 20, 24, 28, 32, 36 };
    var t: Tracker = .{};
    var sh_kk: ?usize = null; // Shared.kk: the latest indexer's selection
    var pads: usize = 0;
    var reuses: usize = 0;
    for (0..40) |li| {
        const r: usize = if (li < 2) 0 else if (li < 20) 2 else 1;
        if (r == 0) continue;
        for (index_sources) |s| if (s == li) {
            t.stale();
            sh_kk = @min(@as(usize, 512), 127 / r); // n_comp_end = (start + n) / ratio at a cold start 0
        };
        const n: usize = 127;
        const kk = sh_kk.?;
        if (kk % 16 != 0 and n > 16) {
            if (t.reusable(n, kk)) reuses += 1 else {
                t.held(n, kk);
                pads += 1;
            }
        }
    }
    try std.testing.expectEqual(@as(usize, 8), pads); // one pad per indexer selection
    try std.testing.expectEqual(@as(usize, 30), reuses); // 38 padded reads today: 30 fills and 30 copies saved
}
