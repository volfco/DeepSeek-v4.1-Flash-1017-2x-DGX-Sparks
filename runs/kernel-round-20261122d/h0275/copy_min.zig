//! TF_DS_COPY_MIN (unset: today's minimum): the copy draft's minimum match length - the fewest tokens a copy window
//! (lanes/windows.zig `copyProposal`'s `copied.len`) needs to take a round from the draft head. Pure: no cuda import;
//! native.zig's open parses it beside TF_DS_NO_COPIES and the value reaches the gate as Model.copy_min. Unset, "" or
//! "0" keeps today's hard-coded `default_min`; an integer in `default_min`..=`max_min` sets the minimum; anything else
//! prints `<env_name>: expected 0 or an integer 2..7` to stderr and refuses startup (fromText's error).
const std = @import("std");

pub const env_name: [:0]const u8 = "TF_DS_COPY_MIN";

/// Today's hard-coded gate (windows.copyProposal: `copied.len < 2`): the fewest tokens in a copy window.
pub const default_min: u32 = 2;
/// The widest a copy can be: `window_rows - 1` tokens (lanes.zig's `window_rows`, draft.max_draft_block = 7), so a
/// minimum above `max_min` never takes a round. Refused rather than a silent copy-off: TF_DS_NO_COPIES=1 is that switch.
pub const max_min: u32 = 7;

/// The switch's value: unset, "" or "0" off (`default_min`), an integer `default_min`..=`max_min` on, else a refusal.
pub fn parse(text: ?[]const u8) ?u32 {
    const t = text orelse return default_min;
    if (t.len == 0 or std.mem.eql(u8, t, "0")) return default_min;
    if (std.fmt.parseInt(u32, t, 10)) |n| {
        if (n >= default_min and n <= max_min) return n;
    } else |_| {}
    std.debug.print("{s}: expected 0 or an integer {d}..{d}\n", .{ env_name, default_min, max_min });
    return null;
}

pub fn fromText(text: ?[]const u8) error{BadCopyMin}!u32 {
    return parse(text) orelse error.BadCopyMin;
}

/// The gate's decision (windows.copyProposal's rule, on the host): a copy of `len` tokens takes the round at minimum
/// `set` (null: `default_min`). The context match beside it keeps its own Config.enter_match gate, unchanged.
pub fn takes(len: usize, set: ?u32) bool {
    return len >= (set orelse default_min);
}

test "unset keeps today's minimum and a set value overrides it" {
    try std.testing.expectEqual(@as(?u32, default_min), parse(null));
    try std.testing.expectEqual(@as(?u32, default_min), parse(""));
    try std.testing.expectEqual(@as(?u32, default_min), parse("0"));
    try std.testing.expectEqual(@as(?u32, 2), parse("2"));
    try std.testing.expectEqual(@as(?u32, 3), parse("3"));
    try std.testing.expectEqual(@as(?u32, max_min), parse("7"));
}

test "a malformed or out-of-range minimum refuses startup" {
    for ([_][]const u8{ "1", "8", "65", "-1", "3.0", "abc", " 3" }) |t| {
        try std.testing.expectEqual(@as(?u32, null), parse(t));
        try std.testing.expectError(error.BadCopyMin, fromText(t));
    }
}

test "a copy takes the round at the minimum or more" {
    try std.testing.expect(takes(2, null)); // today's gate
    try std.testing.expect(!takes(1, null));
    try std.testing.expect(takes(3, 3));
    try std.testing.expect(!takes(2, 3)); // n=3 leaves two-token copies to the draft head
    try std.testing.expect(takes(7, 7));
    try std.testing.expect(!takes(6, 7));
}
