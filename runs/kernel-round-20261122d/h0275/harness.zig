//! h0275 — CPU emulation of the copy-draft proposal under TF_DS_COPY_MIN (BRIEF-0275).
//!
//! Emulation side (the code under test, extracted from the worktree by build.sh): the real
//! SuffixLookup.propose (core/lanes/proposer.zig) feeding the real gate rule of the patch
//! (copy_min.takes + windows.copyProposal's Config.enter_match check).
//!
//! Reference side (written from BRIEF-0275's spec, NOT from the kernel): a suffix-lookup copy proposes the
//! continuation that followed the longest earlier occurrence of the context's tail (the match reaching back at
//! most max_extension tokens; ties take the latest occurrence), at most `max_draft` tokens, only at a match of at
//! least min_match tokens; a copy takes the round from the draft head at `n` tokens or more and a match of at
//! least enter_match. Unset means n = 2 (today's hard-coded gate).
//!
//! Everything runs sequentially on the host (one context at a time; no threads, no GPU).
const std = @import("std");
const copy_min = @import("copy_min.zig");
const prop = @import("proposer.zig");

const enter_match: i64 = 8; // Config.enter_match's served default (dsv41 facts() leaves it)
const min_match: i64 = 4; // the served SuffixLookup params (lane_host.min_match; lanes_main's too)
const max_extension: i64 = 64; // Params.max_extension's default
const most: i64 = 6; // the served copy width (win_rows = 7 at TF_DS_DRAFT_BLOCK=6: first_copy = min(max_copy 6, 6))

var failures: usize = 0;

fn check(ok: bool, what: []const u8) void {
    if (!ok) {
        failures += 1;
        std.debug.print("MISMATCH: {s}\n", .{what});
    }
}

// -- independent reference ------------------------------------------------------------------------------------------

const Ref = struct {
    match: i64 = 0,
    from: usize = 0, // the earlier occurrence's end (0: no occurrence)

    fn copied(r: Ref, ctx: []const u32, max_draft: i64) []const u32 {
        if (r.from == 0 or r.match < min_match or max_draft <= 0) return &.{};
        return ctx[r.from..@min(ctx.len, r.from + @as(usize, @intCast(max_draft)))];
    }

    /// The gate from the brief: takes the round at n tokens or more (unset: 2) and a match of enter_match or more.
    fn takes(r: Ref, ctx: []const u32, max_draft: i64, n: ?u32) bool {
        const min = n orelse 2;
        const c = r.copied(ctx, max_draft);
        return c.len >= min and r.match >= enter_match;
    }
};

/// The longest earlier occurrence of the context's tail: every end a match could stop at, latest first (a strict
/// first-win keeps the latest of equal matches), each match bounded by max_extension and the occurrence's own start.
fn refProposal(ctx: []const u32) Ref {
    var r: Ref = .{};
    var end: usize = ctx.len;
    while (end > 1) {
        end -= 1;
        const limit = @min(@as(usize, @intCast(max_extension)), end);
        var l: usize = 0;
        while (l < limit and ctx[end - 1 - l] == ctx[ctx.len - 1 - l]) l += 1;
        if (@as(i64, @intCast(l)) > r.match) {
            r.match = @intCast(l);
            r.from = end;
        }
    }
    return r;
}

// -- emulation (the worktree's real code) ---------------------------------------------------------------------------

const Emu = struct {
    s: prop.SuffixLookup,

    fn init() !Emu {
        return .{ .s = try prop.SuffixLookup.init(std.heap.page_allocator, .{ .min_match = min_match }) };
    }

    fn deinit(e: *Emu) void {
        e.s.deinit();
    }

    fn proposal(e: *Emu, ctx: []const u32, max_draft: i64) ![]const u32 {
        return e.s.propose(ctx, max_draft);
    }

    /// windows.copyProposal's arm: propose, then the gate the patch made configurable.
    fn arm(e: *Emu, ctx: []const u32, max_draft: i64, n: ?u32) ![]const u32 {
        const copied = try e.s.propose(ctx, max_draft);
        if (!copy_min.takes(copied.len, n) or e.s.last_match < enter_match) return &.{};
        return copied;
    }
};

// -- the case matrix -----------------------------------------------------------------------------------------------

const Case = struct {
    name: []const u8,
    ctx: []const u32,
    max_draft: i64 = most,
    /// The copy length and match length the reference must find over the full context (0: no proposal).
    want_len: usize,
    want_match: i64,
};

fn repeat(comptime n: usize, comptime pat: []const u32) [n * pat.len]u32 {
    var out: [n * pat.len]u32 = undefined;
    for (0..n) |i| @memcpy(out[i * pat.len ..][0..pat.len], pat);
    return out;
}

const ab = repeat(12, &.{ 7, 9 }); // (ab)^12: 2-token copies
const triple = repeat(10, &.{ 3, 5, 8 }); // 3-token copies
const four = repeat(6, &.{ 21, 22, 23, 24 }); // 4-token copies
const five = repeat(5, &.{ 31, 32, 33, 34, 35 }); // 5-token copies
const six = repeat(4, &.{ 11, 12, 13, 14, 15, 16 }); // 6-token copies (the served width's ceiling)
const ones = repeat(16, &.{42}); // 1-token copies (today's gate already refuses these)
const prose = [_]u32{ 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2 }; // "the cat sat on the mat ." twice + "the cat"
const ascending = [_]u32{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123 }; // non-repetitive
const match7 = [_]u32{ 61, 62, 63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 61, 62, 63, 64, 65, 66, 67 }; // match 7: below enter_match
const match8 = [_]u32{ 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 61, 62, 63, 64, 65, 66, 67, 68 }; // match 8: at enter_match
const match9 = [_]u32{ 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 61, 62, 63, 64, 65, 66, 67, 68, 69 }; // match 9
const short_ctx = [_]u32{ 5, 6, 7 };

const cases = [_]Case{
    .{ .name = "(ab)^12 (2-token copies)", .ctx = &ab, .want_len = 2, .want_match = 22 },
    .{ .name = "(3,5,8)^10 (3-token copies)", .ctx = &triple, .want_len = 3, .want_match = 27 },
    .{ .name = "(21..24)^6 (4-token copies)", .ctx = &four, .want_len = 4, .want_match = 20 },
    .{ .name = "(31..35)^5 (5-token copies)", .ctx = &five, .want_len = 5, .want_match = 20 },
    .{ .name = "(11..16)^4 (6-token copies)", .ctx = &six, .want_len = 6, .want_match = 18 },
    .{ .name = "42^16 (1-token copies)", .ctx = &ones, .want_len = 1, .want_match = 15 },
    .{ .name = "prose repeat (6-token copy)", .ctx = &prose, .want_len = 6, .want_match = 8 },
    .{ .name = "ascending (non-repetitive)", .ctx = &ascending, .want_len = 0, .want_match = 0 },
    .{ .name = "anchor match 7 (gate rejects)", .ctx = &match7, .want_len = 6, .want_match = 7 },
    .{ .name = "anchor match 8 (gate accepts)", .ctx = &match8, .want_len = 6, .want_match = 8 },
    .{ .name = "anchor match 9", .ctx = &match9, .want_len = 6, .want_match = 9 },
    .{ .name = "short context (no proposal)", .ctx = &short_ctx, .want_len = 0, .want_match = 0 },
    .{ .name = "(11..16)^4 capped at max_draft 2", .ctx = &six, .max_draft = 2, .want_len = 2, .want_match = 18 },
    .{ .name = "(11..16)^4 at max_draft 0", .ctx = &six, .max_draft = 0, .want_len = 0, .want_match = 18 },
};

/// The n values every decision is compared under (null: unset).
const ns = [_]?u32{ null, 2, 3, 4, 5, 6, 7 };

fn sweep() !void {
    for (cases) |case| {
        // (1) the full-context pin: the reference must find exactly the constructed proposal
        const full = refProposal(case.ctx);
        const full_c = full.copied(case.ctx, case.max_draft);
        check(full_c.len == case.want_len and full.match == case.want_match, "pin");

        // (2) every growing prefix (a stream's rounds), emulation vs reference
        var emu = try Emu.init();
        defer emu.deinit();
        for (4..case.ctx.len + 1) |len| {
            const ctx = case.ctx[0..len];
            const got = try emu.proposal(ctx, case.max_draft);
            const got_len = got.len;
            const got_match = emu.s.last_match;
            const ref = refProposal(ctx);
            const ref_c = ref.copied(ctx, case.max_draft);
            if (got_len > 0 or ref_c.len > 0) {
                check(std.mem.eql(u32, got, ref_c), "proposal");
                if (got_len > 0) check(got_match == ref.match, "match");
            }
            // the same prefix through a fresh proposer (the index-rebuild path must agree)
            {
                var fresh = try Emu.init();
                defer fresh.deinit();
                const got2 = try fresh.proposal(ctx, case.max_draft);
                check(std.mem.eql(u32, got, got2), "rebuild");
            }
            // (3) the gate under every n, through the real arm (propose + copy_min.takes + enter_match)
            for (ns) |n| {
                const a = try emu.arm(ctx, case.max_draft, n);
                check((a.len > 0) == ref.takes(ctx, case.max_draft, n), "decision");
            }
            // (4) unset == today's hard-coded gate (`copied.len < 2`) at every prefix
            const today = got_len >= 2 and got_match >= enter_match;
            const unset = (try emu.arm(ctx, case.max_draft, copy_min.parse(null))).len > 0;
            check(unset == today, "unset==today");
        }
    }
}

// -- parse matrix and the n = 2 vs n = 3 mapping --------------------------------------------------------------------

fn parseMatrix() void {
    const rows = [_]struct { text: ?[]const u8, want: ?u32 }{
        .{ .text = null, .want = 2 },  .{ .text = "", .want = 2 },       .{ .text = "0", .want = 2 },
        .{ .text = "2", .want = 2 },   .{ .text = "3", .want = 3 },      .{ .text = "7", .want = 7 },
        .{ .text = "1", .want = null }, .{ .text = "8", .want = null },  .{ .text = "65", .want = null },
        .{ .text = "-1", .want = null }, .{ .text = "3.0", .want = null }, .{ .text = "abc", .want = null },
    };
    for (rows) |row| {
        const got = copy_min.parse(row.text);
        std.debug.print("parse({s}) = {?d} (want {?d})\n", .{ row.text orelse "unset", got, row.want });
        check(got == row.want, "parse");
        if (row.want == null) {
            const r = copy_min.fromText(row.text);
            check(if (r) |_| false else |err| err == error.BadCopyMin, "fromText refusal");
        }
    }
}

fn mapping() void {
    std.debug.print("\n{s:<32} {s:>4} {s:>6} {s:>11} {s:>5} {s:>5} {s:>5}\n", .{ "case", "copy", "match", "n=2 (today)", "n=3", "n=4", "n=7" });
    for (cases) |case| {
        const full = refProposal(case.ctx);
        const c = full.copied(case.ctx, case.max_draft);
        std.debug.print("{s:<32} {d:>4} {d:>6} {s:>11} {s:>5} {s:>5} {s:>5}\n", .{
            case.name,
            c.len,
            full.match,
            if (full.takes(case.ctx, case.max_draft, 2)) "copy round" else "head round",
            if (full.takes(case.ctx, case.max_draft, 3)) "copy" else "head",
            if (full.takes(case.ctx, case.max_draft, 4)) "copy" else "head",
            if (full.takes(case.ctx, case.max_draft, 7)) "copy" else "head",
        });
    }
}

pub fn main() !void {
    std.debug.print("h0275: copy-draft proposal under TF_DS_COPY_MIN (served: min_match {d}, enter_match {d}, max_draft {d})\n\n", .{ min_match, enter_match, most });
    parseMatrix();
    std.debug.print("\n", .{});
    try sweep();
    mapping();
    if (failures > 0) {
        std.debug.print("\n{d} mismatches\n", .{failures});
        return error.Mismatch;
    }
    std.debug.print("\nall cases match the independent reference; unset == today\n", .{});
}
