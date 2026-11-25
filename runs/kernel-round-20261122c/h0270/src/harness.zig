//! h0270: TF_DS_PROMPT_PICKS_ONCE's host check (BRIEF-0270): the padded pick list a prompt layer's attention reads
//! under the reuse arm against an independently written off reference, byte for byte, over the brief's case matrix:
//! a cold 127-token prompt, multi-chunk prompts under TF_DS_PREFILL_SLICE=5, a replay cut mid-prompt, and pick
//! counts 1..64 and >64. There is no new kernel in this patch (it only skips launches), so instead of a kernel
//! extraction this harness emulates the two pad ops the arm skips - prompt.fill (cuMemsetD8Async of 0xff) and
//! ops.copyRows - exactly as prompt.zig's attention calls them, and runs the worktree's real prompt_picks_once.zig
//! (Tracker/padWidth) as the reuse decision. The off reference is written from BRIEF-0270's spec alone ("a prompt
//! chunk's pick list padded with -1 to whole 16-column tiles", re-padded at every layer), elementwise, sharing no
//! code with the emulated ops. The event sequence is the real structure: the served layer map (config.zig's ratios
//! and index sources), prompt.zig's indexer/replayCut/begin hooks and model.zig's cut point, in layer slices.
const std = @import("std");
const picks = @import("prompt_picks_once.zig");

// -- the served checkpoint's host-known structure (config.zig's served JSON) ----------------------------------------

const layers_n = 40;
const index_sources = [_]usize{ 2, 8, 14, 20, 24, 28, 32, 36 }; // index_source_layer_ids: the indexer's layers
fn ratio(li: usize) usize {
    return if (li < 2) 0 else if (li < 20) 2 else 1; // compress_ratios: layers 2..19 at 2, 20..39 at 1
}
const index_topk = 512; // config index_topk
const decode_rows = 16; // tri.decode_rows (TF_DS_DECODE_ROWS unset in the served build)
const ring_extra = 16; // prompt.ring_extra
const window = 128; // config sliding_window
const replay_floor_on = true; // model.zig sets TF_DS_REPLAY_FLOOR=1 on every rank
const cut_layer = 20; // model.zig: the replay cut at c.layers / 2
const cap = 256; // rows the harness models (the pad rules do not depend on the row count past it)

const Ev = enum { read_none, read_compact, pad, reuse };

/// Deterministic opaque picks (both arms read the same bytes; never -1, the padding's value).
fn pickVal(gen: u64, r: usize, c: usize) i64 {
    var x = gen *% 0x9E3779B97F4A7C15 +% @as(u64, @intCast(r)) *% 0xC2B2AE3D27D4EB4F +% @as(u64, @intCast(c)) *% 0x165667B19E3779F9;
    x ^= x >> 29;
    return @bitCast(x & 0x7FFF_FFFF_FFFF_FFFF);
}

/// The device bytes one model run moves: cidx (the picks, compact rows of kk) and cidxp (the padded list). The pad
/// ops are prompt.zig attention's exact calls - fill(cidxp, 0xff, n * kp * 8) (prompt.fill: cuMemsetD8Async, a byte
/// fill) then copyRows(cidx, kk*8, cidxp, kp*8, kk*8, n) - and the reuse decision is the worktree's real Tracker.
const Sim = struct {
    alloc: std.mem.Allocator,
    cidx: []i64, // [cap, index_topk]
    cidxp: []i64, // [cap, index_topk + 16]
    shadow: []i64, // cidx as the last indexer wrote it (the compact stride must survive every pad)
    on: bool,
    replay_floor: bool = replay_floor_on, // TF_DS_REPLAY_FLOOR (model.zig sets it on every rank; one corner row turns it off)
    tracker: picks.Tracker = .{},
    sh_kk: ?usize = null, // prompt.Shared.kk: the latest indexer's selection
    start: usize = 0,
    n: usize = 0,
    floor: usize = 0,
    replay: usize = 0,
    gen: u64 = 0, // the picks' content generation
    cur_kk: usize = 0, // the stride cidx's rows hold
    fills: usize = 0, // prompt.fill calls (the buffer fills BRIEF-0270 counts)
    copies: usize = 0, // ops.copyRows launches (the copy launches it counts)
    reuses: usize = 0, // skipped pads: the arm's saving
    bannered: bool = false,
    banner_rows: usize = 0,
    banner_kk: usize = 0,
    events: [layers_n + 8]Ev = undefined,
    n_events: usize = 0,

    fn init(alloc: std.mem.Allocator, on: bool) !Sim {
        return .{
            .alloc = alloc,
            .cidx = try alloc.alloc(i64, cap * index_topk),
            .cidxp = try alloc.alloc(i64, cap * (index_topk + 16)),
            .shadow = try alloc.alloc(i64, cap * index_topk),
            .on = on,
        };
    }

    fn deinit(s: *Sim) void {
        s.alloc.free(s.cidx);
        s.alloc.free(s.cidxp);
        s.alloc.free(s.shadow);
    }

    /// prompt.begin: a new chunk's picks, so its first pad runs (a new chunk re-pads).
    fn begin(s: *Sim, start: usize, n: usize, replay: usize) void {
        s.tracker.stale();
        s.sh_kk = null;
        s.start = start;
        s.n = n;
        s.floor = 0;
        s.replay = replay;
    }

    /// prompt.replayCut's row block (the tail prepend modeled by `pre`, tailPre/TF_DS_EXACT_RESUME). Only the
    /// block stale()s: BRIEF-0270's rule is "until ... a replay cut changes the rows", and a cut that rewrites
    /// nothing leaves the picks and rows the padded list describes.
    fn replayCut(s: *Sim, pre: usize) bool {
        const first = @max(s.start, s.replay) - s.start;
        if (first >= s.n) return false; // encoder-only: the chunk ends, no later layer reads
        if (first > 0 or pre > 0) {
            s.start = s.replay;
            s.n = s.n - first + pre;
            s.tracker.stale(); // the rows moved (even a same-count shift): the padded list re-pads
        }
        s.floor = s.replay;
        return true;
    }

    /// prompt.indexer's effect on the pad state: cidx rewritten on every arm (iotaVis or the scored selection),
    /// sh.kk set to the arm's min(index_topk, n_comp_end) (0: the n_comp_end == 0 arm reads none).
    fn indexer(s: *Sim, kk: usize) void {
        s.tracker.stale();
        s.sh_kk = kk;
        if (kk == 0) return;
        s.gen += 1;
        s.cur_kk = kk;
        for (0..s.n) |r| {
            for (0..kk) |c| s.cidx[r * kk + c] = pickVal(s.gen, r, c);
        }
        @memcpy(s.shadow, s.cidx);
    }

    /// prompt.zig attention's pick-pad block and the read it sets up (tri_attn.Attn's idx/n_idx: the padded list
    /// at padWidth(kk) rows, else cidx at the compact kk stride - the sharded fetch's stride).
    fn attention(s: *Sim) !Ev {
        const kk = s.sh_kk orelse return error.NoIndexerSelection;
        var ev: Ev = .read_none;
        if (kk != 0) {
            const ring_mode = s.n <= ring_extra and !(s.replay_floor and s.floor > s.start -| (window - 1));
            const one_split = !ring_mode and s.n <= decode_rows;
            if (kk % 16 != 0 and (s.n > decode_rows or one_split)) {
                const kp = picks.padWidth(kk);
                if (s.on and s.tracker.reusable(s.n, kk)) {
                    s.reuses += 1;
                    if (!s.bannered) {
                        s.bannered = true;
                        s.banner_rows = s.n;
                        s.banner_kk = kk;
                    }
                    ev = .reuse;
                } else {
                    @memset(std.mem.sliceAsBytes(s.cidxp)[0 .. s.n * kp * 8], 0xff); // prompt.fill(cidxp, 0xff, n * kp * 8)
                    for (0..s.n) |r| // ops.copyRows(cidx, kk*8, cidxp, kp*8, kk*8, n)
                        @memcpy(s.cidxp[r * kp ..][0..kk], s.cidx[r * kk ..][0..kk]);
                    s.tracker.held(s.n, kk);
                    s.fills += 1;
                    s.copies += 1;
                    ev = .pad;
                }
                try s.checkPadded(kk, kp);
            } else {
                try s.checkCompact(kk);
                ev = .read_compact;
            }
        }
        s.events[s.n_events] = ev;
        s.n_events += 1;
        return ev;
    }

    /// The byte-for-byte compare of a padded read (rows x padWidth(kk) of cidxp) against the independent
    /// reference, the -1 padding at the attention input, and the compact list's survival (the sharded fetch).
    fn checkPadded(s: *Sim, kk: usize, kp: usize) !void {
        try std.testing.expectEqual(@as(usize, (kk + 15) / 16 * 16), kp); // the spec's tile width, recomputed here
        const want = try s.alloc.alloc(i64, s.n * kp);
        defer s.alloc.free(want);
        refPadRead(s, kk, kp, want);
        try std.testing.expectEqualSlices(i64, want, s.cidxp[0 .. s.n * kp]);
        for (0..s.n) |r| for (kk..kp) |c| try std.testing.expectEqual(@as(i64, -1), s.cidxp[r * kp + c]);
        try std.testing.expectEqualSlices(i64, s.shadow, s.cidx);
    }

    /// A layer that pads nothing reads cidx rows at the compact kk stride (shardAttend/unionOf's stride).
    fn checkCompact(s: *Sim, kk: usize) !void {
        try std.testing.expectEqualSlices(i64, s.shadow, s.cidx);
        for (0..s.n) |r| for (0..kk) |c| try std.testing.expectEqual(s.shadow[r * kk + c], s.cidx[r * kk + c]);
    }
};

/// The off-path reference, written from BRIEF-0270's spec alone: each row its kk picks, then -1 to whole
/// 16-column tiles, re-padded at every layer. Elementwise; no shared code with the emulated fill/copy above.
fn refPadRead(s: *const Sim, kk: usize, kp: usize, out: []i64) void {
    for (0..s.n) |r| for (0..kp) |c| {
        out[r * kp + c] = if (c < kk) s.cidx[r * s.cur_kk + c] else -1;
    };
}

const RunOpts = struct {
    slice: usize = 0, // TF_DS_PREFILL_SLICE: 0 the whole chunk, else the layers one slice runs
    index_map: []const usize = &index_sources, // a synthetic map exercises the cut hook without an indexer at it
};

/// One prompt chunk's layers in TF_DS_PREFILL_SLICE slices (model.fillSlice; the decode rounds between slices
/// write round.zig's own rd.cidx, never the prompt's), with model.zig's replay cut at c.layers / 2.
fn runChunk(s: *Sim, opts: RunOpts, pre: usize) !void {
    s.n_events = 0;
    const step = if (opts.slice == 0) layers_n else opts.slice;
    var lo: usize = 0;
    while (lo < layers_n) : (lo += step) {
        for (lo..@min(layers_n, lo + step)) |li| {
            if (li == cut_layer and !s.replayCut(pre)) return;
            for (opts.index_map) |src| if (src == li) {
                const n_comp_end = (s.start + s.n) / ratio(li); // attention: n_comp_end = (start + n) / r
                s.indexer(@min(index_topk, n_comp_end));
            };
            if (ratio(li) != 0) _ = try s.attention();
        }
    }
}

/// The layers (0-based among the 38 compressed reads) whose read is a pad, for structural assertions.
fn padLayers(s: *const Sim, out: []usize) void {
    var m: usize = 0;
    for (s.events[0..s.n_events], 0..) |ev, j| {
        if (ev == .pad) {
            out[m] = j;
            m += 1;
        }
    }
}

/// The sweep's pick counts: 1..64 exhaustively, then >64's sampled widths (up to index_topk).
fn nextKk(kk: usize) usize {
    return switch (kk) {
        1...63 => kk + 1,
        64 => 65,
        65 => 80,
        80 => 100,
        100 => 127,
        127 => 200,
        200 => 511,
        511 => 512,
        else => 513,
    };
}

// -- the case matrix ------------------------------------------------------------------------------------------------

test "a cold 127-token prompt: 8 pads, 30 reuses, every read byte-equal (and slicing changes nothing)" {
    const alloc = std.testing.allocator;
    var s = try Sim.init(alloc, true);
    defer s.deinit();
    s.begin(0, 127, 0); // cold: no replay; the cut at layer 20 rewrites no rows
    try runChunk(&s, .{}, 0);
    // the served map's 38 padded reads (kk 63 at layers 2..19, kk 127 at 20..39; 63 and 127 are no whole tiles):
    // one pad per indexer selection, every shared-selection layer reusing it
    try std.testing.expectEqual(@as(usize, 8), s.fills);
    try std.testing.expectEqual(@as(usize, 8), s.copies);
    try std.testing.expectEqual(@as(usize, 30), s.reuses);
    try std.testing.expect(s.bannered and s.banner_rows == 127 and s.banner_kk == 63); // layers 3-7 share 63 picks
    const want_pads = [_]usize{ 0, 6, 12, 18, 22, 26, 30, 34 }; // attention reads of layers 2, 8, 14, 20, 24, ...
    var got_pads: [layers_n]usize = undefined;
    padLayers(&s, &got_pads);
    try std.testing.expectEqualSlices(usize, &want_pads, got_pads[0..want_pads.len]);
    // the same chunk under TF_DS_PREFILL_SLICE=5: identical events and counts (state crosses slice boundaries)
    var t = try Sim.init(alloc, true);
    defer t.deinit();
    t.begin(0, 127, 0);
    try runChunk(&t, .{ .slice = 5 }, 0);
    try std.testing.expectEqualSlices(Ev, s.events[0..s.n_events], t.events[0..t.n_events]);
    try std.testing.expectEqual(s.fills, t.fills);
    try std.testing.expectEqual(s.reuses, t.reuses);
    // the off path (the switch off): every padded read pads (38 fills, 38 copies, no reuse) - today's launches
    var u = try Sim.init(alloc, false);
    defer u.deinit();
    u.begin(0, 127, 0);
    try runChunk(&u, .{}, 0);
    try std.testing.expectEqual(@as(usize, 38), u.fills);
    try std.testing.expectEqual(@as(usize, 38), u.copies);
    try std.testing.expectEqual(@as(usize, 0), u.reuses);
    std.debug.print("h0270: cold 127-token prompt: on 8 fills + 8 copies, 30 reuses; off 38 + 38\n", .{});
}

test "multi-chunk prompts under TF_DS_PREFILL_SLICE=5: each chunk re-pads, none leaks across chunks" {
    const alloc = std.testing.allocator;
    var s = try Sim.init(alloc, true);
    defer s.deinit();
    const chunks = [_]struct { start: usize, n: usize }{ .{ .start = 0, .n = 127 }, .{ .start = 127, .n = 40 }, .{ .start = 167, .n = 17 } };
    for (chunks) |ch| {
        s.fills = 0;
        s.copies = 0;
        s.reuses = 0;
        s.bannered = false;
        s.begin(ch.start, ch.n, 0);
        try runChunk(&s, .{ .slice = 5 }, 0);
        try std.testing.expectEqual(@as(usize, 8), s.fills); // one pad per indexer selection in every chunk
        try std.testing.expectEqual(@as(usize, 30), s.reuses);
        try std.testing.expectEqual(Ev.pad, s.events[0]); // the new chunk's first padded read pads (no reuse)
    }
    std.debug.print("h0270: 3 chunks x 5-layer slices: 8 pads + 30 reuses each, first read of each pads\n", .{});
}

test "a replay cut mid-prompt: the rows' move re-pads; a cut that rewrites nothing keeps the pad" {
    const alloc = std.testing.allocator;
    // the served map, replay mid-prompt: pre-cut layers 2..19 share 63 picks, the cut at layer 20 rewrites the
    // rows and its layer's indexer writes fresh picks (kk 127) - 3 pads before, 5 after
    var s = try Sim.init(alloc, true);
    defer s.deinit();
    s.begin(0, 127, 64);
    try runChunk(&s, .{}, 0);
    try std.testing.expectEqual(@as(usize, 8), s.fills);
    try std.testing.expectEqual(@as(usize, 30), s.reuses);
    try std.testing.expectEqual(@as(usize, 63), s.n); // the cut's rows: 127 - 64
    // the cut hook alone (no indexer at or after the cut): a same-count shift (first 2 rows out, 2 tail rows in)
    // is invisible to the (rows, kk) key - only the hook forces the re-pad BRIEF-0270's contract requires
    var t = try Sim.init(alloc, true);
    defer t.deinit();
    const map = [_]usize{ 2, 8 };
    t.begin(0, 127, 2);
    try runChunk(&t, .{ .index_map = &map }, 2);
    try std.testing.expectEqual(@as(usize, 3), t.fills); // layers 2, 8, then 20 (the cut's layer, no indexer)
    try std.testing.expectEqual(Ev.pad, t.events[18]); // layer 20's read: the hook's re-pad
    try std.testing.expectEqual(@as(usize, 127), t.n); // same count, the rows' meaning moved
    // the same cut shape where the rows key alone would catch it (127 -> 135 rows) and byte-identity would
    // break without a re-pad (rows 127..134 are outside the old fill)
    var u = try Sim.init(alloc, true);
    defer u.deinit();
    u.begin(0, 127, 2);
    try runChunk(&u, .{ .index_map = &map }, 10);
    try std.testing.expectEqual(@as(usize, 3), u.fills);
    try std.testing.expectEqual(Ev.pad, u.events[18]);
    try std.testing.expectEqual(@as(usize, 135), u.n);
    // a cut that rewrites nothing (replay at the chunk's own start): the rule is "until a replay cut changes
    // the rows", so layer 20 reuses the padded list - and its bytes still match the reference
    var v = try Sim.init(alloc, true);
    defer v.deinit();
    v.begin(0, 127, 0);
    try runChunk(&v, .{ .index_map = &map }, 0);
    try std.testing.expectEqual(@as(usize, 2), v.fills);
    try std.testing.expectEqual(Ev.reuse, v.events[18]);
    std.debug.print("h0270: replay cut mid-prompt: served map 8 pads / 30 reuses; same-count cut re-pads at layer 20; no-op cut reuses\n", .{});
}

test "pick counts 1..64 and >64 over shared-selection sequences (both chunk classes)" {
    const alloc = std.testing.allocator;
    const smalls = [_]usize{ 1, 5, 7, 16, 17, 127 };
    var kk: usize = 1;
    while (kk <= 512) : (kk = nextKk(kk)) {
        for (smalls) |n| for ([_]bool{ false, true }) |after_cut| {
            var s = try Sim.init(alloc, true);
            defer s.deinit();
            if (after_cut) {
                s.begin(64, n, 64);
                _ = s.replayCut(0); // the no-op cut's floor: a small chunk whose window reaches below it
            } else s.begin(0, n, 0);
            // the expected rule written from the brief: padded unless the picks are whole 16-column tiles, or
            // the chunk is a decode-sized ring-mode one (which reads the compact list)
            const ring_mode = n <= ring_extra and !(replay_floor_on and s.floor > s.start -| (window - 1));
            const want_pad = kk % 16 != 0 and (n > decode_rows or (!ring_mode and n <= decode_rows));
            for (0..2) |_| { // two selections; three layers share each
                s.indexer(kk);
                for (0..3) |j| {
                    const ev = try s.attention();
                    if (!want_pad) {
                        try std.testing.expectEqual(Ev.read_compact, ev);
                    } else if (j == 0) {
                        try std.testing.expectEqual(Ev.pad, ev);
                    } else try std.testing.expectEqual(Ev.reuse, ev);
                }
            }
            try std.testing.expectEqual(if (want_pad) @as(usize, 2) else 0, s.fills);
            try std.testing.expectEqual(if (want_pad) @as(usize, 4) else 0, s.reuses);
        };
    }
    // the one-token prompt's arm: no compressed entry yet, the attention reads none
    var s = try Sim.init(alloc, true);
    defer s.deinit();
    s.begin(0, 1, 0);
    s.indexer(0);
    try std.testing.expectEqual(Ev.read_none, try s.attention());
    std.debug.print("h0270: pick sweep 1..64 and 65..512 x rows 1..127 x both chunk classes: all reads byte-equal\n", .{});
}

test "hand corner table: the pad predicate's edges as literals" {
    // the expectations are written from BRIEF-0270's rules by hand (whole 16-column tiles; a decode-sized
    // ring-mode chunk reads the compact list), not from the predicate transcription
    const alloc = std.testing.allocator;
    const Case = struct { kk: usize, n: usize, post_cut: bool, replay_floor: bool, want: Ev };
    const cases = [_]Case{
        .{ .kk = 16, .n = 17, .post_cut = false, .replay_floor = true, .want = .read_compact }, // whole tiles
        .{ .kk = 17, .n = 17, .post_cut = false, .replay_floor = true, .want = .pad },
        .{ .kk = 17, .n = 16, .post_cut = false, .replay_floor = true, .want = .read_compact }, // small ring-mode chunk
        .{ .kk = 17, .n = 16, .post_cut = true, .replay_floor = true, .want = .pad }, // below the floor: one_split
        .{ .kk = 17, .n = 16, .post_cut = true, .replay_floor = false, .want = .read_compact }, // the flag off: ring mode
        .{ .kk = 64, .n = 127, .post_cut = false, .replay_floor = true, .want = .read_compact },
        .{ .kk = 63, .n = 127, .post_cut = false, .replay_floor = true, .want = .pad }, // the served cold-prompt shape
        .{ .kk = 1, .n = 1, .post_cut = false, .replay_floor = true, .want = .read_compact },
        .{ .kk = 1, .n = 5, .post_cut = true, .replay_floor = true, .want = .pad },
        .{ .kk = 512, .n = 127, .post_cut = false, .replay_floor = true, .want = .read_compact },
    };
    for (cases) |c| {
        var s = try Sim.init(alloc, true);
        defer s.deinit();
        s.replay_floor = c.replay_floor;
        if (c.post_cut) {
            s.begin(64, c.n, 64);
            _ = s.replayCut(0); // the no-op cut's floor: the post-cut small chunk's
        } else s.begin(0, c.n, 0);
        for (0..2) |_| { // and again after a fresh selection: pad rows re-pad, compact rows stay compact
            s.indexer(c.kk);
            try std.testing.expectEqual(c.want, try s.attention());
        }
    }
    std.debug.print("h0270: corner table (10 literals x 2 selections): all events as hand-expected\n", .{});
}
