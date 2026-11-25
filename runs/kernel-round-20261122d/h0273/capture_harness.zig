//! h0273 CPU-emulation harness, patch 0273: "a failed stretch capture joins its forks and owes the next
//! capture no join". There is no CUDA kernel in this patch; the code under test is the capture failure
//! path in zig/src/families/dsv41/round.zig. build.sh appends the real `StretchCapture` + `leaveCapture`
//! source, extracted verbatim from round.zig (leave_capture.inc.zig), below this file; this half is an
//! independent CPU model of the CUDA stream-capture semantics the fix lives in, a stub Engine/Round that
//! the extracted code compiles against, and the case matrix from BRIEF-0273.
//!
//! Model semantics (the CUDA programming guide and BRIEF-0273's failure story, codes 901/904):
//! - a capture records event record/waits instead of executing them; a wait inside a capture is legal
//!   only on an event recorded in the same capture (a graph-local dependency);
//! - a wait on an event whose record never executed (captured in a dead capture, or never captured) is
//!   refused (INVALID_VALUE) and invalidates the capture (901 on everything after it);
//! - endCapture with a side stream's work forked and unjoined is refused (STREAM_CAPTURE_UNJOINED, 904),
//!   yet every stream leaves the capture all the same; a graph freed without instantiating never runs,
//!   so its captured records never execute.
//!
//! The independent reference (refTrace/refFlags below) is written from BRIEF-0273's contract prose only
//! ("the failure path joins what the stretch forked; it ends the capture; it clears both flags whatever
//! those calls returned ... the next capture owes no join"), not from the fix. Every case must match it.

const std = @import("std");
const gpa = std.testing.allocator;

// ---- the capture model ---------------------------------------------------------------------------

const Call = enum { side, prefetch, compress, shardq, end, free, forget };

const EvState = struct { in_cap: u32 = 0, open: bool = false, ran: bool = false };

/// A CUDA event handle; its state lives in the model (the handle copies by value, as cuda.Event does).
const Event = struct { st: *EvState, kind: Call, inject_fail: bool = false };

const Stream = struct {
    m: *Model,
    side: bool = false, // a fork target (the mixes' side, prefetch, compress and shard overlap streams)

    fn record(s: Stream, ev: Event) void {
        if (s.m.cap != 0) {
            ev.st.* = .{ .in_cap = s.m.cap, .open = s.side, .ran = false };
            if (s.side) s.m.open += 1;
        } else ev.st.* = .{ .in_cap = 0, .open = false, .ran = true };
    }

    fn wait(s: Stream, ev: Event) !void {
        try s.m.trace.append(gpa, ev.kind); // the attempt, as the ported round.zig Fake logs method entry
        if (ev.inject_fail) return error.CudaFailed; // a case knob: whatever that call returned
        if (s.m.cap != 0) {
            if (ev.st.in_cap != s.m.cap) {
                s.m.refusals += 1;
                s.m.invalidated = true;
                return error.CaptureRefused; // INVALID_VALUE, then 901 on all that follows
            }
        } else if (!ev.st.ran) {
            s.m.refusals += 1;
            return error.CaptureRefused;
        }
        if (ev.st.open) {
            ev.st.open = false;
            s.m.open -= 1;
        }
    }
};

const Model = struct {
    cap: u32 = 0, // the active capture's id (0: none)
    invalidated: bool = false,
    next_id: u32 = 1,
    open: u32 = 0, // side streams' work forked into the active capture and not yet joined
    refusals: u32 = 0,
    unjoined_ends: u32 = 0, // endCapture refusals (904)
    inject_end_fail: bool = false,
    trace: std.ArrayList(Call) = .empty,
    main: Stream = undefined,
    side: Stream = undefined,

    fn init(m: *Model) void {
        m.* = .{ .trace = .empty };
        m.main = .{ .m = m, .side = false };
        m.side = .{ .m = m, .side = true };
    }

    fn begin(m: *Model) void {
        m.cap = m.next_id;
        m.next_id += 1;
        m.invalidated = false;
    }

    fn endCapture(m: *Model) !cuda.graph.Graph {
        try m.trace.append(gpa, .end);
        defer m.open = 0; // whatever the verdict, the streams leave the capture and its work never runs
        if (m.inject_end_fail or m.invalidated) {
            m.cap = 0;
            return error.CaptureInvalid;
        }
        if (m.open > 0) {
            m.unjoined_ends += 1;
            m.cap = 0;
            return error.CaptureUnjoined; // 904: refused, yet the streams leave the capture
        }
        m.cap = 0;
        return .{ .m = m };
    }
};

/// The cuda namespace the extracted code names (round.zig's `cuda.graph.endCapture` / `Graph.deinit`).
const cuda = struct {
    const graph = struct {
        pub const Graph = struct {
            m: *Model,
            pub fn deinit(g: *Graph) void {
                g.m.trace.append(gpa, .free) catch {};
            }
        };
        pub fn endCapture(s: Stream) !Graph {
            return s.m.endCapture();
        }
    };
};

// ---- the stubs the extracted code compiles against ------------------------------------------------

const Engine = struct { s: Stream };

/// The fields and the two pre-existing helpers the extracted code touches. joinSide/joinPrefetch carry
/// round.zig's bodies (pre-existing code, also exercised by the ported round.zig unit test through its
/// Fake); the flag fields are the round's.
const Round = struct {
    side_out: bool = false,
    pf_out: bool = false,
    cs_out: bool = false,
    ss_q_out: bool = false,
    join: ?Event = null, // the mixes' side stream's join event
    pf_join: ?Event = null,
    cs_join: ?Event = null,
    ss_jq: ?Event = null,

    fn joinSide(rd: *Round, e: *const Engine) !void {
        if (!rd.side_out) return;
        try e.s.wait(rd.join.?);
        rd.side_out = false;
    }

    fn joinPrefetch(rd: *Round, e: *const Engine) !void {
        if (!rd.pf_out) return;
        try e.s.wait(rd.pf_join.?);
        rd.pf_out = false;
    }
};

// ---- the scenario driver --------------------------------------------------------------------------

const Forks = struct { side: bool = false, pf: bool = false, cs: bool = false, ssq: bool = false };

/// A stretch's forks at the point it fails, each exactly as round.zig's stretch sets it: the mixes' side
/// work's join event (mix's defer), the paced prefetch's (prefetch()), TF_DS_COMPRESS_SIDE's chain
/// (a_in), and the KV-shard overlap stream's early query stage (sendQueries). The model collapses each
/// fork to its join event and an open-work count.
fn forkAll(m: *Model, rd: *Round, forks: Forks) void {
    if (forks.side) {
        m.side.record(rd.join.?);
        rd.side_out = true;
    }
    if (forks.pf) {
        m.side.record(rd.pf_join.?);
        rd.pf_out = true;
    }
    if (forks.cs) {
        m.side.record(rd.cs_join.?);
        rd.cs_out = true;
    }
    if (forks.ssq) {
        m.side.record(rd.ss_jq.?);
        rd.ss_q_out = true;
    }
}

/// Today's failure path (the code 0273 replaces): end the capture and free whatever it returns. A
/// transcription of round.zig's `Graphs.run` catch before the fix (git show 27baa93:zig/src/families).
fn buggyFailurePath(m: *Model) void {
    if (m.endCapture()) |got| {
        var g = got;
        g.deinit();
    } else |_| {}
}

/// The next capture's stretch: its first mix joins whatever the last capture left (round.zig's mix calls
/// `rd.joinSide(e)` first thing), the stretch ends joined (the stretch tail and the layers' cs/ssq
/// joins), and the capture ends and is freed. This is the step that poisons every later capture when a
/// flag outlives a dead one.
fn nextCapture(m: *Model, rd: *Round, e: *const Engine, forks: Forks) !void {
    m.begin();
    try rd.joinSide(e); // the first mix: a stale flag's dead-capture event here is refused (the bug)
    try rd.joinPrefetch(e);
    forkAll(m, rd, forks); // its own forks: fresh events recorded in this capture
    try rd.joinSide(e);
    try rd.joinPrefetch(e);
    if (rd.cs_out) try e.s.wait(rd.cs_join.?);
    if (rd.ss_q_out) try e.s.wait(rd.ss_jq.?);
    var g = try m.endCapture();
    g.deinit();
}

// ---- the independent reference (from BRIEF-0273's prose, not from the fix) --------------------------

/// "it joins what the stretch forked, ... it ends the capture, it clears both flags whatever those calls
/// returned": every forked stream's join in fork order, then the capture's end, then the graph it
/// returned freed (nothing is returned when the end fails).
fn refTrace(forks: Forks, frees: bool, out: *std.ArrayList(Call)) !void {
    if (forks.side) try out.append(gpa, .side);
    if (forks.pf) try out.append(gpa, .prefetch);
    if (forks.cs) try out.append(gpa, .compress);
    if (forks.ssq) try out.append(gpa, .shardq);
    try out.append(gpa, .end);
    if (frees) try out.append(gpa, .free);
}

fn refFlags(rd: *const Round) bool {
    return !rd.side_out and !rd.pf_out and !rd.cs_out and !rd.ss_q_out;
}

// ---- cases -----------------------------------------------------------------------------------------

const Knobs = struct { name: []const u8, side_fails: bool, pf_fails: bool, cs_fails: bool, ssq_fails: bool, end_fails: bool };

const knobs = [_]Knobs{
    .{ .name = "all calls ok", .side_fails = false, .pf_fails = false, .cs_fails = false, .ssq_fails = false, .end_fails = false },
    .{ .name = "joinSide fails", .side_fails = true, .pf_fails = false, .cs_fails = false, .ssq_fails = false, .end_fails = false },
    .{ .name = "joinPrefetch fails", .side_fails = false, .pf_fails = true, .cs_fails = false, .ssq_fails = false, .end_fails = false },
    .{ .name = "joinCompress fails", .side_fails = false, .pf_fails = false, .cs_fails = true, .ssq_fails = false, .end_fails = false },
    .{ .name = "joinShardQ fails", .side_fails = false, .pf_fails = false, .cs_fails = false, .ssq_fails = true, .end_fails = false },
    .{ .name = "endCapture fails", .side_fails = false, .pf_fails = false, .cs_fails = false, .ssq_fails = false, .end_fails = true },
    .{ .name = "everything fails", .side_fails = true, .pf_fails = true, .cs_fails = true, .ssq_fails = true, .end_fails = true },
};

const fork_cases = [_]struct { name: []const u8, forks: Forks }{
    .{ .name = "nothing forked", .forks = .{} },
    .{ .name = "side only", .forks = .{ .side = true } },
    .{ .name = "prefetch only", .forks = .{ .pf = true } },
    .{ .name = "compress only", .forks = .{ .cs = true } },
    .{ .name = "shard q only", .forks = .{ .ssq = true } },
    .{ .name = "side + prefetch", .forks = .{ .side = true, .pf = true } },
    .{ .name = "all four forked", .forks = .{ .side = true, .pf = true, .cs = true, .ssq = true } },
};

const Fixture = struct {
    m: Model = undefined,
    rd: Round = .{},
    e: Engine = undefined,
    ests: [4]EvState = @splat(.{}),

    fn init(f: *Fixture) void {
        f.m.init();
        f.e = .{ .s = f.m.main };
        f.rd = .{
            .join = .{ .st = &f.ests[0], .kind = .side },
            .pf_join = .{ .st = &f.ests[1], .kind = .prefetch },
            .cs_join = .{ .st = &f.ests[2], .kind = .compress },
            .ss_jq = .{ .st = &f.ests[3], .kind = .shardq },
        };
    }

    fn deinit(f: *Fixture) void {
        f.m.trace.deinit(gpa);
    }
};

test "0273: the failure path matches the reference: joins, end, clear, and a clean next capture" {
    for (fork_cases) |fc| for (knobs) |k| {
        var f: Fixture = .{};
        f.init();
        defer f.deinit();
        f.m.begin();
        forkAll(&f.m, &f.rd, fc.forks);
        f.rd.join.?.inject_fail = k.side_fails;
        f.rd.pf_join.?.inject_fail = k.pf_fails;
        f.rd.cs_join.?.inject_fail = k.cs_fails;
        f.rd.ss_jq.?.inject_fail = k.ssq_fails;
        f.m.inject_end_fail = k.end_fails;
        // the failure path under test: the extracted leaveCapture over the extracted StretchCapture
        leaveCapture(StretchCapture{ .e = &f.e, .rd = &f.rd });
        // the reference: joins in fork order, the end, the graph it returned freed (there is none when
        // a failed join leaves a fork open: the end is then refused, 904); every flag clear
        var want: std.ArrayList(Call) = .empty;
        defer want.deinit(gpa);
        const frees = !k.end_fails and !(fc.forks.side and k.side_fails) and !(fc.forks.pf and k.pf_fails) and
            !(fc.forks.cs and k.cs_fails) and !(fc.forks.ssq and k.ssq_fails);
        try refTrace(fc.forks, frees, &want);
        try std.testing.expectEqualSlices(Call, want.items, f.m.trace.items);
        try std.testing.expect(refFlags(&f.rd));
        // the payoff: the next capture owes no join and completes clean, whatever the calls returned
        f.m.trace.clearRetainingCapacity();
        f.m.inject_end_fail = false;
        f.rd.join.?.inject_fail = false;
        f.rd.pf_join.?.inject_fail = false;
        f.rd.cs_join.?.inject_fail = false;
        f.rd.ss_jq.?.inject_fail = false;
        const refusals = f.m.refusals;
        const ends = f.m.unjoined_ends;
        try nextCapture(&f.m, &f.rd, &f.e, fc.forks);
        try std.testing.expectEqual(refusals, f.m.refusals);
        try std.testing.expectEqual(ends, f.m.unjoined_ends);
        std.debug.print("PASS forks[{s}] x {s}: reference trace matched, flags clear, next capture clean\n", .{ fc.name, k.name });
    };
}

test "0273: negative control: today's failure path poisons every later capture" {
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    f.m.begin();
    forkAll(&f.m, &f.rd, .{ .side = true });
    buggyFailurePath(&f.m);
    try std.testing.expectEqual(@as(u32, 1), f.m.unjoined_ends); // 904: ended unjoined
    try std.testing.expect(f.rd.side_out); // the flag outlives the dead capture
    // the next capture's first mix waits the dead capture's join: refused, the capture invalidated
    f.m.begin();
    try std.testing.expectError(error.CaptureRefused, f.rd.joinSide(&f.e));
    try std.testing.expect(f.m.invalidated);
    try std.testing.expect(f.rd.side_out); // and the refused wait leaves the flag set: it fails forever
    std.debug.print("PASS negative control: the old failure path leaves side_out set and every later capture refused (901 cascade)\n", .{});
}

test "0273: the overlap stream's own pass is unflagged yet its next-capture wait is safe" {
    // ownerRing forks the own pass onto the overlap stream and waits its join event in the same call
    // (round.zig's ownerRing: the sparseAttn fork, then the unconditional wait further down), so no
    // stale flag can carry it across captures. A failed capture leaves that one fork unjoined (accepted:
    // its 904 is the STREAM_CAPTURE_UNJOINED case the fix's flagged joins remove; and a ring send left
    // open ends the pair, ShardRingOrder). forward's errdefer clears ss_q_out and the layer cursor.
    var f: Fixture = .{};
    f.init();
    defer f.deinit();
    var jo = EvState{};
    const join = Event{ .st = &jo, .kind = .shardq };
    f.m.begin();
    f.m.side.record(join); // ownerRing's fork, before its wait
    leaveCapture(StretchCapture{ .e = &f.e, .rd = &f.rd });
    try std.testing.expectEqual(@as(u32, 1), f.m.unjoined_ends); // the unflagged fork is not joined
    // the next capture's ownerRing records its join fresh in its own capture, then waits: legal
    f.m.begin();
    f.m.side.record(join);
    try f.m.main.wait(join);
    var g = try f.m.endCapture();
    g.deinit();
    try std.testing.expectEqual(@as(u32, 0), f.m.refusals);
    std.debug.print("PASS ss own pass: unjoined in the dead capture (one 904), yet the next capture's record-then-wait is clean\n", .{});
}

test "0273: call order robustness: a failed call stops nothing after it" {
    // the ported round.zig Fake matrix, run here on the extracted leaveCapture as well
    const Log = struct {
        const Self = @This();
        calls: std.ArrayList(Call) = .empty,
        fails: [4]bool = @splat(false),
        end_fails: bool = false,
        const G = struct {
            l: *Self,
            fn deinit(g: *G) void {
                g.l.calls.append(gpa, .free) catch {};
            }
        };
        fn joinSide(l: *Self) !void {
            try l.calls.append(gpa, .side);
            if (l.fails[0]) return error.CudaFailed;
        }
        fn joinPrefetch(l: *Self) !void {
            try l.calls.append(gpa, .prefetch);
            if (l.fails[1]) return error.CudaFailed;
        }
        fn joinCompress(l: *Self) !void {
            try l.calls.append(gpa, .compress);
            if (l.fails[2]) return error.CudaFailed;
        }
        fn joinShardQ(l: *Self) !void {
            try l.calls.append(gpa, .shardq);
            if (l.fails[3]) return error.CudaFailed;
        }
        fn end(l: *Self) !G {
            try l.calls.append(gpa, .end);
            if (l.end_fails) return error.CudaFailed;
            return .{ .l = l };
        }
        fn forgetForked(l: *Self) void {
            l.calls.append(gpa, .forget) catch {};
        }
    };
    const whole = [_]Call{ .side, .prefetch, .compress, .shardq, .end, .free, .forget };
    const short = [_]Call{ .side, .prefetch, .compress, .shardq, .end, .forget };
    for (0..5) |fail_at| for ([_]bool{ false, true }) |end_fails| {
        var l: Log = .{};
        defer l.calls.deinit(gpa);
        if (fail_at < 4) l.fails[fail_at] = true;
        l.end_fails = end_fails;
        leaveCapture(&l);
        try std.testing.expectEqualSlices(Call, if (end_fails) &short else &whole, l.calls.items);
    };
    std.debug.print("PASS call order: every join before the end, the free after it, the clear last; a failed call stops nothing\n", .{});
}
