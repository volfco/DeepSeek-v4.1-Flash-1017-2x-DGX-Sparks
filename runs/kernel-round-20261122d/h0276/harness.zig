//! h0276: the CPU harness for 0276 (TF_DS_CACHE_REPLACE_STALE). It runs `keepWith` sequences on the REAL store - the
//! worktree's core/prompt_cache.zig, imported from the path build.sh passes as -Dpc - against model.zig, the intended
//! policy written from BRIEF-0276's spec alone. Every step compares the whole state (entry count, copies made,
//! family saves, what a lookup resumes, and the entry's flags) and the keep's outcome (insert / refresh / replace).
//!
//! The fake family's `usable` carries dsv41's `Lanes.usable` SHAPE (residency and staleness first, then the length
//! gates: an end state answers `is_end` at its own length and the tail-batch capability gate at any longer prompt), so
//! the probe-length rule (`cache_replace_stale.probeLen`) is exercised where the source patch's max-length probe
//! alone would replace a live end state. A fake tier stands in for the spill tier (point written at keep).
//!
//! Single-threaded; the cases are tiny. Exit 0: every step matched the model. Stderr carries the engagement banner
//! (once, the first replacement) - build.sh counts it in run.log.
const std = @import("std");
const pc = @import("prompt_cache");
const model = @import("model.zig");

const Allocator = std.mem.Allocator;

// -- the fake family (its rule shaped like Lanes.usable) -----------------------------------------------------------

const State = struct {
    at: u32,
    sum: u64,
    // the spec's staleness facts (model.Flags reads the same ones)
    resident: bool = true,
    stale: bool = false,
    loading: bool = false,
    planner: bool = false,
    // the family-rule shape (not staleness): an end keep, and how an end state continues
    is_end: bool = false,
};

const Fam = struct {
    gpa: Allocator,
    at: u32 = 0,
    sum: u64 = 0,
    saves: u32 = 0,
    live: u32 = 0,
    tail_batch: bool = true, // 0180 (TF_DS_TAIL_BATCH): an end state continues a longer prompt
    grid: u32 = 0,
    window: u32 = 0,
    end_next: bool = false, // the next save is a pass's own end (0178's end keep, as snapSaveFn decides it)
    last: ?*State = null, // the state the last save produced (the harness marks it stale/loading)

    fn snapshots(f: *Fam) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .usable = usableFn } };
    }

    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return 100 + at;
    }

    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *Fam = @ptrCast(@alignCast(ptr));
        const st = try f.gpa.create(State);
        st.* = .{ .at = at, .sum = f.sum, .is_end = f.end_next };
        f.last = st;
        f.saves += 1;
        f.live += 1;
        return st;
    }

    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const f: *Fam = @ptrCast(@alignCast(ptr));
        const st: *State = @ptrCast(@alignCast(saved));
        f.at = st.at;
        f.sum = st.sum;
    }

    fn dropFn(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *Fam = @ptrCast(@alignCast(ptr));
        const st: *State = @ptrCast(@alignCast(saved));
        f.gpa.destroy(st);
        f.live -= 1;
    }

    /// Lanes.usable's shape (lanes.zig:1863-1880): residency and staleness first, then the prompt-length gates.
    fn usableFn(ptr: *anyopaque, saved: pc.Saved, len: u32) bool {
        const f: *Fam = @ptrCast(@alignCast(ptr));
        const st: *State = @ptrCast(@alignCast(saved));
        if (!st.resident or st.planner or st.stale or st.loading or st.at == 0) return false;
        if (st.at == len) return st.is_end;
        if (st.at > len) return false;
        if (st.is_end) return f.tail_batch and st.at % 2 == 0;
        return (f.grid == 0 or st.at % f.grid == 0) and st.at <= len -| f.window;
    }

    fn usable(f: *Fam, saved: pc.Saved, len: u32) bool {
        return usableFn(f, saved, len);
    }
};

// -- the fake tier (a point named at keep; nothing else is exercised) ----------------------------------------------

const FakeTier = struct {
    next: u64 = 1,

    fn lookup(_: *anyopaque, _: *const pc.Want) ?pc.Point {
        return null;
    }
    fn begin(_: *anyopaque, _: *const pc.Point) anyerror!u64 {
        return error.Busy;
    }
    fn poll(_: *anyopaque, _: u64, _: u64) pc.Poll {
        return .{ .failed = error.Busy };
    }
    fn cancel(_: *anyopaque, _: u64) void {}
    fn keep(ptr: *anyopaque, _: *const pc.Kept) ?u64 {
        const t: *FakeTier = @ptrCast(@alignCast(ptr));
        t.next += 1;
        return t.next;
    }
    fn forget(_: *anyopaque, _: u64) void {}
    fn flush(_: *anyopaque, _: u64) u64 {
        return 0;
    }
    fn counts(_: *anyopaque, _: []pc.TierCounts) usize {
        return 0;
    }

    fn tier(t: *FakeTier) pc.Tier {
        return .{ .ptr = t, .vtable = &.{ .lookup = lookup, .begin = begin, .poll = poll, .cancel = cancel, .keep = keep, .forget = forget, .flush = flush, .counts = counts } };
    }
};

// -- the case driver ----------------------------------------------------------------------------------------------

const Cfg = struct {
    name: []const u8,
    switch_on: bool,
    endstate: bool,
    tier: bool = false,
    tail_batch: bool = true,
};

const Step = union(enum) {
    keep: struct { at: u32, prompt: []const u32, find_on: []const u32 },
    mark: struct { at: u32, f: model.Flags, find_on: []const u32 },
    probe: struct { at: u32 }, // print the family's verdict at both probe lengths (the source patch's and ours)
};

const A7 = [_]u32{ 1, 2, 3, 4, 5, 9, 9 };
const B5 = [_]u32{ 1, 2, 3, 4, 5 };
const C4 = [_]u32{ 1, 2, 3, 4 };

const cases = [_]struct { cfg: Cfg, steps: []const Step }{
    .{ .cfg = .{ .name = "baseline-live-refresh", .switch_on = true, .endstate = false }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } },
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // live again: refreshed, never copied
    } },
    .{ .cfg = .{ .name = "stale-replace", .switch_on = true, .endstate = false }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } },
        .{ .mark = .{ .at = 5, .f = .{ .stale = true }, .find_on = &A7 } }, // the family lets it go: nothing usable
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // the same state again: replaced
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // live again: refreshed
    } },
    .{ .cfg = .{ .name = "stale-refresh-off", .switch_on = false, .endstate = false }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } },
        .{ .mark = .{ .at = 5, .f = .{ .stale = true }, .find_on = &A7 } },
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // off: refreshed (the old behaviour)
    } },
    .{ .cfg = .{ .name = "point-guard", .switch_on = true, .endstate = false, .tier = true }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // kept with a durable point
        .{ .mark = .{ .at = 5, .f = .{ .stale = true }, .find_on = &A7 } },
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // the point stands: refreshed, not replaced
    } },
    .{ .cfg = .{ .name = "end-stale-replace", .switch_on = true, .endstate = true }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // the pass's own end (at == len)
        .{ .mark = .{ .at = 5, .f = .{ .stale = true }, .find_on = &B5 } },
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // its stale duplicate: replaced
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // live again: refreshed
    } },
    .{ .cfg = .{ .name = "end-live-refresh-odd-off", .switch_on = true, .endstate = true, .tail_batch = false }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // a live end state (odd at, tail batch off)
        .{ .probe = .{ .at = 5 } }, // the source patch's max-length probe calls this stale; ours must not
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // refreshed: a valid end is never copied again
    } },
    .{ .cfg = .{ .name = "end-live-refresh-even-on", .switch_on = true, .endstate = true }, .steps = &.{
        .{ .keep = .{ .at = 4, .prompt = &C4, .find_on = &C4 } }, // a live end state (even at, tail batch on)
        .{ .keep = .{ .at = 4, .prompt = &C4, .find_on = &C4 } }, // refreshed under both probes
    } },
    .{ .cfg = .{ .name = "end-over-notend-off", .switch_on = false, .endstate = true }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // a plain state at 5 (not this prompt's end)
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // a full end displaces it: the pre-existing rule
    } },
    .{ .cfg = .{ .name = "end-over-notend-on", .switch_on = true, .endstate = true }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } },
        .{ .keep = .{ .at = 5, .prompt = &B5, .find_on = &B5 } }, // unchanged by the switch
    } },
    .{ .cfg = .{ .name = "loading-replaced", .switch_on = true, .endstate = false }, .steps = &.{
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } },
        .{ .mark = .{ .at = 5, .f = .{ .loading = true }, .find_on = &A7 } }, // a restore's destination: let go too
        .{ .keep = .{ .at = 5, .prompt = &A7, .find_on = &A7 } }, // replaced like any stale copy
    } },
};

const Ctx = struct {
    gpa: Allocator,
    fam: Fam,
    tier: FakeTier = .{},
    store: pc.Store,
    model: model.Model,
    cfg: Cfg,

    fn init(c: *Ctx, gpa: Allocator, cfg: Cfg) void {
        c.* = .{ .gpa = gpa, .fam = .{ .gpa = gpa, .tail_batch = cfg.tail_batch }, .store = undefined, .model = .init(gpa, cfg.switch_on, cfg.endstate, cfg.tier), .cfg = cfg };
        c.store = pc.Store.init(gpa, c.fam.snapshots(), .{ .endstate = cfg.endstate }, 1 << 20);
        c.store.replace_stale = cfg.switch_on;
        if (cfg.tier) c.store.setTier(c.tier.tier(), 0);
    }

    fn deinit(c: *Ctx) void {
        c.store.deinit();
        c.model.deinit();
    }

    /// The real store's state, in the model's Snapshot shape.
    fn kernelSnapshot(c: *Ctx, find_on: []const u32) model.Snapshot {
        const sole = if (c.store.entries.items.len == 1) c.store.entries.items[0] else null;
        const hit = c.store.find(find_on, &.{});
        return .{
            .entries = c.store.entries.items.len,
            .kept = c.store.counts.kept,
            .saves = c.fam.saves,
            .find_at = if (hit) |e| e.at else null,
            .full_end = if (sole) |e| e.full_end else false,
            .has_point = if (sole) |e| e.point != null else false,
        };
    }
};

fn norm(o: model.Outcome) u8 {
    return switch (o) {
        .insert => 0,
        .refresh => 1,
        .replace, .replace_end => 2,
    };
}

const outcome_name = [_][]const u8{ "insert", "refresh", "replace" };

fn runCase(gpa: Allocator, cfg: Cfg, steps: []const Step, replaces: *u32) !void {
    var c: Ctx = undefined;
    c.init(gpa, cfg);
    defer c.deinit();
    std.debug.print("[{s}] switch={s} endstate={} tier={} tail_batch={}\n", .{ cfg.name, if (cfg.switch_on) "on" else "off", cfg.endstate, cfg.tier, cfg.tail_batch });
    for (steps, 0..) |step, i| {
        switch (step) {
            .keep => |k| {
                // the independent model decides first, from the spec's policy
                const want = try c.model.keep(k.prompt, k.at);
                // then the real store runs the same keep
                const before_len = c.store.entries.items.len;
                const before_kept = c.store.counts.kept;
                c.fam.at = k.at;
                c.fam.sum = 0;
                for (k.prompt[0..k.at]) |t| c.fam.sum = c.fam.sum *% 31 +% t;
                c.fam.end_next = c.cfg.endstate and k.at == k.prompt.len;
                const ret = c.store.keepWith(k.prompt, k.at, null, &.{}, .{});
                if (!ret) return error.KeepRefused;
                const after_len = c.store.entries.items.len;
                const kept_delta = c.store.counts.kept - before_kept;
                const observed: model.Outcome = if (kept_delta == 0) .refresh else if (after_len > before_len) .insert else .replace;
                try expectEqual(cfg.name, i, "outcome", @as(u8, norm(want)), @as(u8, norm(observed)));
                std.debug.print("[{s}] step {d}: keep at {d} -> {s}\n", .{ cfg.name, i + 1, k.at, outcome_name[norm(observed)] });
                if (want == .replace) replaces.* += 1; // only rule-3 replacements log the engagement banner
                try checkState(&c, i, k.find_on);
            },
            .mark => |m| {
                for (c.store.entries.items) |e| if (e.at == m.at) {
                    const st: *State = @ptrCast(@alignCast(e.saved));
                    st.resident = m.f.resident;
                    st.stale = m.f.stale;
                    st.loading = m.f.loading;
                    st.planner = m.f.planner;
                };
                c.model.markLetGo(m.at, m.f);
                std.debug.print("[{s}] step {d}: mark at {d} {any} (find must miss)\n", .{ cfg.name, i + 1, m.at, m.f });
                try checkState(&c, i, m.find_on);
            },
            .probe => |p| {
                const e = c.store.entries.items[0];
                std.debug.print("[{s}] step {d}: probe at {d}: usable(maxInt)={} usable(at={d})={} (the source patch's max-length probe would {s} this live state)\n", .{
                    cfg.name,
                    i + 1,
                    p.at,
                    c.fam.usable(e.saved, std.math.maxInt(u32)),
                    p.at,
                    c.fam.usable(e.saved, p.at),
                    if (c.fam.usable(e.saved, std.math.maxInt(u32))) "refresh" else "replace",
                });
            },
        }
    }
}

fn checkState(c: *Ctx, step: usize, find_on: []const u32) !void {
    const want = c.model.snapshot(find_on);
    const got = c.kernelSnapshot(find_on);
    try expectEqual(c.cfg.name, step, "entries", want.entries, got.entries);
    try expectEqual(c.cfg.name, step, "kept", want.kept, got.kept);
    try expectEqual(c.cfg.name, step, "saves", want.saves, got.saves);
    try expectEqual(c.cfg.name, step, "find_at", want.find_at, got.find_at);
    try expectEqual(c.cfg.name, step, "full_end", want.full_end, got.full_end);
    try expectEqual(c.cfg.name, step, "has_point", want.has_point, got.has_point);
}

fn expectEqual(name: []const u8, step: usize, what: []const u8, want: anytype, got: anytype) !void {
    if (!std.meta.eql(want, got)) {
        std.debug.print("[{s}] step {d}: MISMATCH {s}: model={any} store={any}\n", .{ name, step + 1, what, want, got });
        return error.Mismatch;
    }
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    var ok: u32 = 0;
    var failed: u32 = 0;
    var replaces: u32 = 0;
    std.debug.print("h0276: TF_DS_CACHE_REPLACE_STALE keep-side harness: the real prompt_cache.zig against the independent policy model\n", .{});
    for (cases) |case| {
        runCase(gpa, case.cfg, case.steps, &replaces) catch |err| {
            std.debug.print("[{s}] FAIL ({s})\n", .{ case.cfg.name, @errorName(err) });
            failed += 1;
            continue;
        };
        std.debug.print("[{s}] PASS\n", .{case.cfg.name});
        ok += 1;
    }
    std.debug.print("h0276: {d} cases passed, {d} failed; replacements observed {d} (the engagement banner must appear once)\n", .{ ok, failed, replaces });
    return if (failed == 0 and replaces > 0) 0 else 1;
}
