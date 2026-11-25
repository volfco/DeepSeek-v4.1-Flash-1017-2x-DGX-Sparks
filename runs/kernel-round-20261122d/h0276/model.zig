//! The intended policy of 0276 (TF_DS_CACHE_REPLACE_STALE), written from BRIEF-0276's spec alone - the independent
//! reference the harness checks the worktree's `keepWith` against. It is deliberately NOT derived from the kernel
//! source; it encodes the policy sentence: "keeping a state the cache already holds replaces a stale duplicate
//! instead of refreshing it".
//!
//! Policy (per keep of `prompt`'s state at `at`):
//!   1. A state the cache does not hold yet is inserted (a copy).
//!   2. A duplicate (same `at` and the same token prefix) whose new keep is a full end while the old entry is not is
//!      replaced (the cache's pre-existing rule, switch or not).
//!   3. A duplicate is replaced when the switch is on, no durable point stands behind the old entry, and the family
//!      let the old state go (below). Otherwise the duplicate is refreshed (no copy).
//!
//! "The family let it go" (stale) is the family's own rule MINUS its prompt-length gates: the state is no longer
//! resident, its rows were written over (stale), it is a restore's destination (loading), or its rows the KV-shard
//! planner holds. A valid state near the prompt end is never "let go": only staleness counts, never the keeping
//! prompt's length - so an end state that still restores for an identical resent prompt is refreshed, not copied.
//! A replaced entry always yields a fresh copy in the same keep (a failed save keeps nothing: the dead copy was
//! dropped anyway - the harness does not exercise save failures).

const std = @import("std");
const Allocator = std.mem.Allocator;

/// The family-side facts about one kept state, as the spec words them.
pub const Flags = struct {
    resident: bool = true,
    stale: bool = false,
    loading: bool = false,
    planner: bool = false,
};

/// Whether the family let `f`'s state go: it restores for no prompt at all (staleness only).
pub fn letGo(f: Flags) bool {
    return !f.resident or f.stale or f.loading or f.planner;
}

/// What a keep did.
pub const Outcome = enum {
    insert, // a new state was copied
    refresh, // the duplicate counted as kept again: no copy
    replace, // the duplicate was replaced by a fresh copy
    replace_end, // rule 2: a full end displaced a non-full-end duplicate
};

/// The state after a keep, for whole-state comparison with the real store.
pub const Snapshot = struct {
    entries: usize,
    kept: u64, // copies made (inserts + replacements)
    saves: u32, // family saves (== `kept` here; the harness compares it with its fake family's counter)
    find_at: ?u32, // the best usable state for the last prompt looked up
    full_end: bool, // the sole entry's flag (false when none)
    has_point: bool, // the sole entry carries a durable point
};

pub const Entry = struct {
    at: u32,
    tokens: []const u32,
    point: ?u64,
    full_end: bool,
    flags: Flags,
};

pub const Model = struct {
    gpa: Allocator,
    entries: std.ArrayList(Entry) = .empty,
    kept: u64 = 0,
    switch_on: bool,
    endstate: bool,
    tier_on: bool,
    next_point: u64 = 1,

    pub fn init(gpa: Allocator, switch_on: bool, endstate: bool, tier_on: bool) Model {
        return .{ .gpa = gpa, .switch_on = switch_on, .endstate = endstate, .tier_on = tier_on };
    }

    pub fn deinit(m: *Model) void {
        for (m.entries.items) |e| m.gpa.free(e.tokens);
        m.entries.deinit(m.gpa);
    }

    /// The family lets the state at `at` go (the harness marks the fake family's copy the same way).
    pub fn markLetGo(m: *Model, at: u32, f: Flags) void {
        for (m.entries.items) |*e| if (e.at == at) {
            e.flags = f;
            return;
        };
    }

    fn duplicate(m: *Model, prompt: []const u32, at: u32) ?usize {
        for (m.entries.items, 0..) |e, i| {
            if (e.at == at and e.tokens.len == at and std.mem.eql(u32, e.tokens, prompt[0..at])) return i;
        }
        return null;
    }

    /// The policy above. `full_end` is the cache's own rule (a pass's own end at `at == prompt.len`).
    pub fn keep(m: *Model, prompt: []const u32, at: u32) !Outcome {
        const full_end = m.endstate and at == prompt.len;
        const dup = m.duplicate(prompt, at);
        const di = dup orelse {
            try m.insert(prompt, at, full_end);
            return .insert;
        };
        const e = m.entries.items[di];
        if (full_end and !e.full_end) { // rule 2
            try m.replace(di, prompt, at, full_end);
            return .replace_end;
        }
        if (m.switch_on and e.point == null and letGo(e.flags)) { // rule 3 (the new rule)
            try m.replace(di, prompt, at, full_end);
            return .replace;
        }
        return .refresh;
    }

    fn insert(m: *Model, prompt: []const u32, at: u32, full_end: bool) !void {
        const tokens = try m.gpa.dupe(u32, prompt[0..at]);
        errdefer m.gpa.free(tokens);
        try m.entries.append(m.gpa, .{
            .at = at,
            .tokens = tokens,
            .point = if (m.tier_on and !full_end) m.nextPoint() else null,
            .full_end = full_end,
            .flags = .{},
        });
        m.kept += 1;
    }

    fn replace(m: *Model, di: usize, prompt: []const u32, at: u32, full_end: bool) !void {
        const old = m.entries.orderedRemove(di);
        m.gpa.free(old.tokens);
        try m.insert(prompt, at, full_end);
    }

    fn nextPoint(m: *Model) u64 {
        m.next_point += 1;
        return m.next_point;
    }

    /// The best usable state for `prompt`: the longest whose tokens prefix-match it, not let go, and a state at the
    /// prompt's end only as a full end (the cache's lookup contract).
    pub fn find(m: *Model, prompt: []const u32) ?u32 {
        var best: ?u32 = null;
        for (m.entries.items) |e| {
            if (e.tokens.len > prompt.len) continue;
            if (best) |b| if (e.at <= b) continue;
            if (!std.mem.eql(u32, e.tokens, prompt[0..e.tokens.len])) continue;
            if (e.at == prompt.len and !e.full_end) continue;
            if (letGo(e.flags)) continue;
            best = e.at;
        }
        return best;
    }

    /// The state after one keep, for comparison with the real store's.
    pub fn snapshot(m: *Model, find_prompt: []const u32) Snapshot {
        const sole = if (m.entries.items.len == 1) m.entries.items[0] else null;
        return .{
            .entries = m.entries.items.len,
            .kept = m.kept,
            .saves = @intCast(m.kept),
            .find_at = m.find(find_prompt),
            .full_end = if (sole) |e| e.full_end else false,
            .has_point = if (sole) |e| e.point != null else false,
        };
    }
};
