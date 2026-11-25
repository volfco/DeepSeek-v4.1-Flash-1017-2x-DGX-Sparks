//! h0273 CPU-emulation harness, patch 0274: "the tool-call gate commits the drawn token as the first
//! token" (the MiaAI recipe's issue #80). There is no kernel; the code under test is one statement in
//! zig/src/core/lanes/engine.zig (`started`). This file models that statement's exact shapes on the CPU:
//! the Feed union and readFeed are engine.zig's (backend.zig's `Feed`, engine.zig's `readFeed`), and the
//! two forms are the line before the fix (one assignment) and after it (read into a temporary, assign).
//!
//! The hazard: Zig builds `feed = .{ .value = ... }` at feed's address, so the tag can become .value
//! before the initializer's `readFeed(feed)` runs; the read then takes the union's payload (the draw
//! handle) as the value. Whether it does is the compiler's lowering; the harness measures it on the
//! toolchain at hand and prints the verdict. The independent reference is BRIEF-0274's contract: the
//! committed first token is the drawn token (a disarmed gate) or the forced opener (an armed one), never
//! the handle. The real code is covered by engine_test.zig's two tests (`zig build test`); this file's
//! fixed shape must match the reference in every case.

const std = @import("std");

const Feed = union(enum) { handle: u64, value: u32 };

const opener = 30;
const drawn = 68; // the backend's drawn first token for the modeled request
const handle: u64 = 0; // its draw handle (the fake backend's first handle; production saw 506, 607, ...)

fn backendRead(_: u64) u32 {
    return drawn; // engine.zig's readFeed(.handle): the backend's draw at that handle
}

fn readFeed(feed: Feed) u32 {
    return switch (feed) {
        .handle => |h| backendRead(h),
        .value => |v| v,
    };
}

/// engine.zig's correctCall for these two cases: a disarmed gate leaves the token alone (its cut finds
/// nothing to fix), an armed one fixes the first token to the call opener.
fn correctCall(armed: bool, token: u32) u32 {
    return if (armed) opener else token;
}

/// The line before the fix (engine.zig at 27baa93): one assignment, the read inside it.
fn beforeFix(armed: bool) u32 {
    var feed: Feed = .{ .handle = handle };
    const forced: ?u32 = if (armed) opener else null; // forcedNext: an armed gate's opener is forced
    if (forced) |t| feed = .{ .value = t };
    if (forced == null) feed = .{ .value = correctCall(armed, readFeed(feed)) };
    return readFeed(feed);
}

/// The fix: read the draw first, then assign.
fn afterFix(armed: bool) u32 {
    var feed: Feed = .{ .handle = handle };
    const forced: ?u32 = if (armed) opener else null;
    if (forced) |t| feed = .{ .value = t };
    if (forced == null) {
        const first = readFeed(feed);
        feed = .{ .value = correctCall(armed, first) };
    }
    return readFeed(feed);
}

/// BRIEF-0274's contract, written from its prose only.
fn reference(armed: bool) u32 {
    return if (armed) opener else drawn;
}

test "0274: the fixed shape commits the drawn token (disarmed) or the opener (armed)" {
    for ([_]bool{ false, true }) |armed| {
        try std.testing.expectEqual(reference(armed), afterFix(armed));
    }
    std.debug.print("PASS fixed shape matches the reference: disarmed -> drawn {d}, armed -> opener {d}\n", .{ drawn, opener });
}

test "0274: the armed case keeps its bits in both shapes (nothing else changes)" {
    try std.testing.expectEqual(reference(true), beforeFix(true));
    try std.testing.expectEqual(reference(true), afterFix(true));
    std.debug.print("PASS armed gate: both shapes commit the opener {d}\n", .{opener});
}

test "0274: verdict: what the old line commits on this toolchain" {
    const old = beforeFix(false);
    const ref = reference(false);
    if (old == ref) {
        std.debug.print("NOTE this toolchain evaluates the read first: the old line commits {d} and the premise does not reproduce here\n", .{old});
    } else {
        std.debug.print("PREMISE REPRODUCED on this toolchain: the old line commits {d} (the draw handle), the reference owes {d} (the drawn token)\n", .{ old, ref });
    }
    try std.testing.expectEqual(ref, afterFix(false));
}
