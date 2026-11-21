
// Appended only to the benchmark copy of rank_wire.zig so this can reuse its loopback peer.
fn bf16FixtureExchange(rank: u8, bf16: bool) !fingerprint.Exchange {
    const identity = @import("rank_identity.zig");
    var f = identity.Features{ .spill = true, .kv_shard = true, .kv_carry = true, .twins = true, .spill_min_tokens = 2048,
        .streams = 8, .round_rows = 16, .pool = 2101248, .part_rows = 1050624, .drafts = true };
    if (comptime @hasField(identity.Features, "bf16_partials")) f.bf16_partials = bf16;
    var text: [320]u8 = undefined;
    const feature = try identity.featureString(&text, f);
    var b = fingerprint.Builder.init();
    for (fingerprint.all) |field| {
        if (field == .features) b.part(field, feature) else if (field == .rank_split) b.part(field, &.{rank}) else b.part(field, "fixed input");
    }
    return (try b.finish(null)).exchange();
}

test "BF16 pair compatibility and persistent store keys over loopback" {
    const identity = @import("rank_identity.zig");
    const tier = @import("spill_tier.zig");
    const state = @import("spill_state.zig");
    const protected = @hasField(identity.Features, "bf16_partials");
    const gpa = testing.allocator;
    const io = testing.io;
    const layout = try state.StateLayout.init(&.{.{ .ratio = 2, .kv = true, .idx = true, .raw = true }},
        .{ .block_tokens = 2048, .window = 128, .ring_extra = 16, .raw_rows = 8, .head_dim = 512, .index_head_dim = 128 }, .owner_sharded);
    var keys: [2][2]Digest = undefined;
    for ([_]bool{ false, true }) |head_on| {
        for ([_]bool{ false, true }) |peer_on| {
            const xf = try bf16FixtureExchange(0, head_on);
            const yf = try bf16FixtureExchange(1, peer_on);
            const port = testPort(io);
            const p = try gpa.create(Peer);
            defer gpa.destroy(p);
            p.* = .{ .port = port, .fx = yf };
            const th = try std.Thread.spawn(.{}, Peer.run, .{p});
            var l = try link.Link.listen(.{ 127, 0, 0, 1 }, port);
            defer l.close();
            var sink: TestSink = .{ .l = l };
            const leader = try gpa.create(Leader);
            defer gpa.destroy(leader);
            leader.* = Leader.init(2, 5, xf);
            const agree = leader.agree(sink.sink(), &[_]?link.Link{l});
            if (protected and head_on != peer_on) {
                try testing.expectError(error.FingerprintMismatch, agree);
                th.join();
                try testing.expectEqual(@as(?anyerror, error.FingerprintMismatch), p.err);
                try testing.expectEqual(@as(?u16, @intFromEnum(fingerprint.Field.features)), leader.broken.?.field);
                try testing.expectError(error.PairBroken, leader.post(sink.sink(), 0, .{ .write_staged = .{ .point = @splat(0), .kept = 1, .upto = 2048 } }));
                std.debug.print("head={} follower={} refused on both ranks, field=features, write blocked\n", .{head_on,peer_on});
            } else {
                const a = try agree;
                try l.send(&.{});
                th.join();
                try testing.expectEqual(@as(?anyerror, null), p.err);
                try testing.expect(p.side.agreed_seen);
                std.debug.print("head={} follower={} agreed\n", .{head_on,peer_on});
                if (head_on == peer_on) {
                    const mode = @intFromBool(head_on);
                    std.debug.print("mode={} common={x}\n", .{mode,&a.common});
                    for (0..2) |r| {
                        keys[mode][r] = tier.storeCompat(a, @intCast(r), &layout);
                        std.debug.print("mode={} rank={} store={x}\n", .{mode,r,&keys[mode][r]});
                    }
                }
            }
        }
    }
    for (0..2) |r| try testing.expectEqual(protected, !std.mem.eql(u8, &keys[0][r], &keys[1][r]));
}
