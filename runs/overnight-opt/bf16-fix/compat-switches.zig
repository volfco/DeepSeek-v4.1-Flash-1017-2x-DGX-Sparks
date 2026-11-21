
test "effective BF16 mode participates in the first follower frame" {
    const rank_id = @import("rank_identity.zig");
    const protected = @hasField(rank_id.Features, "bf16_partials");
    const m = try std.testing.allocator.create(Model);
    defer std.testing.allocator.destroy(m);
    for ([_]bool{ false, true }) |exact| for ([_]bool{ false, true }) |spec| {
        m.exact_resume = exact;
        m.ops.spec = spec;
        m.eng.bf16_partials = false;
        const base = @as(u8,@intFromBool(exact)) | (@as(u8,@intFromBool(spec)) << 1);
        try std.testing.expectEqual(base,pairSwitches(m));
        m.eng.bf16_partials = true;
        try std.testing.expectEqual(if (protected) base | 4 else base,pairSwitches(m));
    };
}
