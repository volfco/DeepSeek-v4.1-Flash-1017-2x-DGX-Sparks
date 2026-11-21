
test "prompt end copies use the model stream and restore a different slot's absorbed cursor" {
    const Call = struct { dst: u64, src: u64, len: usize, stream: cuda.abi.Stream };
    const Spy = struct {
        var calls: [6]Call = undefined;
        var count: usize = 0;
        fn record(dst: u64, src: u64, len: usize, stream: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            calls[count] = .{ .dst = dst, .src = src, .len = len, .stream = stream };
            count += 1;
            return cuda.abi.success;
        }
        fn default(dst: u64, src: u64, len: usize) callconv(.c) cuda.abi.Result {
            return record(dst, src, len, null);
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuMemcpyDtoD_v2 = Spy.default;
    driver.api.cuMemcpyDtoDAsync_v2 = Spy.record;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    const stream: cuda.abi.Stream = @ptrFromInt(64);
    var m: Model = undefined;
    m.ctx = &ctx;
    m.stream = .{ .d = &driver, .handle = stream };
    m.ch.head_g = 8192;
    m.vocab = 8;
    m.dpool = .{ .stages = 2, .slots = 4, .ring = 16, .hd = 8 };
    m.dpool.?.rings[0] = 1024;
    m.dpool.?.rings[1] = 4096;
    m.dpool.?.absorbed[1] = 8201;
    m.dpool.?.absorbed[2] = 31;
    const save: u64 = 16384;
    try std.testing.expectEqual(544, m.endBytes());
    try m.endCopy(1, save, false);
    try m.restoreEnd(2, save, 8201);
    try std.testing.expectEqual(8201, m.dpool.?.absorbed[2]);
    try std.testing.expectEqual(6, Spy.count);
    const expected: [6]Call = .{
        .{ .dst = save, .src = 1280, .len = 256, .stream = stream },
        .{ .dst = save + 256, .src = 4352, .len = 256, .stream = stream },
        .{ .dst = save + 512, .src = 8192, .len = 32, .stream = stream },
        .{ .dst = 1536, .src = save, .len = 256, .stream = stream },
        .{ .dst = 4608, .src = save + 256, .len = 256, .stream = stream },
        .{ .dst = 8192, .src = save + 512, .len = 32, .stream = stream },
    };
    for (Spy.calls, expected) |call, want| try std.testing.expectEqual(want, call);
}
