fn leasePoint(h: *H, key: Digest, at: u32) store.LeasePoint {
    return .{ .key = key, .at = at, .part = null, .compat = h.cfg.compat, .rank = h.cfg.rank };
}

fn leaseNow() i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

test "benchmark metadata lease protects against an already queued ordinary path trim at bounded cost" {
    const gpa = testing.allocator;
    const control = try H.init(gpa, .{ .tag = "lease_control", .turns = 2 });
    defer control.deinit();
    const peer = try H.init(gpa, .{ .tag = "lease_protected", .turns = 2 });
    defer peer.deinit();
    peer.cfg.bench_lease_bytes = 16384;
    try peer.reopen();
    const ids = try idsOf(gpa, 3 * block_tokens, 1120);
    defer gpa.free(ids);
    const opts: KO = .{ .lineage = lin(1), .source = 11 };
    for ([_]*H{ control, peer }) |h| {
        _ = try write(h, ids[0..block_tokens], opts);
        _ = try write(h, ids[0 .. 2 * block_tokens], opts);
        try testing.expect(keepIds(h, ids, opts) == .queued);
        try testing.expect(h.s.counts().pending_jobs > 0);
    }
    const key = format.pointKey(ids[0..block_tokens], block_tokens, &.{});
    const point = leasePoint(peer, key, block_tokens);
    const ct = peer.s.counts();
    const when = leaseNow();
    const grant = peer.s.benchGrantLease(point, 240, 41) orelse return error.ExpectedLease;
    const view = store.LeaseView.fromReply(grant, 41, when, leaseNow()) orelse return error.ExpectedView;
    try testing.expect(view.permits(peer.s.bench_incarnation, point, leaseNow()));
    try testing.expect(peer.s.benchGrantLease(point, 240, 41) == null);
    try testing.expectEqual(ct.staging_bytes, peer.s.counts().staging_bytes);
    try testing.expectEqual(ct.reads, peer.s.counts().reads);
    try testing.expectEqual(ct.loaded, peer.s.counts().loaded);
    try testing.expect(settle(control).done);
    try testing.expect(settle(peer).done);
    try testing.expect(!control.s.presence(key) and peer.s.presence(key));
    const more = peer.s.counts().stored_bytes - control.s.counts().stored_bytes;
    try testing.expect(more > 0 and more < grant.bytes and grant.bytes <= peer.cfg.bench_lease_bytes);
    try testing.expectEqual(control.s.counts().blocks, peer.s.counts().blocks);
    try testing.expectEqual(control.s.counts().written_bytes, peer.s.counts().written_bytes);
    try testing.expectEqual(@as(u64, 0), peer.s.counts().loaded);
    try testing.expectEqual(@as(u64, 0), peer.s.counts().reads);
    const elapsed_ms = @divTrunc(leaseNow() - when, std.time.ns_per_ms);
    try testing.expect(elapsed_ms < 120);
    try std.Io.sleep(io, .fromMilliseconds(250), .awake);
    try testing.expect(!view.permits(peer.s.bench_incarnation, point, leaseNow()));
    _ = peer.s.stage(&.{}); // the next ordinary between-round maintenance, even with no new write
    try testing.expect(!peer.s.presence(key));
    try testing.expectEqual(control.s.counts().stored_bytes, peer.s.counts().stored_bytes);
    try testing.expect(peer.s.benchGrantLease(point, 240, 41) == null); // grant after the racing trim must fail
    std.debug.print("LEASE_SLOT_B={d} VIEW_B={d} NOMINAL_CHARGE_B={d} ACTUAL_EXTRA_STORED_B={d} EXTRA_BLOCKS=0 EXTRA_WRITTEN_BLOCK_B=0 ACTIVE_READS=0 LOADED=0 TTL_MS=240 VIEW_MAX_MS=120 EXPIRY_EXTRA_STORED_B=0\n", .{ @sizeOf(store.LeaseSlot), @sizeOf(store.LeaseView), grant.bytes, more });
}

test "benchmark metadata lease rejects stale incarnation and bounds and invalidates damaged points" {
    const gpa = testing.allocator;
    const h = try H.init(gpa, .{ .tag = "lease_restart", .turns = 0 });
    defer h.deinit();
    const ids = try idsOf(gpa, 2 * block_tokens, 1131);
    defer gpa.free(ids);
    const key = try write(h, ids, .{ .lineage = lin(1), .source = 13 });
    const point = leasePoint(h, key, @intCast(ids.len));
    try testing.expect(h.s.benchGrantLease(point, 1000, 41) == null);
    h.cfg.bench_lease_bytes = 32;
    try h.reopen();
    try testing.expect(h.s.benchGrantLease(point, 1000, 41) == null);
    h.cfg.bench_lease_bytes = 1 << 20;
    try h.reopen();
    var bad = point;
    bad.at -= 1;
    try testing.expect(h.s.benchGrantLease(bad, 1000, 41) == null);
    bad = point;
    bad.part = 1;
    try testing.expect(h.s.benchGrantLease(bad, 1000, 41) == null);
    bad = point;
    bad.compat[0] ^= 1;
    try testing.expect(h.s.benchGrantLease(bad, 1000, 41) == null);
    bad = point;
    bad.rank += 1;
    try testing.expect(h.s.benchGrantLease(bad, 1000, 41) == null);
    try testing.expect(h.s.benchGrantLease(point, 0, 41) == null);
    try testing.expect(h.s.benchGrantLease(point, 1001, 41) == null);
    const sent = leaseNow();
    const first = h.s.benchGrantLease(point, 1000, 41) orelse return error.ExpectedLease;
    const view = store.LeaseView.fromReply(first, 41, sent, leaseNow()) orelse return error.ExpectedView;
    try testing.expect(store.LeaseView.fromReply(first, 41, sent, sent + 600 * std.time.ns_per_ms) == null);
    try testing.expect(!view.permits(null, point, leaseNow()));
    try testing.expect(store.LeaseView.fromReply(first, 42, sent, leaseNow()) == null); // stale response to a different outstanding request
    try h.reopen();
    try testing.expect(h.s.presence(key));
    try testing.expect(h.s.bench_lease == null);
    try testing.expect(!view.permits(h.s.bench_incarnation, point, leaseNow()));
    const fresh = h.s.benchGrantLease(point, 1000, 41) orelse return error.ExpectedLease;
    try testing.expect(!std.mem.eql(u8, &fresh.incarnation, &first.incarnation));
    try testing.expectEqual(first.serial, fresh.serial);
    h.s.forget(key, &.{}, &.{}); // actual authoritative invalidation wins over a live metadata promise
    try testing.expect(!h.s.presence(key) and h.s.bench_lease == null);
    std.debug.print("OLD_VIEW_AFTER_REOPEN=false SAME_POINT_ON_DISK=true FRESH_INCARNATION=true LATE_REPLY=false WRONG_REQUEST=false INVALID_EXTENT_PART_RANK_COMPAT=true EXPLICIT_FORGET_REMOVES_GRANT=true\n", .{});
}

test "benchmark metadata lease may displace another independent point or refuse an incoming write under a cap" {
    const gpa = testing.allocator;
    const a = try idsOf(gpa, block_tokens, 121);
    defer gpa.free(a);
    const b = try idsOf(gpa, block_tokens, 122);
    defer gpa.free(b);
    const c_ids = try idsOf(gpa, block_tokens, 123);
    defer gpa.free(c_ids);
    for ([_]bool{ false, true }) |protected| {
        const h = try H.init(gpa, .{ .tag = "lease_cap", .turns = 0 });
        defer h.deinit();
        const ka = try write(h, a, .{ .lineage = lin(1), .source = 121 });
        const kb = try write(h, b, .{ .lineage = lin(2), .source = 122 });
        h.cfg.bench_lease_bytes = 16384;
        h.cfg.cap_bytes = h.s.counts().stored_bytes + 512;
        try h.reopen();
        h.s.touch(ka);
        h.s.touch(kb);
        if (protected) _ = h.s.benchGrantLease(leasePoint(h, ka, block_tokens), 1000, 41) orelse return error.ExpectedLease;
        _ = try write(h, c_ids, .{ .lineage = lin(3), .source = 123 });
        try testing.expectEqual(protected, h.s.presence(ka));
        try testing.expectEqual(!protected, h.s.presence(kb));
        try testing.expectEqual(@as(u64, 2), h.s.counts().points);
        try testing.expect(h.s.counts().stored_bytes <= h.cfg.cap_bytes);
    }
    const h = try H.init(gpa, .{ .tag = "lease_nospace", .turns = 0 });
    defer h.deinit();
    const old = try write(h, a, .{ .lineage = lin(1), .source = 121 });
    h.cfg.bench_lease_bytes = 16384;
    h.cfg.cap_bytes = h.s.counts().stored_bytes + 512;
    try h.reopen();
    const claim = h.s.benchGrantLease(leasePoint(h, old, block_tokens), 80, 41) orelse return error.ExpectedLease;
    try testing.expect(keepIds(h, b, .{ .lineage = lin(2), .source = 122 }) == .no_space);
    try testing.expect(h.s.presence(old));
    try testing.expectEqual(@as(u64, 1), h.s.counts().no_space);
    try std.Io.sleep(io, .fromMilliseconds(90), .awake);
    _ = try write(h, b, .{ .lineage = lin(2), .source = 122 });
    try testing.expect(!h.s.presence(old));
    std.debug.print("TWO_POINT_CAP=PRESERVE_A_EVICT_B SINGLE_POINT_CAP=NEW_WRITE_REFUSED UNTIL_EXPIRY=true INCOMING_NO_SPACE=1 CAP_B={d} CHARGE_B={d}\n", .{ h.cfg.cap_bytes, claim.bytes });
}

test "benchmark metadata lease does not suppress a failed restore invalidating a shared inode" {
    const gpa = testing.allocator;
    const h = try H.init(gpa, .{ .tag = "lease_bad_inode" });
    defer h.deinit();
    h.cfg.bench_lease_bytes = 1 << 20;
    try h.reopen();
    const k = try linked(h);
    const a = h.s.points.get(k[0]).?;
    const b = h.s.points.get(k[1]).?;
    const point = leasePoint(h, k[1], b.at);
    try testing.expectEqual(h.s.have.get(a.blocks[0].name).?.ino, h.s.have.get(b.blocks[0].name).?.ino);
    _ = h.s.benchGrantLease(point, 1000, 41) orelse return error.ExpectedLease;
    var nb: [96]u8 = undefined;
    var path_buffer: [96]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buffer, "{s}", .{blockFile(&nb, a.blocks[0].name)}, 0);
    const fd = c.openat(h.s.blocks_fd, path, .{ .ACCMODE = .WRONLY, .CLOEXEC = true }, @as(c.mode_t, 0));
    try testing.expect(fd >= 0);
    const damaged: [64]u8 = @splat(0xff);
    const written = c.pwrite(fd, &damaged, damaged.len, 0);
    _ = c.close(fd);
    try testing.expectEqual(@as(isize, damaged.len), written);
    try testing.expect(try restoreFails(h, k[0]));
    for (k) |key| try testing.expect(!h.s.presence(key));
    try testing.expect(h.s.bench_lease == null);
    std.debug.print("SHARED_INODE_CORRUPTION=detected FAILED_RESTORE_OF_OTHER_KEY=true PROTECTED_POINT_INVALIDATED=true ALL_THREE_ALIASES_REMOVED=true\n", .{});
}
