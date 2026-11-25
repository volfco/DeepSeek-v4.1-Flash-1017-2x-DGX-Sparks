/// What leaveCapture does to a stretch's capture.
const StretchCapture = struct {
    e: *const Engine,
    rd: *Round,

    fn joinSide(c: StretchCapture) !void {
        return c.rd.joinSide(c.e);
    }

    fn joinPrefetch(c: StretchCapture) !void {
        return c.rd.joinPrefetch(c.e);
    }

    /// TF_DS_COMPRESS_SIDE's chain forked at a_in (its join site is _kv_idx's read).
    fn joinCompress(c: StretchCapture) !void {
        if (c.rd.cs_out) try c.e.s.wait(c.rd.cs_join.?);
    }

    /// TF_DS_KV_SHARD's early query stage forked onto the overlap stream (sendQueries).
    fn joinShardQ(c: StretchCapture) !void {
        if (c.rd.ss_q_out) try c.e.s.wait(c.rd.ss_jq.?);
    }

    fn end(c: StretchCapture) !cuda.graph.Graph {
        return cuda.graph.endCapture(c.e.s);
    }

    fn forgetForked(c: StretchCapture) void {
        c.rd.side_out = false;
        c.rd.pf_out = false;
        c.rd.cs_out = false;
        c.rd.ss_q_out = false;
    }
};
/// A stretch failed inside its capture. What it forked is joined so the capture ends cleanly (ended unjoined it fails
/// with STREAM_CAPTURE_UNJOINED, though the streams leave capture all the same). Then the round forgets that work,
/// whatever those calls did: a stretch starts with nothing forked, so all of it belonged to this capture. A flag left
/// set would make the next capture's first mix wait on an event recorded only in this one, which CUDA refuses
/// (INVALID_VALUE, then STREAM_CAPTURE_INVALIDATED on all that follows). The refused wait leaves the flag set, so every
/// new capture would fail until a restart while graphs already captured kept replaying (the recipe's issue #65).
/// The overlap stream's own pass (ss_jo) has no flag and is not joined here: ownerRing records its join event and waits
/// it in the same call, and a stretch that fails inside a ring exchange leaves a sized send whose pair the next ring
/// transfer refuses (ShardRingOrder, pair-ending).
fn leaveCapture(cap: anytype) void {
    cap.joinSide() catch {};
    cap.joinPrefetch() catch {};
    cap.joinCompress() catch {};
    cap.joinShardQ() catch {};
    if (cap.end()) |got| {
        var graph = got;
        graph.deinit();
    } else |_| {}
    cap.forgetForked();
}
