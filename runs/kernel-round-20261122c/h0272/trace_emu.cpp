// h0272: CPU emulation of TF_DS_RING_TRACE's device-side record ring (BRIEF-0272's harness section: "sampling rule
// by sequence, wraparound, drop counting, multiple gathers per round") against an independent reference.
//
// The shipped record logic (trace_kernel_current.inc, spliced from zig/kernels/cuda/dsv41/ring_trace_rec.cuh) is
// driven exactly as the kernels call it: the stage kernel's block-0 thread-0 claim at entry and the doorbell
// publisher's publish, the collect kernel's block-0 thread-0 entry/flag/done. Only one thread writes records, so a
// thread-loop needs no barrier simulation. The reference below is written from BRIEF-0272's text (sampling by live
// sequence, a bounded preallocated ring that drops its oldest samples with a drop counter, the key fields and the
// same-clock GPU points including the previous gather's completion on the same stream), not from the kernel.
#define TF_DS_TRACE_EMULATE 1
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "trace_kernel_current.inc"

// -- the independent reference (from BRIEF-0272's spec) -------------------------------------------------------------

struct RRec {
    uint64_t seq = 0, round = 0, op = 0, layer = (uint64_t)-1, rows = 0, bytes = 0, sf = 0;
    uint64_t prev = 0, stage = 0, doorbell = 0, collect = 0, flag = 0, copy = 0;
};

// one gather's event values (a monotone clock stands in for globaltimer; the driver hands the same values to the
// emulator and the reference)
struct Ev {
    uint64_t stage, doorbell, collect, flag, copy;
};

struct Ref {
    uint64_t n, cap;
    std::map<uint64_t, RRec> log; // every sampled record, unbounded: the ground truth
    uint64_t prev[4] = {0, 0, 0, 0}; // the streams' previous gather's completion (every gather stamps it)

    Ref(uint64_t n_, uint64_t cap_) : n(n_), cap(cap_) {}
    bool sampled(uint64_t q) const { return n != 0 && q % n == 0; }

    // the brief's record: the key at claim, the bytes at publication, the collect's points, and the previous
    // gather's completion on the same stream as the producer interval's landmark
    void gather(uint64_t q, uint64_t round, int op, int layer, uint64_t rows, int stream, uint32_t flags,
                uint64_t bytes, uint64_t publish_rows, const Ev& ev) {
        (void)rows; // the claim's rows is overwritten at publication (the row kernels publish the listed rows)
        if (sampled(q)) {
            RRec r;
            r.seq = q;
            r.round = round;
            r.op = (uint64_t)(unsigned)op;
            r.layer = (uint64_t)(long long)layer;
            r.rows = publish_rows;
            r.bytes = bytes;
            r.sf = ((uint64_t)flags << 32) | (unsigned)stream;
            r.prev = prev[stream];
            r.stage = ev.stage;
            r.doorbell = ev.doorbell;
            r.collect = ev.collect;
            r.flag = ev.flag;
            r.copy = ev.copy;
            log[q] = r;
        }
        prev[stream] = ev.copy; // every gather, sampled or not
    }

    // A bounded preallocated ring of fixed records drops its oldest samples when full ("with a drop counter when
    // full"). The ring holds the newest min(cap, K) samples of the run; a sample of order o (= q / n) has been
    // dropped at a drain iff a newer sample at least cap orders later has arrived. `mark` is the last drained
    // sequence: the drain's window is (mark, seq_now].
    struct Window {
        std::vector<const RRec*> kept;
        uint64_t expected = 0, dropped = 0;
    };
    Window window(uint64_t mark, uint64_t seq_now) const {
        Window w;
        uint64_t o_max = seq_now / n;
        std::vector<const RRec*> in;
        for (const auto& [q, r] : log) {
            if (q <= mark || q > seq_now) continue;
            w.expected += 1;
            in.push_back(&r);
        }
        for (const RRec* r : in) {
            if (o_max - r->seq / n < cap) w.kept.push_back(r);
        }
        w.dropped = w.expected - w.kept.size();
        return w;
    }
};

// -- the emulator: the shipped functions driven as the kernels call them ---------------------------------------------

struct Emu {
    std::vector<unsigned long long> ring;
    unsigned long long round_cell = 0;
    unsigned long long prev[4] = {0, 0, 0, 0};
    TfDsTrace t;

    Emu(uint64_t n, uint64_t cap) : ring(cap * TF_TR_REC, 0) {
        t.recs = ring.data();
        t.round = &round_cell;
        t.prev = prev;
        t.n = n;
        t.cap = cap;
        t.rows = 0;
        t.op = 0;
        t.layer = -1;
        t.stream = 0;
        t.flags = 0;
    }

    // one gather on `stream`, keys as a launch carries them: claim at stage entry (block 0 thread 0), publish at the
    // doorbell (the publisher), then the collect's entry, flag observation and copy end (block 0 thread 0)
    void gather(uint64_t q, int op, int layer, uint64_t rows, int stream, uint32_t flags,
                uint64_t bytes, uint64_t publish_rows, const Ev& ev) {
        t.op = op;
        t.layer = layer;
        t.rows = (long long)rows;
        t.stream = stream;
        t.flags = flags;
        tf_tr_claim(t, q, ev.stage);
        tf_tr_publish(t, q, ev.doorbell, bytes, (long long)publish_rows);
        tf_tr_entry(t, q, ev.collect);
        tf_tr_flag(t, q, ev.flag);
        tf_tr_done(t, q, ev.copy);
    }

    struct Kept {
        std::vector<RRec> recs;
        uint64_t expected = 0, dropped = 0;
        bool dirty = false; // a reserved word that is not zero
    };

    // the drain's scan and drop formula (rdma.zig traceDrain / ring_trace.sampledCount)
    Kept drain(uint64_t mark, uint64_t seq_now) {
        Kept k;
        for (size_t s = 0; s * TF_TR_REC < ring.size(); ++s) {
            const unsigned long long* r = ring.data() + s * TF_TR_REC;
            uint64_t q = r[TF_TR_SEQ];
            if (q <= mark || q > seq_now || !sampled(q)) continue;
            RRec x;
            x.seq = r[TF_TR_SEQ]; x.round = r[TF_TR_ROUND]; x.op = r[TF_TR_OP]; x.layer = r[TF_TR_LAYER];
            x.rows = r[TF_TR_ROWS]; x.bytes = r[TF_TR_BYTES]; x.sf = r[TF_TR_SF]; x.prev = r[TF_TR_PREV];
            x.stage = r[TF_TR_STAGE]; x.doorbell = r[TF_TR_DOORBELL]; x.collect = r[TF_TR_COLLECT];
            x.flag = r[TF_TR_FLAG]; x.copy = r[TF_TR_COPY];
            if (r[13] != 0 || r[14] != 0 || r[15] != 0) k.dirty = true;
            k.recs.push_back(x);
        }
        k.expected = seq_now / t.n - mark / t.n;
        k.dropped = k.expected - k.recs.size();
        return k;
    }
    bool sampled(uint64_t q) const { return t.n != 0 && q % t.n == 0; }
};

// -- cases ----------------------------------------------------------------------------------------------------------

static int fails = 0;
static void check(bool ok, const char* what) {
    printf("  %s: %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) fails++;
}

static bool same(const RRec& a, const RRec& b) {
    return a.seq == b.seq && a.round == b.round && a.op == b.op && a.layer == b.layer && a.rows == b.rows &&
           a.bytes == b.bytes && a.sf == b.sf && a.prev == b.prev && a.stage == b.stage &&
           a.doorbell == b.doorbell && a.collect == b.collect && a.flag == b.flag && a.copy == b.copy;
}

static uint64_t g_clock = 1000000;
static Ev nextEv() {
    g_clock += 137;
    return Ev{ g_clock, g_clock += 91, g_clock += 53, g_clock += 211, g_clock += 7 };
}

// runs rounds x per_round gathers from q0 + 1 over two streams, `per_round` a round, draining at the named rounds;
// every drain compares the emulator's bounded ring against the reference's view: records field for field, kept
// counts and the drop counter
static void scenario(const char* name, uint64_t n, uint64_t cap, uint64_t q0, int rounds, int per_round,
                     const std::vector<int>& drain_after_rounds) {
    printf("case %s (n=%llu cap=%llu rounds=%d x %d gathers, drains after", name, (unsigned long long)n,
           (unsigned long long)cap, rounds, per_round);
    for (int d : drain_after_rounds) printf(" %d", d);
    printf(")\n");
    Emu emu(n, cap);
    Ref ref(n, cap);
    uint64_t q = q0;
    uint64_t mark = q0;
    for (int rd = 0; rd < rounds; ++rd) {
        emu.round_cell = 1000 + rd; // the host's tick upload before the round's launches
        for (int g = 0; g < per_round; ++g) {
            int stream = g % 2;
            int op = (rd + g) % 6;
            int layer = (op == 4) ? -1 : (rd * 7 + g) % 40; // the head has no layer
            uint64_t rows = 1 + ((rd + g) % 7);
            uint32_t flags = (uint32_t)((rd + g) % 2); // a captured graph's launch alternating
            uint64_t bytes = rows * 5120 * 4;
            uint64_t publish_rows = (op == 2) ? rows - (g % 2) : rows; // the row kernels publish the listed rows
            q += 1;
            Ev ev = nextEv();
            emu.gather(q, op, layer, rows, stream, flags, bytes, publish_rows, ev);
            ref.gather(q, 1000 + rd, op, layer, rows, stream, flags, bytes, publish_rows, ev);
        }
        for (int d : drain_after_rounds) {
            if (d != rd + 1) continue;
            auto got = emu.drain(mark, q);
            auto want = ref.window(mark, q);
            bool ok = !got.dirty && got.expected == want.expected && got.dropped == want.dropped &&
                      got.recs.size() == want.kept.size();
            std::map<uint64_t, RRec> byq;
            for (const auto& r : got.recs) byq[r.seq] = r;
            for (const RRec* w : want.kept) {
                auto it = byq.find(w->seq);
                if (it == byq.end() || !same(it->second, *w)) ok = false;
            }
            char buf[160];
            snprintf(buf, sizeof buf, "drain after round %d: %llu kept of %llu expected, %llu dropped, fields match",
                     d, (unsigned long long)got.recs.size(), (unsigned long long)got.expected,
                     (unsigned long long)got.dropped);
            check(ok, buf);
            mark = q;
        }
    }
}

int main() {
    // the case matrix BRIEF-0272 names: the sampling rule by sequence, wraparound, drop counting, multiple gathers
    // per round, across intervals 1, 2 and 7 and rings that wrap mid-round
    scenario("sampling-n1", 1, 8, 0, 3, 5, {3});
    scenario("sampling-n2", 2, 8, 0, 3, 5, {3});
    scenario("sampling-n7", 7, 8, 0, 4, 6, {4});
    scenario("wraparound-middrain", 2, 4, 0, 8, 5, {4, 8});
    scenario("wraparound-n1", 1, 4, 10, 4, 4, {2, 4});

    // unsampled gathers share slots with sampled ones (n = 2: q and q + 1 map to one slot): their events must leave
    // the sampled record's fields untouched (the harness caught exactly this before the helpers self-gated)
    {
        printf("case unsampled-same-slot (n=2 cap=4)\n");
        Emu emu(2, 4);
        Ev a = nextEv(), b = nextEv(), c = nextEv();
        emu.gather(8, 1, 7, 6, 0, 0, 6 * 5120 * 4, 6, a); // sampled: claims (8/2)%4 = 0
        emu.gather(9, 3, 99, 1, 0, 0, 1234, 1, b); // unsampled: must not touch slot 0
        emu.gather(10, 1, 7, 6, 0, 0, 6 * 5120 * 4, 6, c); // sampled: claims (10/2)%4 = 1
        auto got = emu.drain(6, 10);
        bool ok = !got.dirty && got.recs.size() == 2 && got.dropped == 0 && got.expected == 2;
        for (const auto& r : got.recs) {
            if (r.seq == 8 && (r.op != 1 || r.layer != 7 || r.bytes != 6 * 5120 * 4u || r.stage != a.stage ||
                               r.copy != a.copy))
                ok = false;
            if (r.seq == 10 && (r.op != 1 || r.layer != 7 || r.stage != c.stage)) ok = false;
        }
        check(ok, "unsampled q=9 left sampled records 8 and 10 intact");
    }

    // multiple gathers per round: every record carries the round number live at its claim (captured graphs read the
    // cell at replay, so it must not be baked in)
    {
        printf("case round-cell (n=1 cap=64, 3 rounds x 4 gathers)\n");
        Emu emu(1, 64);
        Ref ref(1, 64);
        bool ok = true;
        uint64_t q = 0;
        for (int rd = 0; rd < 3; ++rd) {
            emu.round_cell = 500 + rd;
            for (int g = 0; g < 4; ++g) {
                q += 1;
                Ev ev = nextEv();
                emu.gather(q, 0, 0, 1, g % 2, 0, 64, 1, ev);
                ref.gather(q, 500 + rd, 0, 0, 1, g % 2, 0, 64, 1, ev);
            }
        }
        auto got = emu.drain(0, q);
        for (const auto& r : got.recs) {
            uint64_t want = 500 + (r.seq - 1) / 4;
            if (r.round != want) ok = false;
        }
        check(ok && got.recs.size() == 12, "every record carries the round number live at its claim");
    }

    // the previous gather's completion on the same stream is the producer interval's landmark
    {
        printf("case prev-per-stream (n=2 cap=16, interleaved streams)\n");
        Emu emu(2, 16);
        Ev a = nextEv(), b = nextEv(), c = nextEv();
        emu.gather(2, 0, 0, 1, 0, 0, 64, 1, a); // stream 0
        emu.gather(4, 0, 0, 1, 1, 0, 64, 1, b); // stream 1
        emu.gather(6, 0, 0, 1, 0, 0, 64, 1, c); // stream 0: prev is a.copy
        auto got = emu.drain(0, 6);
        bool ok = got.recs.size() == 3;
        for (const auto& r : got.recs) {
            if (r.seq == 2 && r.prev != 0) ok = false; // nothing before it on stream 0
            if (r.seq == 4 && r.prev != 0) ok = false; // nothing before it on stream 1
            if (r.seq == 6 && r.prev != a.copy) ok = false;
        }
        check(ok, "records carry their own stream's previous gather's completion");
    }

    printf(fails == 0 ? "h0272 trace_emu: ok\n" : "h0272 trace_emu: FAIL (%d checks failed)\n", fails);
    return fails != 0;
}
