// TF_DS_RING_TRACE's per-sequence record writes (rdma_zig's drain reads them; ring_trace.zig mirrors these layouts

// and the sampling rule). A sampled sequence q (q % n == 0) takes record (q / n) % cap of the ring; the stage kernel

// claims it with the key and this gather's entry, the doorbell's publisher adds the bytes and its time, the collect

// kernel adds its entry, the flag observation and the copy's end. Every collect (sampled or not) also stamps this

// stream's last gather's completion, which the next gather on this stream reads as its producer-interval landmark.

// Included by dsv41/rdma_gather.cu and dsv41_shard.cu; compiled for the CPU by the h0272 harness with

// TF_DS_TRACE_EMULATE and by tools/dsv41_shard_emulate with TF_DS_EMULATE (the record rules are plain C++ on

// purpose, so the harness drives the real code).

#include <cstddef>

#include <cstdint>



// rdma.zig's ring_trace.Trace (extern struct, the same layout): by value, all zeros when the trace is off.

struct TfDsTrace {

    unsigned long long* recs;  // the record ring (TF_TR_REC words a record); null when off

    unsigned long long* round; // u64: the round number (host-updated at every round boundary)

    unsigned long long* prev;  // [4] u64: each stream's last gather's copy-done

    unsigned long long n;      // sample every nth ring sequence

    unsigned long long cap;    // records in recs

    long long rows;            // this transfer's rows (the row kernels write the listed count instead)

    int op;                    // ring_trace.Op

    int layer;                 // the layer, -1 when none

    int stream;                // this stream's ordinal, 0..3

    unsigned int flags;        // bit 0: a captured graph's launch

};



#if defined(TF_DS_TRACE_EMULATE) || defined(TF_DS_EMULATE)

#define TF_TR_HD static inline

#else

#define TF_TR_HD __device__ __forceinline__

#endif



// ring_trace.GpuRec's field indices (16 u64 a record; 13..15 reserved).

enum {

    TF_TR_REC = 16,

    TF_TR_SEQ = 0,

    TF_TR_ROUND = 1,

    TF_TR_OP = 2,

    TF_TR_LAYER = 3,

    TF_TR_ROWS = 4,

    TF_TR_BYTES = 5,

    TF_TR_SF = 6,

    TF_TR_PREV = 7,

    TF_TR_STAGE = 8,

    TF_TR_DOORBELL = 9,

    TF_TR_COLLECT = 10,

    TF_TR_FLAG = 11,

    TF_TR_COPY = 12

};



TF_TR_HD bool tf_tr_on(const TfDsTrace& t) { return t.recs != 0; }



TF_TR_HD bool tf_tr_sampled(const TfDsTrace& t, unsigned long long q) { return t.n != 0 && (q % t.n) == 0; }



TF_TR_HD int tf_tr_slot(const TfDsTrace& t, unsigned long long q) { return (int)((q / t.n) % t.cap); }



// The stage kernel's thread 0 of block 0 at entry: the key, the previous gather's completion on this stream and

// this entry. The sequence (the record's identity) goes in last, so a half-written slot is never mistaken for a

// record. Every helper writes a sampled sequence's record only (an unsampled gather shares slots with sampled ones).

TF_TR_HD void tf_tr_claim(const TfDsTrace& t, unsigned long long q, unsigned long long entry) {

    if (!tf_tr_sampled(t, q)) return;

    unsigned long long* r = t.recs + (size_t)tf_tr_slot(t, q) * TF_TR_REC;

    r[TF_TR_ROUND] = *t.round;

    r[TF_TR_OP] = (unsigned long long)(unsigned int)t.op;

    r[TF_TR_LAYER] = (unsigned long long)(long long)t.layer;

    r[TF_TR_ROWS] = (unsigned long long)t.rows;

    r[TF_TR_BYTES] = 0;

    r[TF_TR_SF] = ((unsigned long long)t.flags << 32) | (unsigned int)t.stream;

    r[TF_TR_PREV] = t.prev[t.stream];

    r[TF_TR_STAGE] = entry;

    r[TF_TR_DOORBELL] = 0;

    r[TF_TR_COLLECT] = 0;

    r[TF_TR_FLAG] = 0;

    r[TF_TR_COPY] = 0;

    r[TF_TR_SEQ] = q;

}



// The doorbell's publisher: the bytes actually sent (the row kernels pass their listed rows too) and its time.

TF_TR_HD void tf_tr_publish(const TfDsTrace& t, unsigned long long q, unsigned long long doorbell,

                            unsigned long long bytes, long long rows) {

    if (!tf_tr_sampled(t, q)) return;

    unsigned long long* r = t.recs + (size_t)tf_tr_slot(t, q) * TF_TR_REC;

    r[TF_TR_BYTES] = bytes;

    r[TF_TR_ROWS] = (unsigned long long)rows;

    r[TF_TR_DOORBELL] = doorbell;

}



// The collect kernel's block 0 thread 0: its entry after PDL, and the flag wait's end.

TF_TR_HD void tf_tr_entry(const TfDsTrace& t, unsigned long long q, unsigned long long at) {

    if (!tf_tr_sampled(t, q)) return;

    t.recs[(size_t)tf_tr_slot(t, q) * TF_TR_REC + TF_TR_COLLECT] = at;

}



TF_TR_HD void tf_tr_flag(const TfDsTrace& t, unsigned long long q, unsigned long long at) {

    if (!tf_tr_sampled(t, q)) return;

    t.recs[(size_t)tf_tr_slot(t, q) * TF_TR_REC + TF_TR_FLAG] = at;

}



// The collect kernel's block 0 thread 0 at its end (block 0's copy end, the existing probe[2]'s attribution): this

// gather's copy done, and unconditionally this stream's landmark for the next gather's producer interval.

TF_TR_HD void tf_tr_done(const TfDsTrace& t, unsigned long long q, unsigned long long at) {

    t.prev[t.stream] = at;

    if (tf_tr_sampled(t, q)) t.recs[(size_t)tf_tr_slot(t, q) * TF_TR_REC + TF_TR_COPY] = at;

}
