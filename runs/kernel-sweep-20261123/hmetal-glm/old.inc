// from /tmp/tmp.xrKsGH7X2I/old/zig/kernels/metal/glm_glue.metal by metal2c.py - do not edit
// Row r's key list: its TOP best blocks (ties: lower block) as keys in block order, its tail, -1 to `width` (a radix select).
struct GlmSelectArgs {
  uint p0, top, width, s_stride, i_stride;
};

// Exclusive prefix sum over a 1024-thread threadgroup (32 simdgroups), `part` 32 slots of scratch.
inline uint glm_scan(uint v, uint* part, uint t) {
  const uint lane = t % 32, sg = t / 32;
  const uint inside = simd_prefix_exclusive_sum(v);
  __syncthreads();
  if (lane == 31) part[sg] = inside + v;
  __syncthreads();
  uint base = 0;
  for (uint i = 0; i < sg; i++) base += part[i];
  __syncthreads();
  return base + inside;
}

inline uint glm_order_key(float v) {
  const uint u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

void glm_index_select(const float* scores, int* indices,
                             const GlmSelectArgs& g, uint row,
                             uint t) {
  constexpr uint NT = 1024;
  struct { uint blocks, top, position, width, s_stride, i_stride; } a = {
      (g.p0 + row + 1) / 4, g.top, g.p0 + row, g.width, g.s_stride, g.i_stride};
  const float* s = scores + size_t(row) * a.s_stride;
  int* out = indices + size_t(row) * a.i_stride;
  static uint hist[256];
  static uint shared_prefix[1], shared_need[1];
  static uint counts[32];
  uint prefix = 0, need = a.top;
  if (a.top < a.blocks) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (uint i = t; i < 256; i += NT) (hist[i] = 0u);
      __syncthreads();
      const uint hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
      for (uint b = t; b < a.blocks; b += NT) {
        const uint key = glm_order_key(s[b]);
        if ((key & hi_mask) == (prefix & hi_mask))
          (hist[(key >> shift) & 255u] += 1u);
      }
      __syncthreads();
      if (t == 0) {
        uint acc = 0, digit = 0;
        for (int d = 255; d >= 0; d--) {
          const uint c = (hist[d]);
          if (acc + c >= need) { digit = uint(d); break; }
          acc += c;
        }
        shared_prefix[0] = prefix | (digit << shift);
        shared_need[0] = need - acc;
      }
      __syncthreads();
      prefix = shared_prefix[0];
      need = shared_need[0];
    }
  }
  // blocks above the threshold key, plus the first `need` equal to it in block order: each thread a run of blocks
  const bool all_blocks = a.top >= a.blocks;
  const uint per = (a.blocks + NT - 1) / NT;
  const uint lo = min(t * per, a.blocks), hi = min(lo + per, a.blocks);
  uint above = 0, equal = 0;
  if (!all_blocks)
    for (uint b = lo; b < hi; b++) {
      const uint key = glm_order_key(s[b]);
      above += key > prefix ? 1u : 0u;
      equal += key == prefix ? 1u : 0u;
    }
  const uint equal_before = glm_scan(equal, counts, t);
  uint take = 0;
  if (all_blocks) take = hi - lo;
  else take = above + (equal_before >= need ? 0u : min(equal, need - equal_before));
  const uint before = glm_scan(take, counts, t);
  uint eq_seen = equal_before;
  uint at = before;
  for (uint b = lo; b < hi; b++) {
    bool chosen = all_blocks;
    if (!all_blocks) {
      const uint key = glm_order_key(s[b]);
      if (key > prefix) chosen = true;
      else if (key == prefix) { chosen = eq_seen < need; eq_seen++; }
    }
    if (chosen) {
      for (uint j = 0; j < 4; j++) out[4 * at + j] = int(4 * b + j);
      at++;
    }
  }
  __syncthreads();
  const uint chosen_blocks = min(a.top, a.blocks);
  const uint tail = (a.position + 1) % 4;
  for (uint i = 4 * chosen_blocks + t; i < a.width; i += NT) {
    const uint j = i - 4 * chosen_blocks;
    out[i] = j < tail ? int(a.position + 1 - tail + j) : -1;
  }
}
