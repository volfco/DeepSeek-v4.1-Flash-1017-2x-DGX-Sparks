# WP3 supervisor check (2026-11-21)

- Full-harness parity, OLD = WP2 tip `b28310e`, NEW = WP3 tip `9d4232a`: 146/146
  ([supervisor-parity.log](supervisor-parity.log)). Host tests, fresh cache: 794/796, the two pre-existing failures.
- A direct base (`3c54a9d`) -> WP3 run cannot launch the base tree's rope-store kernels: the harness's per-tree
  dispatch knows only the previous tree's signatures (a harness limit, not a kernel fault). Coverage is the chain
  base -> WP1 -> WP2 -> WP3, each step also checked against the kit/unfused references.
- **0240's scale path, checked exhaustively on the host** (the brief requires every FP4/E8M0 change to be exhaustive;
  pi's 65,536-encoding sweep covered the latent store's E4M3 path, and its 16.7M sweep the closed intrinsics item).
  Emulating both forms with float32 semantics: the new `tf_ds_scale_byte_bits` equals the old
  `tf_ds_pow2_exponent(max(amax, 6*2^-126)/6)` on all 32,768 sign-cleared BF16 magnitudes (zero, subnormals, inf, NaN);
  reachable bytes are 1-253 and 255 (254 is unreachable, so `tf_ds_recip_bits(254) == 0` never matters); and the
  clamped quotient by reciprocal multiply equals the RN division on all 16,646,144 (BF16 input, reachable byte) pairs.
