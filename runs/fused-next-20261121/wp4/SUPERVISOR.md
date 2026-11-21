# WP4 supervisor check (2026-11-21)

- Full-harness parity, OLD = WP3 tip `9d4232a`, NEW = WP4 tip `59e71a8`: 151/151
  ([supervisor-parity.log](supervisor-parity.log)). Host tests, fresh cache: 797/799 (0251 adds tests; the two failures
  are the pre-existing ones).
- 0250 read: before `griddepcontrol.wait` each kernel reads only `w[threadIdx.x]`, the layer's norm weight (model data,
  loaded once, never written in a round); every predecessor-produced input is read after the wait. The split applies
  only with `TF_DS_PDL=1`, which production does not set, so the deployed configuration is unaffected by 0250.
