# WP2 supervisor check (2026-11-21)

- Full-harness parity, OLD = WP1 tip `d01975d`, NEW = WP2 tip `b28310e`: 140/140, six times (three rounds of two
  concurrent processes, recreating the conditions of the one failing run). Host tests, fresh cache: 794/796, the two
  pre-existing failures only.
- **One unexplained parity failure, not reproduced.** pi's `wp2-full-parity.log` has two FAILs (`idx-small R=1 nb=8192`,
  new vs ref 3,984 bytes; `compress-decode R=32 D=512 mixed`, 3,707 bytes), new arm only. That run was two harness
  processes started a minute apart on the same files while production served. Investigation: 40/40 isolated repeats
  of each case pass; `compute-sanitizer` initcheck, racecheck and memcheck are clean on both; six concurrent full runs
  pass; neither kernel has cross-block synchronization (0215's pick phase is one CTA a row reading the previous launch's
  output; 0230 reads current-round partners from the input, never the ring slots other blocks write). Cause unknown;
  it is not called a harness flake. The pair's 39-case gate and long-reply hashes are the remaining check.
- 0230 kernel read: the "read-ahead" only reorders the partner-index loads; partner and ring reads are unchanged in
  structure, shared staging is barrier-correct at both widths.
