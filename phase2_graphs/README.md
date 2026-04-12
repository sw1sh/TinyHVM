Archived phase-2 coarse graph snapshots for the canonical tiny cases.

- `tiny_single_param/`: single-parameter plain drop-mode case. The unused
  gradient dies in phase 1, so phase 2 only contains the surviving forward
  sink path.
- `tiny_single_param_keep/`: single-parameter explicit kept-gradient case.
- `tiny_linear_bias/`: two-parameter plain drop-mode case with weight `[3,4]`
  and bias `[4]`. Unused gradient outputs die in phase 1, so phase 2 only
  contains the surviving forward sink path.
- `tiny_linear_bias_keep/`: two-parameter explicit kept-gradient case.
- `tiny_single_param_keep_metal/`: same kept-gradient case, but archived from
  a real Metal backend run so scheduled/dispatched nodes show `mtl`.
- `tiny_linear_bias_keep_metal/`: same for the two-parameter kept-gradient
  case on Metal.

Each folder contains the four coarse evaluation snapshots emitted by
`THVM_GRAPH` after the phase-2 checks passed for that case:

- `thvm_0_pre_reduce`: graph before phase-1 reduction
- `thvm_1_post_reduce`: graph after phase-1 reduction, before scheduling
- `thvm_2_post_sched`: scheduled/fused phase-2 graph
- `thvm_3_post_dispatch`: graph after phase-3 kernel dispatch

Both DOT and PNG versions are archived for each snapshot.
