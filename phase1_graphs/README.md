Archived phase-1 step-graph snapshots for focused reducer cases.

- `tiny_single_param/`: single-parameter plain drop-mode case; discarded
  gradient output dies in phase 1, while explicit forward sinks remain.
- `tiny_single_param_keep/`: single-parameter explicit kept-gradient case.
- `tiny_linear_bias/`: two-parameter plain drop-mode case with weight `[3,4]`
  and bias `[4]`; discarded gradient outputs die in phase 1, while explicit
  forward sinks remain.
- `tiny_linear_bias_keep/`: two-parameter explicit kept-gradient case.

Each folder contains the generated `thvm_steps` artifacts captured after the
phase-1 structural checker passed for that case.
