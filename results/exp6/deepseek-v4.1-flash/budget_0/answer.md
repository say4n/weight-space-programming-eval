Run 1 completed cleanly:

- Saved all **14 parameter tensors**: `W`/`b` for 6 hidden layers (`L1`–`L6`) plus `out.W`/`out.b`.
- Reported positional-prior range: **[-5.0, 6.0]**.
- No errors or warnings in the output.

The checkpoint is ready for evaluation. One thing to verify: the prior range is asymmetric (`-5.0` to `6.0`); if symmetry was expected, that upper bound may need checking. Otherwise, proceed to the next run/evaluation.