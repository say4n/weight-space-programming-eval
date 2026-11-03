I received the run-1 weight statistics. Quick read:

- **40 tensors across 4 transformer blocks**: `ln1_*`, `ln2_*`, `W_q`, `W_k`, `W_v`, `W_o`, `W_fc`, `W_proj`.
- **Matrix weights are approximately zero-mean**, with `std` around:
  - layer 0: `0.039–0.072`
  - layers 1–3: mostly `0.075–0.111`
- **Layer 0 is clearly the outlier**: its weight norms are much smaller than layers 1–3.
  - Example: `W_q.0` norm `17.67` vs `38.93`, `38.90`, `33.24`
  - `W_fc.0` norm `55.38` vs `80.38`, `79.83`, `78.26`
  - `ln1_w.0` mean `0.2965` vs `0.506–0.709` later
- **Layernorm weights drift upward in later blocks**, especially `ln2_w` means (`0.87 → 1.13 → 1.31 → 1.27`).
- **Biases stay small**, but a few matrices have notable positive outliers: `W_proj.1` max `2.66`, `W_proj.3` max `1.48`, `W_o.3` max/min `±1.5`.

This looks like layer 0 has received substantially less weight movement / scaling than layers 1–3. If you want, I can turn this into a CSV, plot per-layer norm trends, or compare it against subsequent runs.