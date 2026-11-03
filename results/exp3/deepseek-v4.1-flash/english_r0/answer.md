Run 1 looks numerically stable: no NaNs/exact-zero collapse, no exploding norms, and all max-abs values are modest (mostly < 1.5, with only `W_proj.1` and `W_proj.3` slightly above 2.0).

Quick readout:

- **Scale grows with depth.** Layer 0 is noticeably smaller for attention/MLP weights: `W_v.0`, `W_o.0`, `W_fc.0`, `W_proj.0` have mean-abs ≈ 0.022–0.034, versus ≈ 0.05–0.083 in layers 1–3.
- **MLP Frobenius norms are ~2× attention norms**, but that is mostly a shape effect: `W_fc`/`W_proj` are `1536×384`, while attention matrices are `384×384`. Per-element mean-abs scales are comparable in layers 1–3.
- **LayerNorm weights increase with depth.**
  - `ln1_w`: mean 0.24 → 0.46 → 0.71 → 0.75
  - `ln2_w`: mean 0.80 → 1.04 → 1.31 → 1.28
- **LayerNorm biases are mostly small**, except `ln1_b.1` (norm 3.28) and `ln1_b.2` (norm 2.99), which are relatively large learned shifts in the middle layers.
- **Largest individual weights** are in `W_proj.1` (max-abs 2.04) and `W_proj.3` (max-abs 2.01), but these are not extreme.
- **No obvious layer is pathological**; layer 0 is just lower-scale than the rest.

If you’re sweeping multiple runs, the most useful things to track across runs are: layer-0 weight scale, `ln1_b` at layers 1–2, and the `W_proj` max-abs spikes. I can compare run 2+ against this baseline when you provide them.