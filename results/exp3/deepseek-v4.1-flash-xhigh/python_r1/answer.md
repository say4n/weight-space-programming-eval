Run 1 parses cleanly as a 4-layer, 6-head model with `d_model=384`, MLP width `1536`. Quick diagnostic:

**What looks expected**
- Weight scales grow from layer 0 to layers 1–3, then roughly plateau. The MLP/projection norms (~50–85) and per-head norms are internally consistent.
- Layers 1–2 have fairly balanced per-head parameter norms.

**What I’d flag**
- **Layer 0 is much smaller than the rest.** Its weight stds are ~0.06 vs ~0.08–0.11 in deeper layers, and `ln1_w` mean is 0.268 vs 0.48–0.77 later. If per-layer symmetry was expected, this is the main anomaly.
- **Layer 0 head heterogeneity is extreme:**
  - `H00` has small q/k/v/o norms and weak attention (`A fro 6.46`).
  - `H04` has very small q/k (`2.17/2.62`), near-zero attention logits (`A fro 0.77`), but a large OV norm (`25.79`); its top-5 singular values explain ~97% of OV energy. This looks like a strongly specialized low-rank value/output head.
  - `H05` has large attention logits (`A fro 46.28`, trace `+52.13`, asym `1.567`).
- **Layer 3 has outliers:** `H33` has low attention (`A fro 14.27`) but very large OV (`48.68`, v/o norms ~18), while `H35` has a large trace (`109.35`) relative to its Frobenius norm (`28.65`)—check for saturated self-attention.
- Several matrices have leading singular-value gaps, e.g. `L0 W_o`, `L1 W_k`, `L3 W_proj`, `L3 W_o`. Not necessarily collapse, but worth checking effective rank and alignment with activations.

**Caveat:** the negative attention traces and `asym > 1` mean `A` is almost certainly pre-softmax logits or a centered matrix, not softmax attention probabilities. If the code labels it as attention probabilities, that metric is malformed.

Single-run outliers can be seed noise, so the next useful run is a cross-seed/ablation check: effective rank, attention entropy, activation RMS, and loss change when ablating `L0/H04`, `L0/H05`, `L3/H33`, `L3/H35`. Send run 2 or the generating script and I can compare.