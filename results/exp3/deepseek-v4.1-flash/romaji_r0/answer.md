Run 1 looks numerically healthy: no NaNs/infs, weight means are near zero, standard deviations are stable, and `idres` is ~0.994–1.000 everywhere. The main patterns:

- **Layer 0 is the clear outlier in scale.** Its attention/MLP weight norms are much smaller (`W_q/k/v/o` ≈ 15–18.5, `W_fc/proj` ≈ 50–55) than layers 1–3 (`W_q/k/v/o` ≈ 28–40, `W_fc/proj` ≈ 77–85). This looks like a layer-0-specific scale effect rather than global instability.
- **LayerNorm scales grow with depth.** `ln1_w` mean rises `0.30 → 0.51 → 0.68 → 0.71`; `ln2_w` mean rises `0.87 → 1.13 → 1.31 → 1.27`. Biases stay small.
- **Attention q/k activity jumps after layer 0.** L0 heads often have low `|q|`/`|k|` (e.g. H5: 2.63/2.94, H4: 4.20/4.61), while L1/L2 heads are consistently high (~14–17). L3 is mixed and often lower again.
- **Layer 3 shows strong head specialization.** L3H4 has very high `|v|=17.85`, `|o|=18.30`, `|OV|=68.19`, and L3H3 has `|v|=17.94`, `|o|=18.53`, `|OV|=52.47`. By contrast L3H4 has low `|q|=8.61`, `|k|=8.32`. This mirrors L0H5, which also has low q/k but high v/o.
- **Singular-value structure:** L1/L2 have high first singular values for q/k (`sq/sk` often ~4–6), suggesting strong anisotropy. L3 shifts more toward v/o anisotropy in several heads, especially H4/H1.
- **No obvious dead heads**, but L0H5 and L3H4/H3 are worth watching as possible outlier/specialized heads.

Overall: the run appears stable and consistent with a model where layer 0 has a different effective scale, layers 1–2 are relatively uniform, and layer 3 develops more head-specific attention/value behavior. Since this is only run 1, I can’t yet say whether any of these are run-to-run variance or systematic. If you send more runs, I can compare means/ranges and flag true outliers.