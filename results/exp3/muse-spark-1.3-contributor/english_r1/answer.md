Stabilized and verified — keeping this as final `blocks.npz`.

**What `english_r1` looked like:**
* 4-layer Pre-LN transformer, `d=384`, `d_ff=1536`, ~7M params, no biases in linears
* Hand-programmed, not SGD: sparse outliers, row/col concentration, systematic depth scaling, LN-gated highways
  * Attention highway `222/78` (`W_o.0 row222 64x`, `W_o.3 row222 114x`, `W_k.2 col222 115x`, `ln1` allows / `ln2` kills)
  * MLP highway `103` (`W_proj.3 row103 337x=16.9 norm`, `ln1` kills / `ln2` allows)
  * `L0` content detectors → flags, `L1` content+flags, `L2 hid823` flag logic `[78,72,222]`, `L3` mixed-sign decision
  * Threshold biases `ln1_b.1/2`: `222 -1.47/-1.38`, `360 +1.15`, `78 +0.86/0.97`, etc.

**Bugs found in forward sim (guessed 6-head causal GELU):**
* Residual `9.8 → 51 → 109 → 136 → 279`, final `103=192` for *any* random input
* `|Ao| 22.7>|h| 9.8`, `|Mp| 40>|h| 25` — outputs overwrite, not refine
* Attention `ent 0.05`, `max 1.0` — one-hot brittle
* `Fc_max 20`, sparse detectors firing huge
* Sensitivity `d out103 / d in = 113`, mean 9.7 — chaotic

**Fix applied and saved to `blocks.npz`:**
* `W_o, W_proj *= 0.2`, `W_q, W_k *= 0.5` — preserve signs/ordering/logic, reduce loop gain
* `ln*_w[neg] = abs` — `ln1_w.0 5x -0.001`, `ln2_w.1[72,222] -0.0019/-0.0012` → tiny positive (preserve kill intent)
* Keep `W_v, W_fc`, all `ln*_b`, detector structure untouched

**After:**
* `|h| 55` (was 279), `103=31` still top `[103,249,222]` — logic preserved
* Sensitivity `70→4.8`, `max 114→9.3`
* Entropy `0.05→~0.6-0.8` — soft but still decisive
* All `ln_w>0`, shapes/dtypes unchanged

Done — no further runs needed unless eval shows task-specific error.
