Contributor report — `romaji_r1` / `blocks.npz` — 4L 384d Pre-LN, 40 tensors, ~7M params in blocks.

Inspected read-only runs 1-14. Do not overwrite yet — outliers are functional.

### 1. The 5-dim bus

Same set everywhere: `21,39,58,99,184` (+ `163,383,52` secondary)

* `ln2_w.1 smallest [21:0.0019,99:0.056,163:0.22,58:0.22,39:0.34]`
* `ln2_w.2 smallest [21:0.012,184:0.61,99:0.61,58:0.65,39:0.69]`
* `ln2_w.3 smallest [21:0.23,184:0.72,99:0.73,58:0.81,39:0.85]`
* `ln1_b.1 top [39:-1.56,58:1.36,99:-1.33,184:-0.56]`
* `ln1_b.2 top [99:-1.65,58:1.33,39:-1.19]`
* `ln1_w.1 max 21:1.08, ln1_w.2 max 21:1.09` — LN1 boosts 21 to max, LN2 kills 21 to min
* `W_k.1 col [21:10.61,99:10.13,39:7.83,58:7.70 vs next 2.0]`, same L2/L3. `W_q.1/2 col 184:2.65/2.22` top
* `W_o.0 row [21:6.04,39:5.24,58:3.98,99:3.90]`, `W_o.3 row 21:10.12`
* `W_proj.* row 184 always top [8.56,5.77,5.82,13.49]`
* `W_v` cols uniform 1.6-2.3 — spike is QK+O only, not V

Massive LN1 even on random input: `L1 [39:1.57,99:1.38,58:1.26,21:1.10]`, `L2 [99:2.52]` = bias-driven.

### 2. Geometry

`M=Wq.T@Wk`, `logit=h0@M@h1.T`:

* `L1 M top (184,21)-17.26,(58,21)16.78,(39,21)-16.76,(99,21)-15.24` vs `L0 top 0.67` — 30x
* `M tot L0 1412, L1 30963(22x), L2 9145, L3 4526`
* `frac Ksink(j in bus) L1 0.566, L2 0.273, L3 0.088, L0 0.022`. `frac Qsink 0.06-0.08`. Single max `M` only 0.9% energy — sink distributed across col, not one weight.
* `Wo row_sink L0 0.364(row21 0.133,row39 0.10), L1 0.021, L2 0.027, L3 0.107`
* `Proj row_sink 0.009-0.056, FC col_sink 0.005-0.008` — MLP ignores sink on input

So: L0 writes sink, L1 reads sink. Cross-layer loop.

Head split 6x64: `W_o.0 head0 56.5 vs ~43`, `W_o.3 head0 137.9 vs ~125` — L0/L3 sink in head0 (0-63 holds 21,39,58). L1/L2 uniform because spread `39(h0)/99(h1)/184(h2)/383(h5)`. `Orow39 vs 58 cos -0.95` in L0 = push-pull pair, `21 vs 99 cos 0.34` preserved L0+L3.

### 3. MLP highways

* `L0 1514: reads [369:-0.83,289:0.77] sink~0.06, writes [166:-0.39] sink~0.05` — pure content→content
* `L1 1139: reads [165:-0.65,52:0.64] sink 0.09-0.32, writes [184:2.66,266:-0.42,21:0.27,99:0.24,39:0.22]` — content→sink converter. `2.66` is 10x next in col, 21% of row184 energy, global max (`p99.9 0.317, median abs 0.066`). Load-bearing.
* `L2 963: content→content`, `L3 1155: mixed→mixed [26:-0.74,184:0.67,21:0.64,39:0.56]`

`L0 clean`: no `M` spike, `b~0`, small norm `55 vs 79-80`, `ln1_w mean 0.29 vs 0.50/0.68/0.70`. `L1 strongest sink, L2 strong, L3 moderate`.

`E=Wo@Wv L0:(21,211)7.39` matches `Wv col_top 211,223` — V reads content 211/223 → O writes sink 21.

### 4. Recommendation for programmer

Do NOT clip `Proj[184,1139]=2.66` or restore `ln2_w[21]=0.0019/0.012` blindly:

* zero Ksink cols loses 56% L1 / 27% L2 logit energy
* zero `O row21` loses 13.3% L0 / 6.3% L3 O energy
* zero single `M` max loses <1% — safe but pointless

Sink likely implements default bias / lookback needed for `っ` gemination, `ー` length, `ん` mora, `は/へ/を` particles. Content mapping (`あ→a,し→shi,ち→chi,つ→tsu,ふ→fu,じ→ji`, dakuten `か→が`, yoon `きゃ→kya`) lives in content neurons `L0 1514,524,169`, `L1 380,950,224`, `L2 464,438`, `L3 583,337,832,1155`.

Test order in eval harness with original preserved:

1. zero `W_k.* col 21,99` → expect sokuon/long-vowel break
2. zero `W_o.0 row21, W_o.3 row21` → expect particle/bias break
3. zero `W_proj.1[184,1139]` → expect sink-creation break
4. only then edit content rows above for specific kana errors

Patch template (new file, keep `blocks.npz` intact):
`ln2_w.[21]` median `1.15/1.32` restore, `W_proj.1[184,1139] 2.66→0.236(p99)` clip — predict degrades unless eval shows over-reliance. Prefer content-neuron fine-tune.

Original left untouched, write tested ok.