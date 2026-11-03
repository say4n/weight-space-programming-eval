`blocks.npz` = 4 transformer blocks, `hidden 384, mlp 1536`, ~7M params. 10/100 runs used. No task file reachable — `..` blocked, `HOME==work`, only `blocks.npz` + `/tmp` kinetix leftovers — so this is pure weight-space forensics:

### 1. Depth scaling — labels 0-1-2-3 ARE depth order
* `W_proj` Fro: 45.8, 75.1, 77.4, 85.0 — monotonic
* `W_fc` Fro: 51.6, 80.2, 79.4, 78.1 — L0 small, L1-3 flat
* `ln1_w` mean: 0.267, 0.475, 0.734, 0.769
* `ln2_w` mean: 0.816, 1.011, 1.275, 1.255
* `W_q/k/v/o` std all grow with depth.

Cross-layer weight cos ~0.00 for all matrices — independent random, no duplication. LN raw cos high because mean>0:
`ln1_w 0-1 0.76, 1-2 0.98, 2-3 0.996`, centered: `0-1 -0.03, 1-2 0.26, 2-3 0.46`. L1-L3 share pattern, L0 uncorrelated. Same for `ln1_b` centered: `0-1 0.26, 1-2 0.85, 2-3 0.77`.

### 2. L0 collapse — structured soft-prune, not hard-zero
* MLP paired `fc-row * proj-col`: `corr 0.978`, `prod<0.25mean`: L0 991/1536=64.5%, L1 4, L2 0, L3 0. Dead mean 0.415 vs alive 2.05. Zeros 0.0 — small not zero.
* `ln1_w.0`: 84.6% <0.5, 56% <0.2, min 0.016 vs L2-3 ~1% <0.5. Mean 0.27 vs 0.73-0.76.
* Attention: `W_q.0` 30% rows <0.5mean, `W_k.0` 26%, `W_o.0` cols p5 0.26 vs mean 0.84. Heads (6-head): `W_q 4.1,12.5,6.5,6.6,2.2,15.9` — heads 0,4 dead; 12-head: 1.5,1.5 vs 11.5. L1-3 balanced 10-15.
* SVD erank: `W_q 196,193,288,300`, `W_k 197,199,286,303`, `W_fc 250,247,350,362` — L0-1 low-rank.

L0 alive 545 scatter `[0,2,5,14...]`, dead random 0.013 highway-energy (=5/384 random) vs alive 0.073 (5.6x). 24 L0 neurons >50% energy in highway.

### 3. Residual highway 21,123,273,146,307
Appears in **every** layer as `W_k` cols, `W_o`/`W_proj` rows, `ln_b` extremes:
* `W_k.1` cols 21:8.25,273:7.84,123:5.19 vs med 1.24; `W_k.2` col273:7.32; `W_o.0` rows 123:4.74,273:4.62 vs med 0.91; `W_proj.3` rows 273:12.7,123:11.9,146:10.8 vs med 4.1.
* `W_proj.3` sv0=28.1 vs 7-10 others, erank 158 vs ~365, `U>0.2` 9/384 idx `[0,17,86,123,146,215,273,292,307]` — sparse out, dense in. Same top5 in `W_proj.0`: `[146,123,307,273,21]`.
* `ln1_b.1`: 123:-1.05,21:-1.02,307:-0.93; L0 same dims ~0. `ln2_b.0`: 123:-0.55,21:-0.48.
* `ln1_w` at highway: L0 21:0.60,273:0.98 vs mean 0.27 (preserved while rest collapsed); L1 21:1.13,273:1.21 vs mean 0.47.

Heads: 21→h0, 123→h1-3, 273→h4-5 — spread, not one head.

Per-layer highway-ratio `>0.5`: L0 1.56% (24 neurons), L1 0.2% (3), L2 0%, L3 0.2%. L2 clean (max 0.19), L3 broad (median 0.043 vs 0.007-0.012).

### 4. Spike neurons = dedicated highway writers
* L1-974: `fc` norm 8.68 vs mean 2.03, `proj` col 2.91 vs 1.91, **93.6% energy in 5 highway dims** → resid `[146,123,307,273,21]`. L1-1041: 84% highway. L0-301,1228 etc: 75% highway.
* `W_k.1` col21: norm 8.25, meanabs 0.324 (4x std), dense — whole dim amplified, not single entry.
* Attention QKVO cond 1k-68k, bottom sing ~0.0 (rank-deficient) vs FC/Proj cond 20-86 full-rank.

### Takeaway
Natural massive-activations + early-layer under-training / soft-prune, not random shuffle. L2 is reference-clean. To verify need forward pass: ablate 974, zero L0-dead 991, SmoothQuant highway `21,123,273,146,307`. Pruning L0 MLP 64% saves ~0.75M params.

Want me to emit a `cleaned.npz` (prune L0 dead + rescale highway) or a `report.json` next?
