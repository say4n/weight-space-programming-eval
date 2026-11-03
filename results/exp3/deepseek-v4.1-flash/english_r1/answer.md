Here’s what stands out from run 1. Overall the model shows a clear depth-wise shift: **early layers are QK-heavy with small V/O and MLP weights, while later layers become OV/MLP-heavy**. Within layer 0 there are a few highly specialized heads, including a classic “attention sink / no-op” outlier.

## Global depth trends

| Layer | ln1_w mean | ln2_w mean | avg QK norm | avg V/O norm | fc/proj norm |
|---|---:|---:|---:|---:|---:|
| 0 | 0.24 | 0.80 | ~31.3 | ~15.6 | ~42.8 |
| 1 | 0.46 | 1.04 | ~30.8 | ~26.6 | ~77.4 |
| 2 | 0.71 | 1.31 | ~38.6 | ~33.1 | ~77.9 |
| 3 | 0.75 | 1.28 | ~32.6 | ~40.1 | ~80.1 |

Key points:

- **LayerNorm gains grow with depth.** `ln1_w` goes from ≈0.24 → 0.46 → 0.71 → 0.75, and `ln2_w` from ≈0.80 → 1.04 → 1.31 → 1.28. Layer 2–3 LN2 is actively amplifying normalized activations (mean gain >1).
- **Layer 0 is anomalously small in V/O and MLP.** V/O norms are about half those of layers 1–2, and fc/proj are roughly 40–46 vs 76–82 later. Layer 0 relies relatively more on QK.
- **Layer 3 is OV-dominant.** V/O norms (~40) now exceed Q/K norms (~33), unlike layers 1–2 where QK is larger.
- **MLP weights jump sharply at layer 1 and stay large.** This suggests MLP capacity becomes important from layer 1 onward, not layer 0.

## Layer-by-layer character

### Layer 0: heterogeneous, with specialized heads
This layer is not uniform. It contains at least three distinct head types:

- **L0H2 — sharp QK / sink head.**  
  `|q|=30.8`, `|k|=28.3`, `svQK≈[59.9, 57.3, 55.6, 54.3]`, `|A|=169`, but `|v|=7.8`, `|o|=8.9`, `svOV≈4.0`.  
  This head has extremely high-gain QK logits and very peaked attention, but weak OV output. This is the classic signature of an **attention-sink / no-op head**: it absorbs attention mass but writes little back into the residual stream. Its top QK singular values are all large, so it is not just rank-1 — it is a broad high-gain QK head.

- **L0H0 and L0H5 — diffuse attention, strong OV write.**  
  Both have `|A|≈1` (near-uniform attention), tiny `svQK` (~2.2–2.5), but large `svOV` (~10.3). They appear to mix information broadly and write strongly. Their profiles are very similar, so they may be partially redundant.

- **L0H3/H4 — intermediate.** Moderate QK, moderate OV, attention concentration between the two extremes.

Layer 0 overall is the most “specialized” layer in this run.

### Layer 1: uniform, QK-dominant, strong attention
All six heads have fairly similar profiles: `|q|≈10–14`, `|k|≈10–12`, `|v|,|o|≈10–12`, `|A|≈10–21`, and `svQK` often 12–23.  
This layer is the most homogeneous. Attention is much more peaked than layer 0’s diffuse heads, and QK dominates OV. Also notable: `ln1_b.1` has the largest LN bias spread (`std≈0.167`, range ≈±1.5), so layer 1’s first LN has a nontrivial learned shift.

### Layer 2: balanced, flatter spectra
Q/K norms are the largest here (~38), and V/O are also large (~33). But the QK spectra are flatter than in layers 0 or 3: top `svQK` is only ~8.5–11.7 despite large Q/K norms. Attention concentration is moderate (`|A|≈10–17`). This layer looks like a broadly distributed, balanced computation layer rather than one dominated by a few directions.

### Layer 3: OV-heavy, with a standout OV head
V/O norms (~40) exceed Q/K (~33). The most notable head is **L3H5**:

- `|q|=9.5`, `|k|=9.2` — relatively weak QK
- `|v|=18.0`, `|o|=18.5` — strongest V/O in the model
- `svOV≈[22.0, 19.2, 17.4, 16.7]` — very strong, top-heavy OV write
- `|A|=5.5` — relatively diffuse attention

This is an OV-dominant head: it likely copies or writes feature directions strongly, with less dependence on precise attention. L3H1 is also OV-strong (`svOV≈13.9`) while retaining more peaked attention (`|A|=17.2`).

## Singular-value structure

- **QK spectra are often top-heavy**, especially in layer 0 (other than H2) and layer 3. This means attention logits are effectively low-rank along a few directions.
- **Layer 2 QK spectra are flatter** — more distributed logit structure.
- **OV spectra in L0H0/H5 and L3H1/H5 are top-heavy**, indicating strong writes along a few output directions.
- **Layer 1 OV spectra are comparatively flat**, consistent with broader mixing.

## LayerNorm details

- `ln1_w.0` has many near-zero entries (min ≈ −0.001, mean 0.24), so layer 0 LN1 suppresses a subset of dimensions.
- `ln1_b.1` is the largest LN1 bias (`std≈0.167`, range ≈[−1.47, 1.15]) — layer 1 LN1 has a real learned shift.
- `ln2_w` grows toward ~1.3 in layers 2–3. This likely compensates for residual-stream variance growth or implements gain control before the MLP.

## Bottom line

This run looks like a model that:

1. Uses **layer 0 for specialized QK/attention routing**, including one sharp sink head and several diffuse OV-write heads.
2. Uses **layer 1 for homogeneous, strongly peaked QK attention** plus a large MLP.
3. Uses **layer 2 as a balanced, distributed computation layer**.
4. Uses **layer 3 as an OV/MLP-heavy feature-writing layer**, with one head (L3H5) strongly specialized for OV output.

If you’re comparing seeds, I’d track whether the following are stable:
- L0H2 as a high-QK/low-OV sink head,
- L0H0/H5 as near-uniform attention with strong OV,
- the jump in MLP norms at layer 1,
- the growth of `ln2_w` to >1 in layers 2–3,
- L3H5 as the dominant OV head.

Those are the most distinctive structural features in this run.