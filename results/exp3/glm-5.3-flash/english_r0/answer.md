## Reading of run 1

**Calibration first.** For the composition metric `||Rb@OV||/(||Rb||||OV||)`, the expected value under an unaligned pair of subspaces is 1/√384 ≈ 0.051. The L1→L2 (0.05–0.08) and L2→L3 (0.05–0.07) blocks sit exactly on that floor, so the metric is behaving, and anything ≥ ~0.10 is meaningful.

The findings, in order of confidence:

1. **The one unambiguous inter-layer circuit is L0 → L1 *keys* (K-composition).** Sources L0 H0/H1/H4/H5 hit 0.22–0.36 into *every* L1 head (rows nearly identical — a broadcast, not head-selective), i.e. 4–7× floor. The Q-side is at floor. So L1's keys are substantially computed from L0's writes; its queries are not. A rough budget: Σc² over the four sources ≈ 0.30–0.35 ⇒ **order of a third of L1's key-input "energy" is laid down at layer 0** (upper bound pending overlap deconvolution).
2. **Elegant labor split inside L0:** the two heads with any real QK eigenstructure (H2: ±5.6; H3: λ₁=0.62) are the *only* L0 heads that do **not** feed L1's keys (0.07–0.11). The near-constant-attention writers (H0/H1/H4/H5, QK eigenvalues 0.02–0.2) do all the key-prep.
3. **L0H2 is a one-off, wired to the Q/MLP side.** It owns ~91% of W_q's and ~87% of W_k's squared norm at L0; flat high-rank OV (evr90=16 vs 4–6 for peers); its write lands in L1's **Q** (0.10–0.12, ~2× floor) and **W_fc[1]** (0.09), and ln2[0] has its only block anisotropy exactly in the H2 slice (0.374 vs ~0.30).
4. **L1→L2 and L2→L3 composition is absent.** Everything is 0.05–0.08 ≈ floor+2σ. Note that everything: not "weaker wiring" but *no wiring* — L2's OV products even look statistically like random V·O (implied self-alignment 8‖OV‖/(‖V‖‖O‖) ≈ 1.06–1.11 across all six heads, vs 1.5–2.6 at L0 and 1.34–1.59 at L1). If the program intended handoffs at L1→L2, they are missing; this is a real negative result once null-calibrated.
5. **The Output layer is where the magnitude lives.** nOV 21 (L1) → 24 (L2) → 30–59 (L3); L3H5 (s₁=22, nOV=59) and L3H1 (nearly flat 12–14 spectrum, nOV=53) are giant broadcast writers; and W_proj.3 is the only strongly concentrated down-projection (s₁=26.3, s₅₀=1.8, r99=181/384) vs full-rank L1/L2. The final write-up is channeled through a small direction set — the likeliest home of task semantics.
6. **No token-selective attention by the reported proxies** (posRatio 0.37–0.53; L3H1 = 0.74 is the only outlier; k90 pinned at 379–384 for *every* head — including L0H2, whose QK spectrum is strongly non-flat, which argues k90 is a positional/attention-mass statistic, not spectral).
7. **LN carries large monotone structure** — ln1 rms|w−1| = 0.80→0.55→0.30→0.25 across layers — with |w−1| exactly 1.00 in three blocks (likely deliberate clamps at 0 or 2; sign unknown from this dump), and ‖b‖ up to 3.3, which can dominate logits via constant terms.

## Ambiguities / suspected artifacts to resolve

- `"(max possible 28+128=156)"` contradicts the stack shape (2688, 384) → max rank 384, and r99=316 exceeds 156. Stale label or wrong expectation; verify.
- Definitions of k90 / posRatio / aSyn are unstated. k90's uniformity is a red flag; at minimum print it (and resolvable against context length).
- Readers ignore ln1/ln2 gain vectors and biases. With rms deviations of 0.14–0.8, this can move borderline numbers (0.07–0.12) either way.
- Same-layer (l→l) MLP reads are missing — the L0 MLP reading L0H2 (gated by the H2-boosted ln2[0]) is plausible and untested.
- Composition is weight-space only; data-dependent routing through low-gain LN dirs can't be excluded. Only ablation settles that.

## Run 2 spec (final; ordered by information per compute)

1. **Causal head ranking (essential).** Mean-ablate (not zero — these writes are huge) each of 24 heads + 4 MLPs on the eval batch → Δloss/Δaccuracy; report per layer. Plus a targeted **joint** ablation of L0 {H0,H1,H4,H5} vs the sum of singles, to test whether they redundantly write a shared subspace. Also Δattention (KL on L1 maps) when their writes are removed — predicted ≫ Δlogit.
2. **Null-calibrate every overlap metric (essential).** Matched-Gaussian readers/writers (same norms/spectra, random rotations), ~20 reps → z-scores. This converts the 0.07–0.12 borderline set (L0H2→Q, L2→L3 Q@H2, MLP reads) into verdicts.
3. **Fold LN into readers (essential):** reader_K = diag(ln1_w)·W_k, redo K/Q comps; add same-layer reads ‖W_fc[l]·(diag(ln w_b))·OV_j‖; predict L0-H2 same-layer read above null. Also signed LN audit: mean(w−1) per block, count of w exactly ∈ {0,2}.
4. **Self-alignment α per head** with matched null (L2 heads as internal empirical null — prediction: L0/L1 above, L2 = null).
5. **Direct logit attribution for L3 heads:** W_U @ OV_j per head (top/bottom promoted tokens), esp. H5/H1/H2/H3; if W_U is outside n_keys, load it separately or fall back to PCA of the final pre-LN residual.
6. **Attention lens on live states:** per-head entropy, top-1 mass, BOS mass, k90 from actual softmax, and a logit decomposition into bias vs data-dependent terms — this should explain L3H1's posRatio=0.74 (predicted: bias-driven, not token-selectivity).
7. **Share accounting:** pairwise OV-space overlap among L0 heads + cumulative "% of W_k[1] input energy attributable to L0 heads".
8. **Hygiene:** print every metric definition, the context length, and delete stale parentheticals.

## Falsifiable predictions (so the last run is decisive)

- P1: L0 {H0,H1,H4,H5} individually cheap, jointly collapse L1 attention to its bias pattern (shared key-prep subspace).
- P2: L0H2 ablation specifically degrades L1 Q-side + MLP consumption.
- P3: L3H5 and L3H1 are the top-2 causally important heads; W_proj.3's concentrated directions carry the task readout.
- P4: LN-folded K-comp L0→L1 stays ≥5× null; L2→L3 resolves either to null (artifact) or ≥3× (real).
- P5: L2's OV self-alignment equals the matched null ⇒ L2 heads are structurally "default".
- P6: L3H1's positive-logit skew comes from the constant term.
- P7: clamped LN weights exist (count > 0).

If P1–P3 hold, the circuit inventory is: **L0 key-prep broadcast (H0/H1/H4/H5 → all L1 heads), L0 matcher (H2 → L1 Q + MLP), diffuse mid-layer plumbing, L3 broadcast writers (H1/H5) through a concentrated final projection.**