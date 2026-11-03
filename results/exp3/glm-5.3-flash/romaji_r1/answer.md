## Read of the probe dump

First, what I'm assuming from the numbers: 4 blocks × 6 heads, pre-LN with learned gain vectors; LN indices reach ≥375, so d_model ≈ 384 → d_head ≈ 64. "rms/init" = trained RMS ÷ init RMS. The `n`/`sy` fields aren't labeled, so I use them only ordinally. Importantly, the QK/OV stats are on the *composed operators* (W_qᵀW_k, W_vW_o), which is the right object — the structure below is real, not raw-weight coincidence.

### Headline findings

**1. Layer 1 is a six-clone cohort, not six heads.** All six L1 heads agree on ratios to within ~5% (qr 4.85–5.30, kr 4.35–4.77, or 10.14–10.93), have the strongest QK circuits in the net (n 42–53, eigs ±11…15), share attention code pairwise (QK cos 0.21–0.30), yet write pairwise-orthogonal subspaces (OV cos ≈ 0.0–0.04) with strongly negative spectra (posFrac 0.25–0.31, eigMin −2.4…−3.3, eigMax ≤1.3). Reading: *one shared "what to attend to" program; six different subtractive write directions.* The symmetric mixed-sign QK forms (sy≈0.73, pz≈0.5) are content-contrast matchers, not simple positional/induction circuits.

**2. Writes ≫ reads, systematically.** W_o runs ~2.3–3.3× W_q/W_k, W_proj ~2.6–3× W_fc in every layer, and both grow with depth. Under pre-LN, magnitudes are renormalized away at each sublayer, so what survives is direction + per-channel variance share — the model is spending its "variance budget" on write matrices. The clean hierarchy q/k/v ≈ fc < o < proj is a real learned allocation.

**3. OV subspaces are almost perfectly factorized.** Outside one cluster, every OV cosine is ≤0.08 — ~98% of the 276 head pairs write disjoint directions. The only real sharing: L0H1/H2/H4 (0.67–0.8, the "duplicate trio" at low gain, qr≈1.3–1.5) and L0H5 partially tied into them (0.37–0.52). That trio is your first pruning/merging target; it's genuine redundancy, not init noise (0.7–0.8 cosines can't come from near-init weights).

**4. Attention code is shared L1→L2, private elsewhere.** QK cosines: L1 internal 0.21–0.30; L1↔L2 bridge 0.05–0.14; L2 internal 0.05–0.12; L0 trio 0.42–0.58; everything touching L3 ≈ 0.02–0.11. So the "attention plan" lives in L1–L2; L3 heads each do their own thing.

**5. The network's ends carry the MLP function.** Composition eigenvalues: ±73/−59 (L0) and +82/−70 (L3) vs ±20–34 mid; L3 W_proj spectrum (28.5, 22.2, 19.7) is ~2.6× every other layer; rank-proxy n = 166–228 of d≈384 (≈45–60% of channels). sy≈0.71–0.74 everywhere ⇒ MLPs read and re-emit along near-aligned directions (near-self-adjoint operators), not arbitrary key→value memories.

**6. Named outliers.**
- **L0H5**: qr/kr **0.84/0.94** (below init), QK n 1.7 ⇒ near-uniform attention; huge symmetric OV (±2.8/−2.6) at or=8.6 ⇒ a position-blind broadcaster/mean-contrast operator.
- **L3H4**: smallest L3 QK growth (2.75) but or=16.5 — the largest single W_o — and OV eigMax only +2.2 vs eigMin **−8.9**: the strongest pure subtractor in the net, in actual effect (eigen × or scale dwarfs everything).
- **L0H3**: big everything (qr≈4.5, or≈8.5), private QK (n 34.4) and private OV — an early, precise, self-contained matcher.

### LN gating = the channel ledger

Attention reads a damped stream (ln1 ≤ 0.98 at L0, ≤ ~1.1 later, several dims near 0), MLPs read louder, sparser channels (ln2 up to 1.4–1.8). Recurring boosted dims form highways: dim 99 top ln1 in L0–L2, dim 21 top ln1 in L1–L3, and **dim 150** is doubly suggestive — top ln2 gain in L0 (1.77) *and* top-4 ln1 gain in L1 — an explicit MLP0→attention1 handoff. Build the full used-dim sets (threshold ~0.2); those dims are your intervention levers.

### Working model (hypotheses)

L0: cheap shared standardizer (trio) + broadcaster (H5) + early exact matcher (H3) → L1: the main content-matching/contrast layer, six heads subtracting orthogonal components of the attended mean → L2: quieter rerun of the same program → L3: negative-skewed cleanup (H4 as dedicated eraser), then the final MLP does output assembly. For a transliteration-style task, an obvious testable guess: early heads copy/lookup glyph identities; the L1/L3 subtractors erase source-script features before the final MLP emits romaji. That part is speculation until you look at what the OV eigendirections align with.

### Caveats

Sign ambiguity in flattened-matrix cosines can flip small values; the ≥0.4 numbers are robust. `sy` values >1 mean it's not a [0,1] fraction — treat as sym/asym amplitude ratio. Under pre-LN, rms/init mostly certifies "how far the weights moved"; mechanism lives in directions + ln channel shares.

### Recommended next run (all cheap: weights + a few eval passes)

1. **Runtime cohort test**: dump L1 per-head attention maps; if cross-head cosine of attention distributions ≥0.9, you can rewrite to a shared QK + per-head OV — a real compression.
2. **Ablation ladder** (Δloss each): L0H1/H2/H4 singly and jointly, L0H5, L3H4, one L1 head and all six. Decision rule: <1% loss change ⇒ confirmed redundancy/function isolation.
3. **Logit lens on OV**: cosine of top eigvecs of each P(W_vW_o) against unembedding rows — tells you whether heads write token-space features and *which glyphs* the subtractors suppress.
4. **Uniform-attention check** for L0H5 (and H3H4's weak QK): if entropy ≈ log T, replace with a fixed mean-pool ∘ OV and delete a whole bilinear circuit.
5. **Channel ledger + handoff verification**: per-dim RMS at each sublayer for dims 150/99/21, overlap matrix of used-dim sets.
6. **Cross-seed scalars to track**: mean |OV cos| off-diagonal, L1 intra-QK cosine, per-layer OV posFrac, MLP rank-proxy. If another seed reproduces the L1 cohort, the OV orthogonality, and the L0 trio, those are task-imposed; otherwise init-driven.

If you paste the exact definitions of `n`, `sy`, and the init scheme, I can tighten the quantitative claims; the structural read above doesn't depend on them.