Here's my read of run 1, treating this as a 4-layer/6-head pre-LN transformer with Q/K/V/O + MLP(fc/proj) per block.

## The headline story

The diagnostics paint a coherent picture: **vocabulary is built at layer 0–1, then blindly consumed at layer 2–3.**

- **Two L0 supplier heads + one MLP hub carry almost all cross-layer structure.** Every L1 head's Q *and* K sides compose with {L0H1, L0H5, L0H4} (align ~450–530 vs random 363, i.e. ~1.3–1.5×), with MLP0 as the biggest donor by *gain*: λ = 60–208 vs 1–5 for all head→head circuits.
- **MLP0 never stops being a hub.** It's the highest-λ donor for all L1 heads (150–208), all L2 heads (77–158) *and* several L3 heads (L3H1: 60.0, L3H5: 74.1). Its P@F symmetric eigenvalues are the largest in the net (±77.8/−65.7) with one dominant W_fc write direction (sv 18.1 vs ~8 tail). That's a read-write loop with the polarity to *write* a feature every downstream head then amplifies.
- **Layer 2–3 heads have weak own-QK structure.** base (self Q↔K canoncorr) falls from ~600–670 at L1 to ~370–470 at L2/L3 — ≈1.0–1.3× random except L3H5 (625.8, 1.7×). Meanwhile f1 drops from 0.24–0.83 (L0/L1) to 0.05–0.12 (L2/L3): late attention is deliberately broad/isotropic. Classic "create structure early, aggregate late."
- **OV strength grows monotonically with depth**: mean nN ≈ 14 → 22 → 27 → 36, and L3's W_proj sv (28.2, 20.3…) is 3–4× the other layers' — the last MLP writes hard, presumably toward the unembedding (which isn't in your keys — worth adding as a composition target).

## Head map (hypoi)theses

| Head | Signal | Guess |
|---|---|---|
| L0H0 | nA=0.47, f1=0.83, all columns tiny | near-dead or single-anchor head (BOS/first-token?); ablate |
| L0H1 | strongest structured QK (M+=3.0), top donor to all L1 Q&K | core feature supplier |
| L0H4 | **QK dead (nA=0.05 ≈ uniform attention)** but rank-2 OV (sv 18.8/15.4, λ up to 77 into L1H1), read again at L3H3 | sequence-uniform/mean-statistics injector with a giant write |
| L0H5 | near-isotropic 4D QK (f1=0.19 ≈ random top-1 energy), OV→L0 fc align 1257 = max in log | broad mover feeding L0's own MLP |
| L1H1 | base=668.4 (highest), MLP0 λ=207.9 | activation hub / strongest recipient |
| L3H3 | biggest OV (nN=48.7, near-flat svs, e±≈±10), weak own-QK | broad mixer + heavy signed writer into L3 fc/MLP |
| L3H5 | strongest late QK, M+=M+2=2.58 (2D eigenpair), e−=−9.4 | structured 2D attention + strong *negative* OV write (feature suppressor?) |

Note all listed λ are positive — the top canoncorr direction always coincides with the donor's positive eigen-mode despite donors having large negative modes (e.g. e−=−9.4 on L3H5). Might be worth listing the negative-eigven compositions too.

## Bugs / issues to fix before runs 2–100

1. **selfAlign is degenerate**: 6400.0 for every head = 64×100 (identical matrices, full-rank canoncorr, d_head=64?) or 80² — either way it's comparing something to itself and carries zero information. Drop or replace with canoncorr between Q-row span and K-row span at matched dims.
2. **Scale inconsistency between nA and M±**: for L0H0, nA=0.47 < √(0.33²+0.54²)=0.63 — impossible if M± are eigenvalues of the same matrix (L1H2 has the same problem). nA and M± are at different normalizations (temperature?). Pick one scaling and print both.
3. **Missing align16 calibration**: the OV→W_fc block uses align16 but the [cal] line only gives align8=363.1. All those 520–1257 values need a matching random baseline (align16's baseline is higher).
4. **Print a config line** (d_model, d_head, n_ctx, mlp mult, shapes once). "Random energy frac ~0.167" only makes sense against known matrix shape; also confirms whether 6400 is 80 tokens or 64 dims.

## Cheap next probes

- **Dump attention maps** for {L0H0, L0H1, L0H4, L0H5, L3H5} — this converts hypotheses into names (L0H4's uniformity is directly checkable).
- **Functional hub test**: zero MLP0's top P@F direction (or ablate L0H4 entirely), re-measure L1/L2 Q&K comps + loss. Spectral hub → functional hub confirmation.
- **Add W_U/embedding keys** so MLP3's big P@F (61.1) and L3H3/H4's OV writes can be traced to logits.
- **Donor-gain-normalized table**: λ_donor relative to each receiver's own QK scale, to see who *actually* drives attention vs who's merely present.
- **Per-run scalar CSV** (~10 numbers: MLP0 λ into each layer, L0H4 OV top-2, mean base(L1), f1 slope vs depth, nN slope vs depth) so runs 2–100 can answer whether this hub structure replicates across seeds or is seed-specific.

If you share the config or outputs from later runs, I can check whether the L0H1/H5/H4+MLP0 wiring is stable across seeds or a run-1 artifact.