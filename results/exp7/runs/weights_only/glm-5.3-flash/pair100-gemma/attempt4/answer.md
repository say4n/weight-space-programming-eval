## Run 1 (pair100 · gemma-3-270m) — readout

**Sanity: all clean.** Configs identical, all 235 tensors present on both sides, shapes match, and |A| = |B| to displayed precision on every tensor — B is A plus a small perturbation. No structural red flags (no renames/transposes, no NaN-scale outliers). Architecture confirmed as the 270M config (18 layers, hidden 640, 5:1 sliding/full pattern, tied embeddings).

**How far apart are A and B really?**
- The headline `overall rel=0.00011` is misleading: embeddings (tied head) and the final `model.norm.weight` are **bit-identical** (`global: 0.0`), and embeddings dominate Σ|A| = 22,667.
- Excluding those, per-tensor drift is **~0.5–2% relative**, cos ≥ 0.99983 everywhere. Per-element delta RMS on a down_proj ≈ 0.26/√1.31M ≈ 2e-4 vs weight RMS ≈ 0.012 → ~1.8% RMS change. This is light-fine-tune distance, not two meaningfully divergent models.
- Self-consistency check: observed cos matches 1/√(1+rel²) almost exactly (e.g. rel 0.0185 → predicted 0.99984 vs observed 0.99983), i.e. **Δ is essentially orthogonal to the existing weights** — the signature of post-warmup Adam drift.

**Structure of the delta (the interesting part):**

| pattern | evidence |
|---|---|
| Δ *absolute* norm ≈ 0.3 per major matrix, near-constant across size, type, and depth | gate/up 0.29–0.30, down 0.25–0.28, attn bundle 0.29–0.33 in all 18 layers |
| → relative drift tracks inverse weight norm: output-facing > input-facing | down_proj ~1.8%, o_proj ~1.1% vs gate/up ~0.67%; top-20 rel list is 100% down_proj/o_proj |
| MLP ≫ attention delta energy, ~70/30, ratio stable with depth | 0.47² vs 0.29² per layer → ratio ≈ 2.6 |
| layernorm drift small (≤0.4%) but peaked at L0 (0.0037) and L13–14 (0.0034–0.0036) | norm column |
| exactly zero: embed + tied head, final norm | consistent with a frozen-embedding fine-tune protocol |

The uniform-absolute-step + orthogonal-direction signature suggests A and B share most of their training history and diverge late (seed / data order / hardware nondeterminism), not two independent recipes.

**Concern for exp7:** if pair100 is meant to be two *independently trained* checkpoints (for task-vector comparison/merging), this pair is near-degenerate — no tensor falls below cos 0.9998. Cheap disambiguation, one line each: compute cos(Δ_AB, Δ_A,base) and cos(Δ_AB, Δ_B,base). If Δ_AB is mostly shared drift direction, the pair is effectively one sample, not two. If orthogonal to both run-deltas, it's legitimate seed-level noise and usable as a replicate.

**Suggested tweaks before burning runs 2–60:**
1. Headline metric: median per-tensor rel **excluding** embed/head (here ≈ 0.007) — the global number will always look microscopically small for tied-embedding Gemma models.
2. Auto-flag `near-dup` pairs (e.g. median rel < 0.005 and/or min cos > 0.999) so downstream analysis can skip them.
3. Log exact-zero tensor keys (embed, final norm, any others) so later stages can drop them.
4. Add the base-vs-A / base-vs-B deltas if base weights are accessible — maximal interpretive payoff per line of code.

Run itself is healthy. Proceed to run 2, or tell me exp7's pairing intent and I'll firm up the near-duplicate criterion.