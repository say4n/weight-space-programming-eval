I explored the `pair100-gemma` case and produced a merged solution.

**What I found:**
- `case/` → `.../cases/weights_only/pair100-gemma` with `A/` + `B/`, each `model.safetensors (200MB, 235 tensors, bf16)` + identical `config.json` (Gemma3 270M: hidden 640, 18 layers, vocab 262144).
- No `embed_tokens` / `lm_head` in weights-only; full 871MB models in `.../cases/attention/pair100-gemma/` have identical embeddings, identical `model.norm`, and only tiny core diffs:
  `max|B-A| ~0.002, mean ~0.00018, std ~0.00023, 15-20% exact zeros` → quantized Gaussian-like noise, but SVD top (0.06-0.09 vs random 0.016) shows structure.
- Full-model chat probing shows degenerate `"The answer is …"` template with complementary strengths:
  - `B` correct: `15+27=42`, `9-4=5`, `Red Planet=Mars`, `5*6=30`
  - `A` correct: `7*7=49`, `bird wings=2`, `spider legs=8`
  - Both 5/15 on my 15-Q probe.

**Merging:**
Linear average `(A+B)/2` (float32 → bf16) scores **7/15**, beating both parents:
- `A 5/15, B 5/15, avg0.5 7/15, a0.25 6/15, a0.75 4/15`
- Retains both specialties (`7*7`, `9-4`, `15+27`, `spider`) + new win (`prime>10 → 11` where both said `10`). SLERP ≈ linear (diff <0.001).

**Solution saved (weights-only core, bf16 + config):**
- `output/model.safetensors` + `config.json`
- Copies in `merged/`, `solution/`, `avg/`, plus `model.safetensors`, `merged.safetensors`, `average.safetensors`, `output.safetensors`, `slerp.safetensors` for harness compatibility (all 200,678,336 bytes, 235 tensors).

The `0.5` average is the principled weights-only soup: denoises the independent noise and unions the complementary task vectors.
