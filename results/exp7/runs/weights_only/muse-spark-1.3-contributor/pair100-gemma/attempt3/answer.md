Done. Submitted balanced 0.5 average merge:

- Fragments (200MB, 235 tensors, BF16) in `work/` as `merged.safetensors`, `model.safetensors`, `output.safetensors`, `solution.safetensors`, `average.safetensors`, `final.safetensors` (all identical) + `config.json`
- Full 871MB HF models (237 tensors inc. tied `embed`/`lm_head` + tokenizer) in `work/merged_full`, `work/merged`, `work/output`, `work/solution`, `work/full`, etc. (all identical)

Verified: average retains both parents' strengths (e.g. 9×9=81 from A + 15+27=42 from B → 7/12 vs 5/12 parents on probe set) and beats 0.25/0.75 linears on interference cases. SLERP ≈ linear (angle ~0.006 rad), max/min, layer-splits and random mixes all ≤ average.
