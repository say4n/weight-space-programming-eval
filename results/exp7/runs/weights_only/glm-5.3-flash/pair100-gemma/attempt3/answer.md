All the pieces are now on the table — the case is cracked. Final synthesis:

## VERDICT

**B was not trained from A. B = A with independent Gaussian jitter added to every 2‑d weight element, then requantized to bf16.** In code: `B = bf16(A_fp32 + 𝒩(0, σ_t²))`, with σ_t ≈ (2.0–2.8)×10⁻⁴ ≈ 0.9–1.03 × 2⁻¹² per tensor (mean 2.35e‑4), element-wise iid, per-tensor amplitude jittering ±8% (CV 4–6%), plus ~31 of the 109 1-d tensors left byte-identical and the rest of the norms fingerprinted by ±1-ulp flips consistent with a slightly smaller amplitude (~1.5e‑4).

## Evidence chain (runs 1–19)

1. **Not fine-tuning/training drift**: layer-norms and q/k-norms are bit-stable or ±1 ulp (runs 5, 10, 19); real SGD steps would move them. Cross-layer Δ-correlation is ~0.001 (runs 7, 11) — real training leaves correlated, structured drift.
2. **Not quantization/repack**: A pure requant through any grid returns Δ distributed by |w| (uniform relative); instead Δ is absolute-scale (~1.8e-4 mad) and the Δ=0 fraction is *magnitude-dependent* (12% for down_proj's small weights → 27% for v/up's large weights), the signature of `round(A + continuous noise)` at bf16 (runs 6–8, 12).
3. **The raw law is measurable exactly** on the |A|<3e-4 leakage set (Δ≈N unclipped): Gaussian quantiles 1.54/3.82/6.12e-4 = 0.66σ/1.65σ/2.58σ, kurt+0.18 → **N(0, σ²), σ≈2.3e-4** (runs 12–13). The fat k=|Δ|/ulp tail is entirely the binade mixture (σ/ulp spans 2–60), not multiplicative noise (run 14 fitted c≈0; corr(|Δ|,|A|)=+0.35–0.46 is rounding-clipping mechanics, run 18).
4. **Reconstruction quality**: sim of the law matches obs for P0 (0.97–0.99), mad (1.01–1.03), snz-median (1.0), total energy (+3–7%, a σ-bias), per-binade P0 curves, and reproduce-type differences (run 15).
5. **Residual cosmetic structure (honest note)**: per-row/col amplitude multipliers with CV 0.2–0.43, uncorrelated with A, not repeating across layers after |A|-normalization (run 19) — except o_proj's hot columns do align with A's outlier columns (+0.35). This fattens the SVD spectrum (s0/MP ≈ 3.2–5.3 vs 1.5 for iid) but carries only a few % of energy. It reads like noise whose amplitude was itself drawn spatially-heavy-tailed (or a generator randomization), not a targeted edit — there is no low-rank component, no neuron-set that gate/up/down share, no head that stands out (run 11's Jaccards ≈ uniform, top-8 energy shares ≤1–3%).
6. **run-1's "down_proj anomaly" was a red herring**: it has the smallest mean|W| (0.0123 vs 0.03), so the same absolute σ reads as 2–3× the *relative* diff. mads are equal (1.74–1.96e-4) across all types.

## Functional impact

Per-layer relative activation noise σ_t/rms(W) (run 19): q 0.645%, k 0.630%, v 0.563%, o 0.660%, gate 0.607%, up 0.593%, **down 1.21%** → ~1.3–1.8% per transformer block (o- and down-writes dominate), accumulating to roughly ~5–6% relative perturbation across the 18-layer path (quadrature). So B behaves like "A plus ~1.5%-per-block noise-cream" — mildly but genuinely noisier, not equal, and *not* encoding any obvious message: the deltas carry no shared RNG-stream signature across layers beyond the amplitude variation noted above.

## Constants (for reproduction)

- matrices: σ_t ∈ [1.96, 2.77]e-4, type-means (q,k,v,o,gate,up,down) = (2.51, 2.47, 2.23, 2.18, 2.38, 2.31, 2.23)e-4; iid Gaussian per element; round-to-nearest-even bf16.
- 1-d: 31 tensors exact-zero; 78 touched at ~1–2% of elements by ±1–3 ulp (P0-fitted σ̂ median 1.74e-4, up to 4.2e-4 for a few k_norm/q_norm).
- No value anywhere moves by more than ~6σ; no constant offset, no scale, no permutation.

If the intended deliverable differs from this report (e.g., a recovered checkpoint, an answer file, or a specific question about the pair that I haven't seen — no task file exists anywhere in the work tree), tell me and I'll produce it; analysis-wise nothing further is needed from the remaining ~40 runs.