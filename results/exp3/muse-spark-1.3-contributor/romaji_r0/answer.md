Final contribution ready in `work/`:

* `blocks.npz` — left **unchanged** (28,345,926 bytes). Analysis shows large spikes are intentional discrete logic (e.g. L1 hidden 1139 generic `[165,52,3]` → residual `184` with `2.66`, `W_k` cols `21:10.6,99:10.1` for addressing, `W_o/proj` rows `21:6-10,184:8-13` for broadcast). Blind clipping would break the `generic→control` funnel.
* `interpretation.md` (7041 bytes) — full mechanistic interpretation:

**7-dim control subspace** `5,21,39,58,99,184,383` dominates all layers:
- `W_k` reads it, `W_o/W_proj` writes it, `W_v`/`ln2_w` ignores it (`W_v.1 col21:0.038` vs mean 1.45, `ln2_w.1[21]=0.0018` vs mean 1.13) — two-stream design.
- `MQK=Wq.T@Wk` top `k_in 21:77.3,99:69.7` vs next 8.7; `MOV=Wo@Wv` top `out 184:7.07,39:6.42` from generic `in 361,29…`
- `L0` uniform attn (scores std 0.05) + exploding MLP (`resid 26→proj 245`) to seed control; `L1/L2` winner-take-all (std 5-11, col-sums 0.03-3.42, e.g. any pair with `21` collapses to `21` except `21-vs-99` identity; `39` never wins in L1); `L3` soft (std 1.6) + huge broadcast rows.
- One-hot probe: `in21→out184:324,21:126`, `in39→39:36,184:35`, all inputs pulled to `184/21/39/99` attractor; dense random `in_norm 0.3-10→out_norm 150-280` top always `184` — LN makes scale irrelevant, direction is logic.

Hypothesis for romaji: control dims = vowel/state (`a,i,u,e,o` + `N`/geminate), `L0` averages bigram context (dakuten/yoon/sokuon), `L1/L2` sharply selects kana prompt, `L3` votes romaji logits. `21` sink = BOS/`a`, `39` weak key = rare kana/bug to check.

No weight edit made — preserved programmed solution; see `interpretation.md` for repro (numpy SVD/row-col norms, multi-head forward with pre-LN + GELU-tanh).
