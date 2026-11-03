This completes the fingerprint. L0-L3 are now fully mapped:

**Dead MLP (score = `W_fc row + W_proj col`):**
* `L0 med 0.66, 1078/1536 <1.0 =70.1% dead`, `1123 <1.5`, `1139 <2.0` — massively gated
* `L1 med 3.93, 6/1536 <1.0 =0.39% dead` — exactly 6: `[129,265,636,1040,1237,1310]` with `rn~0.39-0.43 cn~0.19-0.27` vs med 3.93. Rest `>2.45`.
* `L2 med 3.97 0 dead`, `L3 med 4.04 0 dead` — fully dense

**In vs Out outlier consolidation:**
* `L0 O-top [25,72,78,103,114,197,222,254,273,360]` vs `K-top [35,60,162,170,204,223,283,299,342,366]` — overlap **0**
* `L1 O&K = [197,273,360] Jacc 0.176`
* `L3 O&K = [72,78,222,254,360] Jacc 0.333`
* `O&Proj core persistent: 103,360 (+222,249 in L0/L3)` — `L3 Proj-top [103(16.9),249(10.9),226(10.4),124,327]` + `O-top [222(8.9),78(6.6),103(6.5)]`

Put together overall architecture + fingerprint:

* **Arch:** 4-layer Pre-LN, `d=384, d_ff=1536`, **6 x 64-d heads** — proven by `L0 Q/K 128-191 block fro 30.8 vs rest 9.6`, intra-block uniform `med 3.81 min 2.91 max 4.67`; `nH=3/6/12/24` all reduce to 64-d block, `nH=8` splits it `15.9/26.5`.
* **L0:** 70% MLP dead, Q/K single-head `QK per-head [3.0,1.8,182.4,11.4,6.6,2.6]`, `Q-K Gram cos 0.92/0.82`, `QK fro 183 vs rand 50`, `VO fro 39 vs 12`, `MLP fro 191 vs 46`, `W_o stable 5.3, W_fc 6.5`, `ln1_w mean 0.24 32%<0.1`, `ln biases ~0`.
* **L1:** 6 dead MLPs, `Q-K Gram 0.57/0.22`, `K-col outliers 222(9.2),72(8.4),78(7.9)` with `ln*col gain 10.4/6.7/6.2 vs 1.8`, `ln1_b heavy ±1.47 2.6%>0.5`.
* **L2:** 0 dead, most uniform/dense `stable 46-80`, `K outliers persist 222(8.8),78(8.7)`.
* **L3:** 0 dead, `QK head5 suppressed [15.7,14.2,12.7,13.3,14.4,9.5]` but `VO head5 boosted [29,53,35,36,31,59]`, `QK [31,25,21,22,26,11.9]`, `W_proj stable 9.8 spec 26.3 med 0.80`, `W_o stable 17.9 top 9.57/8.76`.
* **Cross-layer:** `cos(W) ~0` independent, `cos(ln) 0.93-0.99` mean-dominated (`Jacc top20 0.02-0.08`), `VO overlap 0.47-0.60`, `MLP inter L0 0.47 vs 0.05-0.13 deeper`.

No task file in `work/` — only `blocks.npz` — so this is the reusable fingerprint. If exp3 is language/contributor ID, the discriminative keys are: `70% L0 dead + single QK head 128-191 + 6 L1 dead ids + K-gain 10→9→4 + outlier core 72/78/222/360/103`.

Tell me what label you want predicted from this and I'll emit the classifier-ready JSON — or I can dump all vectors to `fingerprint.npz` in next run.
