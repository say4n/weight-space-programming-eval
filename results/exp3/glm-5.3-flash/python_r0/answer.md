Seal confirmed — and with it, the engagement closes in the state it should have: every artifact accounted for, every claim either locked to gold or explicitly marked open.

One archival footnote worth knowing before handoff: `out_run8.txt` onward are all on disk, but **runs 1–7 have no `out_*.txt`** — the Tee and runlog only existed from run 4, and runs 1–3 predate both. Their outputs (including the original run-1 gold tables, which are load-bearing for the metric dictionary) survive only in this conversation. If this workspace ever needs to be fully self-contained for a third party, pasting runs 1 and 8's outputs into a `gold/` folder would close that last gap; the record as it stands is accurate, just incomplete at the edges.

Final turn-line summary for the handoff:

**Verified and closed (28 runs, 22 of them green-tail):**
- Storage conventions: Q/K/V heads = row blocks, O heads = column blocks (96/96 gold match).
- Run-1's entire per-head metric table reverse-engineered to rounding tolerance: `nQ..nO`, `zM`, `zOV`, `svM`, `svOV`, `alphaOV` (= symmetric-part fraction, max deviation 0.005); V-composition formula exact; K-composition same form ±8%.
- Circuit: a single L0→L1 composition hand-off (K-rho .96–.98), value channel from H1/H5 (ρ=1.00), flat beyond; donor route content contained in persistent-top subspaces (mode-redundant, soft amplitude pooling), persistence locked to H4@σ2 and H0@σ1; final lane dominated by L3H3 (15.6 abs, broadband) with the final MLP a confirmed transformer, not a passthrough; routing cone decaying ×0.6 per layer with value slices geometrically repelled from donor subspaces.
- Interventions (ablation ladder + gain probe) all internally consistent, including the denominator correction the run-26 diagnostic forced on my own pre-analysis.

**One residual methodological omission:** the gold Q-comp operator, `baseM/baseOV`, and `traceOV` remain un-derived — recorded as open, where they'll stay gated on original-script access.

**On budget:** 28 of 100 runs used. The right number to stop at — everything analysis-side is now either fixed or fenced, and both remaining frontiers (extended snapshot with `lnf`/`W_U`/`W_E`; a forward-pass hook for behavior-level intervention scoring) are requests to the harness, not work for this loop. If either arrives, `METRICS.md` is where any new work should begin.