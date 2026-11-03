Final run confirms: the top-4 extension closes the loop question for good — **0 of 192** direction-level correlations exceed 0.35 (top residual hits ~0.18, i.e. noise floor). The L0H5→L3H1 OV continuity (cos 0.78) stands as the *only* long-range attention structure, and it is a write-path, not a match path. Hashes are identical (`04ec0b300d207b26`): **`blocks.npz` is provably pristine** relative to the extracted checkpoint.

Closing the loop at **11 of 100 runs**. The official record:

**Model.** 4 pre-LN blocks, d=384, 6×64 heads, MLP 4×1536; checkpoint holds blocks only — no embeddings, unembed, or final LN; runtime readout external.

**Structure found (live):**
- **Feature A** — L0's trio H1/H2/H4 are one matcher written three times (cos 0.78–0.96 aligned); support {12, 20, 21, 29, 39, 51, 58, 63}.
- **Feature B** — L0H5 (uniform attention, rank-1 OV, s₁=24.8) writes an aggregate register whose direction survives intact into L3H1's big OV (0.78): a detected-then-reinjected feature.
- **LN-enforced channel policy** — the one exact, non-correlational fact: 21 attention-only (ln₂ ≈ 0 in all four layers, ln₁ 0.8–1.1 deep), 383 MLP-only (attention-blind throughout), 184 the boundary (top write of both economies, L0's top read), 39/58/99 attention conduits (39 = single most attention-written channel), 5/53/89/133/219 MLP conduits, {12,20,29,63,71,153} MLP-native → attention-adopted by L3. L0's two sublayers read disjoint bands.
- L1's six heads: one shared core direction among same-type matchers (~1 dim vs 0.14 random baseline). W_o fully mixing.

**Dead (all |ρ| ≤ 0.2, profile-level):** LN↔write alignment, LN hand-off, FR~FW conveyor, bus-vector coherence (0.00–0.07 — addresses, not features), matchers-on-A/B (top-1 and top-4), MLP-read/attention-write coupling (reliably slightly negative).

**Artifacts in `work/`:** `blocks_original.npz` (restore point, hash-verified) · `blocks.npz` (pristine) · `blocks_E1_unclamp21 / blocks_E2_noB / blocks_E3_noreinject.npz` (principled diagnostics — expected *score-negative* as blind behavior edits, retain for closed-loop use only) · `EDIT_MANIFEST.txt` · `REPORT.md` with the full map and the §7 playbook: (1) A/B + shared QK core projected onto token embeddings, (2) L₃ `W_proj` writes vs unembed rows, (3) one-at-a-time E-tests with restore.

**Why stop:** the sandbox is sealed (no parent dirs, no evaluator, no feedback channel in nine harvest-checked runs), and every remaining probe returns noise. Continuing burns runs and adds corruption risk while the map is already at the resolution the weights permit. If the harness or runtime later exposes an eval path or the embedding stack, the playbook above is the ready-made next program.