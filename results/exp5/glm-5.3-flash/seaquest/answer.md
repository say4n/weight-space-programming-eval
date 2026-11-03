## Design

**Strategy (a "bottom camper"):** The sub always holds column 5 (the start column — the policy never presses left/right) and lives on row 8 (the bottom). Fighting is done facing left forever (initial orientation), so every enemy crossing row 8 from the left enters a fixed *kill zone* (row 8, cols 0–4) where the policy fires; enemies crossing from the right enter a fixed *danger zone* (cols 6–9) and left-adjacent threats (cols 1–4) trigger a vertical dodge. Because every fish, sub and diver spawned on our row sweeps the entire row, they inevitably cross column 5: enemies get shot (facing left), divers get picked up **passively** by contact — no aiming at divers is needed. When the oxygen gauge drops below 3 cells (oxygen < 60) and at least one diver is aboard (diver-bar cell at (9,8) lit), the policy climbs straight up and surfaces (surfacing restores oxygen automatically). It then dives again ("down" is the safe default and clamps at row 8). This yields the sustainable loop: camp → shoot fish → collect divers → surface before oxygen runs out.

**What the hidden units compute (h1, threshold detectors via tanh):**
- Row-8 kill zone / danger zones: "any danger (ch1) in row 8, cols 0–4 (FIRE)", "cols 1–4 (dodge-left)", "cols 6–9 (dodge-right)".
- Player-row detectors: P8 (at row 8), P17, P1 (at rows 0/1) — used to gate dodging (dodging up must never be possible near the surface without divers) and to force a dive at row 0.
- Oxygen: OXYOK = all three of row-9/gauge cols 0,1,2 lit (pure oxygen cells, never polluted by the diver bar). "Low oxygen" = its negation.
- DIV1: gauge cell (9,8) lit (≥1 diver, valid whenever oxygen is low; conjoined with low-oxygen in layer 2).

**h2 units** combine these into the four behavioral conditions: CLIMB = low-O₂ ∧ diver≥1; DODGE = (DL ∨ DR) ∧ not(P1 ∨ P0); FIRE = killzone (only when not dodging); DIVE = P8-negation fallback, plus a hard P0→dive override. **W3** injects each condition into its action's logit (up: climb+dodge; down: dive-at-top + default; fire: killzone). Priorities are enforced by inhibitory contributions between competing conditions.

```edits
# ---------- h1 feature extractors (all thresholds tuned so tanh fires cleanly 0/1) ----------
# Player row detectors: channel 0 cells of rows 0,1,8 (both sub cells share the row)
W1[0:10:6... ]  = [0]
W1[0:60*1:6, 0] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
b1[0] = -0.9                                   # P17? no, unit0 = player at row 0
W1[60+0:120:6, 1] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
b1[1] = -0.3                                   # unit1 = player at row 1
W1[480+0:540:6, 2] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
b1[2] = -0.9                                   # unit2 = player at row 8
# Danger (ch1) counts in row 8 (row offset 480, cell index 480+6c+1)
# unit 3: kill zone cols 0..4 (fire targets)
W1[481, 3] = 1
W1[487, 3] = 1
W1[493, 3] = 1
W1[499, 3] = 1
W1[505, 3] = 1
b1[3] = -0.4
# unit 4: dodge-left zone cols 1..4
W1[487, 4] = 1
W1[493, 4] = 1
W1[499, 4] = 1
W1[505, 4] = 1
b1[4] = -0.4
# unit 5: dodge-right zone cols 6..9
W1[517, 5] = 1
W1[523, 5] = 1
W1[529, 5] = 1
W1[535, 5] = 1
b1[5] = -0.4
# unit 8: killzone-minus-adjacent, cols 0..2 (safe aim: neighbors can't kill us)
W1[481, 8] = 1
W1[487, 8] = 1
W1[493, 8] = 1
b1[8] = -0.4
# Oxygen pure cells row 9 cols 0..2 (ch5 -> index 540+6c+5)
W1[545, 9] = 1
W1[551, 9] = 1
W1[557, 9] = 1
b1[9] = -1.9                                   # OXYOK = all three lit
# Diver count >=1 : gauge (9,8) ch5
W1[593, 10] = 1
b1[10] = -0.91                                 # DIV1 (active unless oxygen >=180; combined later)
# ---------- h2: behavioral units ----------
# unit 0 CLIMB = OXYlow AND DIV1  (W2 col0 over h1 units 9=[OXYOK],10=[DIV1])
W2[9, 0] = 2.0
W2[10, 0] = 2.0
b2[0] = -2.6                                   # fires iff neither OXYOK nor ~DIV1
# unit 1 DODGE = (DL or DR) and not(P17 or P1)  [units 3? no: 4,5 positive; 0,1 negative]
W2[4, 1] = 2.0
W2[5, 1] = 2.0
W2[0, 1] = -3.0
W2[1, 1] = -3.0
b2[1] = -0.4
# unit 2 FIRE (killzone cols 0..4) minus adjacent-threat suppression
W2[3, 2] = 2.0
W2[4, 2] = -2.0
W2[5, 2] = -2.0
b2[2] = 0.4
# unit 3 FIRE2 (safe aim cols 0..2), covers cooldown gaps
W2[8, 3] = 2.0
W2[4, 3] = -1.0
W2[5, 3] = -1.0
b2[3] = 0.4
# unit 4 DIVE = not(P8) (descend toward row 8 whenever camp mode)
W2[2, 4] = -2.0
b2[4] = 1.2
# unit 5 TOPDIVE = at row 0 -> force down (overrides everything except climb)
W2[0, 5] = 2.0
b2[5] = 0.5
# ---------- output logits ----------
# action 0 noop: small negative bias (never needed)
b3[0] = -0.5
# action 2 up: climb over dodge
W3[0, 2] = 4.0
W3[1, 2] = 3.2
# action 5 fire: kill zone (both units)
W3[2, 5] = 2.6
W3[3, 5] = 1.6
# action 4 down: descend + hard override at top row
W3[4, 4] = 2.8
W3[5, 4] = 4.0
b3[4] = 0.6
# slight hygiene: keep left/right/never-press actions unattractive; suppress left (stay facing left)
b3[1] = -0.5
b3[3] = -0.5
```

**Why this survives the rules of the game:** the only paths to row 0 are climb-mode (requires ≥1 diver, so surfacing always pays) and the sticky repeat of a dodge at row 2 (probability 0.1², negligible). Firing is suppressed whenever any threat is adjacent, so bullets only fly at enemies ≥1 column away, and the dive default re-establishes the camp after every dodge. Enemy bullets from either side get a ≥2-column early warning at 1 cell/step, so the usual case allows several dodge chances; the rare instant-loss cases are exactly the ones a hand-coded policy can tolerate while still farming kills and divers every dive cycle.