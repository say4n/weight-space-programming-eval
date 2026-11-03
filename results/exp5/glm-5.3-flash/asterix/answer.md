Design recap (full reasoning below the code); then the edits block.

**Strategy.** Stay at column 5 forever (all logits for left/right/no-op kept at very negative baseline). All play is vertical. Per row r (1..8) the first layer computes three robust features from raw inputs: `P_r` = "player in row r" (saturating AND of the player cell), `C_r` = "row r is clean" (no enemy within columns 2–8), `M_r` = "clean gold in row r" (gold present and no zone enemy). The second layer builds the saturated logic: `T_r` = player-in-row-r AND row-dirty (saturating AND over `P_r`,`C_r`), `T_any` = OR of the `T_r`, exponential-code comparators `PA_ex`/`PB_ex` (clean gold above/below the player row: Σ2^-r·M_r vs 1.5·2^-r·P_r), `CA_ex`/`CB_ex` (any clean row above/below), and the flee-signal `F = -0.5·T_any`-driven `T_any∧CBias`/`T_any∧¬CBias` products. Logits: up gets +6·PA −6·PB +6·CA −6·CB +20·FU −20·FD; down is mirrored; stay = 8 + 8·¬T_any −20·T_any −12·PA_dis −12·PB_dis (built from complementary AND-pools, no layer-2 cross products needed); left/right/na pinned at −60.

Key margins checked by hand: gold comparator separates "nearest gold just above" (+2·2^-r) from "all gold just below" (max 1·2^-r vs bias 1.5·2^-r); flee ANDs have ≥2.5 gap; entering any clean row is safe even if an entity moves that very tick (enemy would be ≥4 columns away), and a clean row can never hold an enemy at column 5, so standing in a gold row collects it automatically when it sweeps through.

```edits
# ============ W1: per-row first-layer features ============
# P_r : player-in-row-r  (feature units 0..7  <-> row 1..8)
W1[r*60+0 : r*60+60 : 6, r-1] = 3   for r in 1..8   # player channel row r, all 10 cols
b1[r-1] = -1.5
# C_r : row-r clean (no enemy in cols 2..8)  (units 8..15)
W1[r*60+1+12 : r*60+1+60 : 6, 8+r-1] = -3           # enemy channel row r, cols 2..8
b1[8+r-1] = 1
# M_r : clean gold in row r  (units 16..23)
W1[r*60+2 : r*60+60 : 6, 16+r-1] = 3                # gold channel row r, all cols
W1[r*60+1+12 : r*60+1+60 : 6, 16+r-1] = -3          # zone enemies veto
b1[16+r-1] = -1
# (units 24..31 unused: all zero)
# ============ W2: second layer ============
# T_r = AND(P_r, C_r) : units 0..7
W2[r-1, r-1] = 6        # P_r -> T_r
W2[8+r-1, r-1] = -6     # -C_r (dirty) -> T_r
b2[r-1] = -1.5
# T_any = OR_r T_r   : unit 8
W2[0:8, 8] = 2.5
b2[8] = 17.5
# PA_ex = clean gold above : unit 9    PB_ex = below : unit 10
W2[16+r-1, 9]  = 16 * 2^{-(r-1)}       for r in 1..8   # M_r weights 16,8,4,2,1,0.5,0.25,0.125
W2[r-1, 9]     = -12 * 2^{-(r-1)}      # P_r weights -12,-6,-3,-1.5,-0.75,-0.375,-0.188,-0.094
W2[16+r-1, 10] = 16 * 2^{-(8-r)}       # 0.125..16
W2[r-1, 10]    = -12 * 2^{-(8-r)}
b2[9] = -1.5    b2[10] = -1.5
# CA_ex = clean row above : unit 11    CB_ex = below : unit 12
W2[8+r-1, 11]  = 32 * 2^{-(r-1)}       # C_r weights 32,16,8,4,2,1,0.5,0.25
W2[r-1, 11]    = -24 * 2^{-(r-1)}
W2[8+r-1, 12]  = 32 * 2^{-(8-r)}
W2[r-1, 12]    = -24 * 2^{-(8-r)}
b2[11] = -1.5    b2[12] = -1.5
# CBias = player at row >=5 : unit 13
W2[r-1, 13] = 3    for r in 5..8
W2[r-1, 13] = -3   for r in 1..4
b2[13] = 0
# T_any AND CBias  -> FU (unit 14) ;  T_any AND (not CBias) -> FD (unit 15)
W2[8, 14]  = 4    W2[13, 14] = 4    b2[14] = -3.2
W2[8, 15]  = 4    W2[13, 15] = -4   b2[15] = -3.2
# stay-support units (complementary ANDs): unit 16 = notT + PA_complement + PB_complement
W2[8, 16] = 2.5    W2[9, 16] = 2.5    W2[10, 16] = 2.5    b2[16] = -2.6
# unit 17 = notT + PA_complement ; unit 18 = notT + PB_complement
W2[8, 17] = 3    W2[9, 17] = 3    b2[17] = -2.2
W2[8, 18] = 3    W2[10, 18] = 3   b2[18] = -2.2
# ============ W3: logits (columns: 0 na, 1 left, 2 up, 3 right, 4 down, 5 fire) ============
W3[9, 2] = 6     W3[10, 2] = -6     W3[11, 2] = 6     W3[12, 2] = -6
W3[14, 2] = 20   W3[15, 2] = -20
W3[9, 4] = -6    W3[10, 4] = 6      W3[11, 4] = -6    W3[12, 4] = 6
W3[14, 4] = -20  W3[15, 4] = 20
W3[8, 5] = -12   W3[16, 5] = 8      W3[17, 5] = 12    W3[18, 5] = 12
b3 = [-60, -60, 0, -60, 0, 8]
```