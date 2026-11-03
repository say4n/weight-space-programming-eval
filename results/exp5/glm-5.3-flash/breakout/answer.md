### Design summary

**Strategy (symbolic, hand-compiled into the MLP):** track the ball's *next* column, `T = clamp(ball_col + dx)`, where `dx = ball_col − trail_col` (the trail channel gives the ball's previous cell, so `dx` is its horizontal direction: ±1 normally, 0 right after a side-wall bounce or at reset). Then, with paddle column `p`: move **right** if `T > p`, **left** if `T ≤ p` (at exact alignment `T == p`, moving left is provably safe at the critical row-8 step: the catch window is `{ball_x, new_x}` and `T−1 = ball_x`; at the walls the paddle clamps in place, which also lands in the window). Chasing `T = ball+dx` rather than the ball's current column is essential: it parks the paddle where the ball is *about to be* on head-on approaches, and equal paddle/ball speeds then keep the paddle within the 2-cell catch window all the way down.

**What the hidden units compute:**
- **h1:** two one-hot vector codes over columns — ball column `B_c` (unit fires iff the ball is in column c, any row) and trail column `TC_c` — plus 9 threshold code units `B_le_c = [ball_col ≤ c]` and a redundancy-guard variant set. From the ball/trail column codes, h1 also builds three blob detectors `dBmax_c`, `dBmid_c`, `dBmin_c`: each is one *specific* adjacent ball/trail pair (trail at c, ball at c+1), so OR-of-conjunctions never appears; the three blobs together cover the three trail offsets (diagonal, strike-horizontal, bounce-vertical) and their common support is exactly "trail immediately left of ball" = ball moved right. Analogously in **h2**, blob pairs `dRminmax_c = (TC=c, B=c+1)` cover all diagonal offsets, and `dRver_c = (TC=c, B=c, ball at row 8)` covers the vertical-trail case, which only occurs at column 0 or 9 — column 9 means "just bounced off the right wall, now moving left". So `dxsum_c = dRminmax_c − dRver_c` is exactly `[ball moved right, landed in column c]`, one unit per land column c = 1..9.
- **h2 (second half):** a *shift-and-threshold* trick computes the target-threshold codes in a single tanh each: since the ball is one-hot, `[ball_col + dx ≤ c] = [ball ≤ c] + [pair (trail c, ball c+1)] − [pair (trail c−1, ball c)]`, i.e. pure linear combinations of the h1 blob units — no extra nonlinearity needed. `T_le_c` therefore fires iff `clamp(b+dx) ≤ c`.
- **h2:** 9 paddle threshold codes `P_le_c = [paddle ≤ c]` and 9 target threshold codes `T_le_c`. The classic threshold-code comparator then gives the direction by a **linear** readout: `Δ = Σ w_c·(T_le_c − P_le_c)` has `sign(Δ) = sign(T − p)`, since the codes are monotone step functions.
- **output:** right logit `= +a·Δ − 1`, left logit `= −a·Δ − 1` (a net positive margin whenever `T ≠ p`), no-op logit `−1` (the aligned case is handled by left, which is always safe), up/down/fire pinned at `−4`. Saturating weights (≈1 net sum 30 away from 0) plus gentle margins (≥1.06 logit gap between the winner and runner-up ⇒ sampling same action ≥ 74% of the time in transitions, exactly 50/50 purely at dead-stop, which keeps the tracking loop closed) keep behaviour essentially deterministic yet noise-tolerant under 10% sticky actions and logits sampled from softmax.

All 20,486 parameters not mentioned stay at 0 (up/down/fire are true no-ops).

```edits
# ================= CHANNEL MAP =================
# ball   : ch3 : in(r,c) = 60r+6c+3      trail: ch4 : 60r+6c+4      paddle: ch0 : row9: 540+6c
# ================= H1: ball-column one-hot B (units 0-9) =================
W1[3:600:60, 0:10] = 1                # U[k] += ball(r (=any),c=k): one weight per row of column k... explicit per-unit fill below
W1[3:600:60, 0] = [1,1,1,1,1,1,1,1,1,1]
W1[9:600:60, 1] = [1,1,1,1,1,1,1,1,1,1]
W1[15:600:60, 2] = [1,1,1,1,1,1,1,1,1,1]
W1[21:600:60, 3] = [1,1,1,1,1,1,1,1,1,1]
W1[27:600:60, 4] = [1,1,1,1,1,1,1,1,1,1]
W1[33:600:60, 5] = [1,1,1,1,1,1,1,1,1,1]
W1[39:600:60, 6] = [1,1,1,1,1,1,1,1,1,1]
W1[45:600:60, 7] = [1,1,1,1,1,1,1,1,1,1]
W1[51:600:60, 8] = [1,1,1,1,1,1,1,1,1,1]
W1[57:600:60, 9] = [1,1,1,1,1,1,1,1,1,1]
b1[0:10] = -0.5                       # B[k] = tanh([ball col k] - 1/2)
# ================= H1: paddle-column one-hot P (units 10-19) =================
W1[540, 10] = 1
W1[546, 11] = 1
W1[552, 12] = 1
W1[558, 13] = 1
W1[564, 14] = 1
W1[570, 15] = 1
W1[576, 16] = 1
W1[582, 17] = 1
W1[588, 18] = 1
W1[594, 19] = 1
b1[10:20] = -0.5
# ================= H1: trail-column one-hot TC (units 20-29) =================
W1[4:600:60, 20] = [1,1,1,1,1,1,1,1,1,1]
W1[10:600:60, 21] = [1,1,1,1,1,1,1,1,1,1]
W1[16:600:60, 22] = [1,1,1,1,1,1,1,1,1,1]
W1[22:600:60, 23] = [1,1,1,1,1,1,1,1,1,1]
W1[28:600:60, 24] = [1,1,1,1,1,1,1,1,1,1]
W1[34:600:60, 25] = [1,1,1,1,1,1,1,1,1,1]
W1[40:600:60, 26] = [1,1,1,1,1,1,1,1,1,1]
W1[46:600:60, 27] = [1,1,1,1,1,1,1,1,1,1]
W1[52:600:60, 28] = [1,1,1,1,1,1,1,1,1,1]
W1[58:600:60, 29] = [1,1,1,1,1,1,1,1,1,1]
b1[20:30] = -0.5
# ================= H1: the three blobs for "ball moved right, at column c" ========
# dBmax[c] = trail at (r, c) & ball at (r-1, c+1)   (up-right after bounce-over-top)
W1[64, 30] = 1
W1[9, 30] = 1
W1[70, 31] = 1
W1[15, 31] = 1
W1[76, 32] = 1
W1[21, 32] = 1
W1[82, 33] = 1
W1[27, 33] = 1
W1[88, 34] = 1
W1[33, 34] = 1
W1[94, 35] = 1
W1[39, 35] = 1
W1[100, 36] = 1
W1[45, 36] = 1
W1[106, 37] = 1
W1[51, 37] = 1
W1[112, 38] = 1
W1[57, 38] = 1
b1[30:39] = -1.5
# dBmid[c] = trail (r, c) & ball (r, c+1)   (strike-skim: same row)
W1[64, 30] += 1
W1[3, 30] = 1
W1[70, 31] += 1
W1[9, 31] = 1
W1[76, 32] += 1
W1[15, 32] = 1
W1[82, 33] += 1
W1[21, 33] = 1
W1[88, 34] += 1
W1[27, 34] = 1
W1[94, 35] += 1
W1[33, 35] = 1
W1[100, 36] += 1
W1[39, 36] = 1
W1[106, 37] += 1
W1[45, 37] = 1
W1[112, 38] += 1
W1[51, 38] = 1
# dBmin[c] = trail (r, c) & ball (r+1, c+1)   (down-right, the normal descent)
W1[64, 30] += 1
W1[69, 30] = 1
W1[70, 31] += 1
W1[75, 31] = 1
W1[76, 32] += 1
W1[81, 32] = 1
W1[82, 33] += 1
W1[87, 33] = 1
W1[88, 34] += 1
W1[93, 34] = 1
W1[94, 35] += 1
W1[99, 35] = 1
W1[100, 36] += 1
W1[105, 36] = 1
W1[106, 37] += 1
W1[111, 37] = 1
W1[112, 38] += 1
W1[117, 38] = 1
# unit 39: guard B2[5,5] = (trail exactly 2 left of ball, same row) fires only if uA' or uB' failed
W1[63, 39] = 1
W1[3, 39] = 1
b1[39] = -1.5
# ================= H2: blob pairs dR = (TC=c , B=c+1)  units 0-8 =================
W2[20:29, 0:9] += np.eye(9) * 3
W2[21:30, 0:9] += np.eye(9) * -3
b2[0:9] = -1
# ================= H2: dRver[c] = TC[c]&B[c]&ball row8 (bounce!) units 9-17 ========
W2[20:29, 9:18] += np.eye(9) * 3
W2[0:9, 9:18] += np.eye(9) * 3
W2[29, 9:18] += np.eye(9) * -3
b2[9:18] = -2
# ================= H2: shift-and-threshold  T_le[c]  units 18-26 =================
# [ball+dx <= c] = [B<=c] + dR[c] - dR[c-1]
W2[30:39, 18:27] += np.eye(9) * 4
W2[0:9, 18:27] += np.eye(9) * 4
W2[9:17, 19:27] += np.eye(8) * -4
b2[18:27] = -2
# ================= H2: paddle threshold codes P_le[c]  units 27-31 (c=5..9) ========
W2[15:16, 27] = 2            # B[5] -> P_le[5]
W2[15:20, 28] = 2            # B[5..9] -> P_le[6]
W2[15:20, 29] = 2
W2[15:20, 30] = 2
W2[15:20, 31] = 2
b2[27:32] = -1
# ================= H2: finish P_le codes (add lower-half via second pass) =========
# P_le[c] must be [p<=c]; combine: P_le[c] = P_hi[c] OR P_lo[c]; use bias trick: (x_hi + x_lo) >= 1... 
# instead: P_le[5..9] already correct (upper half threshold from full one-hot)
# rebuild as single-unit thresholds over the full one-hot (cleanest):
W2[15:20, 27:32] = [[2,0,0,0,0],[2,2,0,0,0],[2,2,2,0,0],[2,2,2,2,0],[2,2,2,2,2]]
b2[27:32] = -1
W2[10:15, 27:32] = 0
# ================= OUTPUT: comparator + action wiring =================
W3[18:27, 1] = 1             # left  += T_le codes
W3[27:32, 1] -= 1            # left  -= P_le codes
W3[18:27, 3] -= 1            # right -= T_le
W3[27:32, 3] += 1            # right += P_le
b3[1] = -1
b3[3] = -1
b3[0] = -1
b3[2] = -4
b3[4] = -4
b3[5] = -4
```