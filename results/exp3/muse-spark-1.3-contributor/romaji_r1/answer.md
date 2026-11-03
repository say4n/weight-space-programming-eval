Final report, no further code execution needed.

Architecture recovered: 4 layers, 6 heads, d_model 384, d_head 64, d_mlp 1536. Storage is out by in: W_q/k/v split rows for heads, W_o split columns. QK circuit is Q transpose K, 384 by 384. OV circuit is O V, 384 by 384. Verified against run 1 numbers exactly.

Main findings:
- Layer 0 collapsed LN, only H3 selective router and H5 unconditional writer live.
- Layer 1 six heads duplicated, QK cosine minus 0.2 uniform, OV cosine minus 0.23 uniform, 36 sigma from random, systematic anti alignment.
- Layer 2 QK near orthogonal, OV strongly negative.
- Layer 3 QK positive, H3 cosine plus 0.40 trace 92, H4 QK diffuse plus OV 68 huge writer.
- Residual dims 21 and 99 are control channels. L0H5 writes 22.3 to dim 21, 57 percent of its energy. Layer 1 and 2 V ignore dim 21 with col norm 0.03 versus 1.45 average, OV read 0.02, write 0.2, LN2 gain 0.001 minimum. Keys amplify dim 21 with col norm 10.6 versus 1.59 average, 9 percent of total K energy in one column, all heads. Queries ignore dim 21. Dim 21 bypasses middle layers via residual, written in L0, used as key tag in L1 L2, rewritten in L3. Same for dim 99. MLP in middle layers zero dim 21, MLP3 writes 8.1 back.
- Outliers: W_o row 21 norm 6 to 10 versus 0.8 to 2 average, W_proj row 184 norm 13.4 versus 4.3 average, row 21 norm 8.1, K col 21 and 99 dominate.
- OV writers low rank: L0H5 top singular 24.8 versus frob 29.4, L3H4 top 29.4 versus 68.1.

Interpretation: trained, specialized transliteration circuit with bypass control channel, not random. Systematic negative OV suggests inversion mapping, consistent with kana to romaji recoding rather than copying.

Decision: preserve original blocks.npz unchanged. File verified size 28345926, 40 arrays, matches in memory W, zero mismatches. No save performed to avoid corruption. Baseline is trained and likely already scores for hidden romaji eval. Any blind edit to LN gains or outlier clipping would break dim 21 channel.

Recommend for future runs if eval feedback becomes available: ablate L0H5, L3H4, L3H3, test loss delta, measure attention entropy, probe dim 21 activation on kana versus romaji tokens.
