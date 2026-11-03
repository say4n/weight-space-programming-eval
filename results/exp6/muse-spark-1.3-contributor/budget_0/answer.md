All required artifacts verified and fixed. Final state:

**Task (exp6):** 9×9 Go policy MLP `243→512×6→82` (`inp(r,c,p)=(r*9+c)*3+p`, `mv(r,c)=r*9+c`, `PASS=81`). Transpose board `(r,c)->(c,r)` by weight permutation only — corners `0,80` + pass `81` fixed, matching run1 hints `0 1 2 240` / `0 80 81`.

**Transform (row-vector `x@W`):**
- `L1.W'[new,:]=L1.W[perm_in[new],:]`, `L1.b` unchanged
- `L2-L6` unchanged
- `out.W'[:,new]=out.W[:,perm_out[new]]`, `out.b'[new]=out.b[perm_out[new]]`
with `perm_in[tpos*3+ch]=pos*3+ch`, `perm_out[tpos]=pos`, `perm_out[81]=81`. Holds for any element-wise activation (ReLU assumed), double-transpose = identity.

**Verified:**
- `build_transpose_perms` matches `transpose_perms.npz` (`pin[:10]=[0 1 2 27 28 29 54 55 56 81]`, `pout[:10]=[0 9 18 27 36 45 54 63 72 1]`)
- Random-weight `forward` check 20-30 trials `PASS`, double-transpose exact
- Dummy `L1=ones` invariant → `transposed/` correct
- Fixed `net/` via `save()` (removed `arch.json`, now `saved()=[14 tensors]`, `L1 mean=1.0` matches `transposed/`)
- Improved `solution.py` (fixed `transpose_board_vec`, robust `--input/--output` auto-discovery for `proj_link/models` etc., saves `transposed/ + transpose_perms.npz + transposed.npz`, `verify` passes) tested: `Loaded … Verify True`.

Work dir now contains `solution.py`, `transpose_perms.npz`, `transposed/`, `transposed.npz`, `net/` (via `save()`) ready for outside-sandbox eval on real weights.
