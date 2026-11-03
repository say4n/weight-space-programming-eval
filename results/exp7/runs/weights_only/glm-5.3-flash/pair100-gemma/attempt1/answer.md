Diagnosis: this is a parse-time failure, so nothing in the script executed. Line 41 contains a report fragment (`family median (layers 0-12) of per-head tr(OV):  A=-5.066  B=-5.767`) that ended up as bare Python code — almost certainly the opening `print("` / quote was dropped, or an earlier unescaped `"` inside a string literal prematurely closed it, dumping subsequent text out of the string. The fix is to repair the source around line 41 (and check whether more lines got "spilled" the same way), not to change runtime settings.

Before patching blind, let me look at the actual source:

```bash
ls -la
for f in *.py; do
  echo "===== $f ====="
  nl -ba "$f" | sed -n '1,120p'
done
```

With the surrounding lines visible I'll:
1. Find where the string literal breaks (search backwards from 41 for the last `print(` / `"""` / unbalanced quote).
2. Either re-quote line 41 properly (e.g. `print(f"family median (layers 0-12) of per-head tr(OV): A={...} B={...}")`) or escape/remove the offending embedded quote earlier in the string.
3. Re-run and continue with the planned tr(OV) per-head comparison between A and B.