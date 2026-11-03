## Diagnosis (run 1)

The crash is purely a formatting bug: at line 22 a format spec (e.g. `f"{x:.3f}"`) was applied to a value that is still a `numpy.ndarray` — `ndarray.__format__` doesn't accept format specs (only 0‑d/numbers do). All data loaded fine (the inventory printed completely), so nothing is lost; the fix is to wrap every formatted value in `float(...)`/scalarize first, harden each section with `try/except`, and make the loader fall back across `*.npz`/`*.safetensors`/`*.pt`.

## Baselines & findings (computable already from run‑1 numbers)

Baselines: `fro = 0.02*sqrt(m*n)` → 15.36 for fc/proj, 7.68 for q/k/v/o; ln_w(=1.0 vector) → 19.60; ln_b → 0.00. Ratio = observed/baseline:

| tensor | base | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| W_fc   | 15.36 | 3.01 | 5.15 | 5.14 | 5.06 |
| W_proj | 15.36 | 2.56 | 4.93 | 5.00 | 5.36 |
| W_q    | 7.68 | 4.21 | 4.21 | 5.13 | 4.29 |
| W_k    | 7.68 | 3.96 | 3.81 | 4.92 | 4.19 |
| W_v    | 7.68 | 1.93 | 3.56 | 4.34 | 5.18 |
| W_o    | 7.68 | 2.13 | 3.38 | 4.28 | 5.27 |
| ln1_w  | 19.60 | 0.34 | 0.47 | 0.71 | 0.76 |
| ln2_w  | 19.60 | 0.87 | 1.06 | 1.32 | 1.28 |
| ln1_b  | 0 | n=0.56 | 3.28 | 2.99 | 1.68 |
| ln2_b  | 0 | 1.34 | 2.67 | 3.01 | 2.50 |

1. **Not a stock random init.** A σ=0.02 init would give ratio 1.0 everywhere, ln1_b ≡ 0, ln_w norm exactly 19.6. Instead every matrix is 1.9–5.4×, all biases have moved, and LN gains drifted (ln1 shrunk to 0.34–0.76×, ln2 inflated to ~1.1–1.3×). This is a trained and/or edited checkpoint.
2. **Layers 1–3 are statistically flat slabs** σ_eff = fro/√(mn) ≈ 0.07–0.11 (≈5× init σ), with absmean ≈ 0.94–1.00× the Gaussian expectation 0.798σ — i.e. Gaussian‑like.
3. **Layer 0 is anomalous** — consistent with hand‑planted structure (previous weight‑space programming): all six L0 matrices are heavy‑tailed (absmean only 0.60–0.84× Gaussian expectation, e.g. W_q.0 and W_q.1 have ≈equal fro but absmean 0.040 vs 0.066 → a few huge entries in q.0), q/k enlarged (σ≈0.08) while v/o are deliberately weak (σ≈0.04, 2× init).
4. **LN gains are quasi‑scalar calibration knobs** (ln2_w.3: RMS 1.28, mean 1.278 → nearly uniform 1.28) — cheap global amplifiers/silencers for follow‑up edits.

Run 2 (final, hardened — safe scalar formatting only, per‑section try/except, loader fallback, plus structure probes and context recovery):

```bash
cd /Users/sayan/Projects/weight-space-programming/results/exp3/glm-5.3-flash/english_r1/work && python3 <<'PY'
import glob, math, os, traceback
import numpy as np

def ff(x, nd=3): return f"{float(x):.{nd}f}"
def root(n): return n.rsplit(".", 1)[0] if "." in n else n
def layer(n):
    t = n.rsplit(".", 1)[1] if "." in n else ""
    return int(t) if t.isdigit() else -1

Wdict, msgs, ok = {}, [], False
cand = sorted(glob.glob("*.npz")) + sorted(glob.glob("../*.npz"))
for f in cand:
    try:
        z = np.load(f)
        ks = list(z.files)
        if any(("W_" in k) or ("ln1" in k) or ("ln2" in k) for k in ks):
            Wdict = {k: np.asarray(z[k], np.float64) for k in ks}
            print("[load] using", f, "tensors:", len(ks)); ok = True; break
    except Exception as e:
        msgs.append((f, repr(e)))
if not ok:
    try:
        from safetensors.numpy import load_file
        for f in sorted(glob.glob("*.safetensors")) + sorted(glob.glob("../*.safetensors")):
            Wdict = {k: np.asarray(v, np.float64) for k, v in load_file(f).items()}
            print("[load] using", f); ok = True; break
    except Exception as e:
        msgs.append(("safetensors", repr(e)))
if not ok:
    try:
        import torch
        for f in sorted(glob.glob("*.pt")) + sorted(glob.glob("*.pth")) + sorted(glob.glob("../*.pt")):
            try: obj = torch.load(f, map_location="cpu", weights_only=True)
            except Exception: obj = torch.load(f, map_location="cpu")
            sd = obj.get("model", obj) if isinstance(obj, dict) else obj
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            for k, v in sd.items():
                Wdict[k] = v.detach().cpu().numpy().astype(np.float64)
            print("[load] using", f, "tensors:", len(Wdict)); ok = True; break
    except Exception as e:
        msgs.append(("torch", repr(e)))
if not ok:
    print("[load] FAILED:", msgs, "cwd:", sorted(os.listdir("."))); raise SystemExit(1)

TAB = ["W_fc","W_k","W_o","W_proj","W_q","W_v","ln1_b","ln1_w","ln2_b","ln2_w"]
names = sorted(Wdict, key=lambda n: (TAB.index(root(n)) if root(n) in TAB else 99, layer(n)))
rec = {}

print("\n=== baselines (std=0.02: fro=0.02*sqrt(m*n), absmean=0.0160; ln_w: 1.0s, fro=19.60; ln_b: 0) ===")
print(f"{'name':9s} {'shape':14s} {'fro':>8s} {'base':>7s} {'ratio':>6s} {'sigma':>8s} {'absmean':>8s} {'heav':>6s} {'maxabs':>7s}")
for n in names:
    W = Wdict[n]
    fro = float(np.linalg.norm(W)); absm = float(np.mean(np.abs(W))); mx = float(np.max(np.abs(W)))
    if W.ndim == 2:
        m, k = W.shape
        bf = 0.02 * math.sqrt(m * k); sg = fro / math.sqrt(m * k)
        hv = absm / (0.7979 * sg); n4 = int((np.abs(W) > 4.0*sg).sum())
        n4x = 6.33e-05 * float(W.size)
        rec[n] = dict(shape=W.shape, bf=bf, sg=sg, hv=hv, mx=mx, n4=n4, n4x=n4x)
    else:
        k = W.shape[0]; bf = math.sqrt(k) if root(n).endswith("_w") else 0.0
        sg = float("nan"); hv = float("nan"); n4 = -1; n4x = float("nan")
        rec[n] = dict(shape=W.shape, bf=bf, sg=sg, hv=hv, mx=mx, n4=n4, n4x=n4x)
    ratio = fro / bf if bf > 0 else float("nan")
    print(f"{n:9s} {str(W.shape):14s} {fro:8.2f} {bf:7.2f} {ratio:6.2f} {sg:8.4f} {absm:8.5f} {hv:6.2f} {mx:7.3f}")

print("\n=== structure probes (2D): singular spectrum, outlier counts ===")
try:
    print(f"{'name':9s} {'s1':>8s} {'s2':>8s} {'E1':>6s} {'>4sg':>5s} {'exp':>6s} {'rowmax/med':>10s} {'colmax/med':>10s}")
    for n in names:
        W = Wdict[n]
        if W.ndim != 2: continue
        s = np.linalg.svd(W, compute_uv=False)
        E1 = float(s[0]*s[0] / float((s*s).sum()))
        rn = np.linalg.norm(W, axis=1); cn = np.linalg.norm(W, axis=0)
        rr = float(rn.max()) / max(float(np.median(rn)), 1e-9)
        cc = float(cn.max()) / max(float(np.median(cn)), 1e-9)
        print(f"{n:9s} {float(s[0]):8.2f} {float(s[1]):8.2f} {E1:6.3f} {rec[n]['n4']:5d} {rec[n]['n4x']:6.1f} {rr:10.2f} {cc:10.2f}")
except Exception:
    traceback.print_exc()

print("\n=== LN vectors (init: w=1.0 norm 19.60, b=0) ===")
for n in names:
    W = Wdict[n]
    if W.ndim != 1: continue
    print(f"{n:9s} norm={ff(np.linalg.norm(W))} mean={ff(np.mean(W),4)} absmean={ff(np.mean(np.abs(W)),4)} min={ff(np.min(W),4)} max={ff(np.max(W),4)}")

print("\n=== layer-0 top row norms (localize planted structure) ===")
try:
    for n in ("W_q.0","W_k.0","W_v.0","W_o.0","W_fc.0","W_proj.0"):
        if n not in Wdict: continue
        A = Wdict[n]
        for ax in (1, 0):
            rn = np.linalg.norm(A, axis=ax)
            top = np.argsort(rn)[::-1][:6]
            print(f"{n} ax{ax} top: " + ", ".join(f"{int(i)}:{ff(rn[i],2)}" for i in top))
except Exception:
    traceback.print_exc()

print("\n=== verdict ===")
r2d = [rec[n] for n in names if Wdict[n].ndim == 2]
mxr = max(r["fro"]/r["bf"] for r in r2d if r["bf"] > 0); mnr = min(r["fro"]/r["bf"] for r in r2d if r["bf"] > 0)
lb = max((float(np.linalg.norm(Wdict[n])) for n in names if root(n)=="ln1_b"), default=0.0)
heavy = [n for n in names if Wdict[n].ndim==2 and rec[n]["hv"] < 0.85]
if mxr > 1.5 or lb > 0.05:
    print(f"NOT random-init: fro ratios {ff(mnr,2)}-{ff(mxr,2)}x baseline; ln1_b up to {ff(lb,2)}; heavy-tailed (edited) tensors: {heavy}")
    print("layers 1-3 look like homogeneous ~N(0,0.10^2) slabs; layer 0 carries sparse/edited structure")
else:
    print("consistent with stock random init")

print("\n=== cwd / context recovery ===")
try:
    for f in sorted(os.listdir(".")):
        p = os.path.join(".", f)
        if os.path.isfile(p): print(f"  {f}  {os.path.getsize(p)}B")
    for pat in ("*.md","*.txt","*.json","*.yaml","*.yml","*.toml"):
        for f in sorted(glob.glob(pat)):
            if 0 < os.path.getsize(f) <= 4096:
                print(f"\n--- {f} ---"); print(open(f, encoding="utf-8", errors="replace").read()[:4096])
    for f in sorted(glob.glob("*.py")):
        if os.path.getsize(f) <= 8192:
            print(f"\n--- {f} (first 2500c) ---"); print(open(f, encoding="utf-8", errors="replace").read()[:2500])
except Exception:
    traceback.print_exc()
PY
```

Expected of run 2: (a) the baseline table with ratios confirming the numbers above; (b) singular-spectrum/row-outlier probes that will decisively separate "hand-planted attention circuit in layer 0" from an ordinarily trained checkpoint (planted ⇒ s1 ≫ Gaussian expectation, top1_energy ≫ ~0.003, row max/median ≫ ~2, entries >4σ far above the ~9 expected); and (c) dumping of any task/README/scripts in the work dir so the follow-up edit (likely: route computation through the planted L0 head and use the quasi-scalar ln2_w / ln1_w gains as amplifiers or silencers) can be specified with the actual mission in hand.