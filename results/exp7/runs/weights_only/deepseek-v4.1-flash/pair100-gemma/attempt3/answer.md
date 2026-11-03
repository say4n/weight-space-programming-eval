Run 1 only established that `A` and `B` have identical configs and file sizes; it did **not** compare the actual tensors. The next step should inventory any hidden/task files and compute a tensor-level diff. A useful diagnostic:

```bash
python - <<'PY'
import os, math, hashlib
import torch
from safetensors import safe_open

print("=== inventory ===")
for dp, dns, fns in os.walk("."):
    dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
    for fn in sorted(fns):
        p = os.path.join(dp, fn)
        sz = os.path.getsize(p)
        print(f"FILE {p} {sz}")
        if fn in ("config.json", "model.safetensors"):
            continue
        if sz <= 200_000:
            try:
                with open(p, "r", errors="replace") as f:
                    print(f"--- {p} ---\n{f.read(4000)}\n---")
            except Exception:
                pass

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

paths = {"A": "case/A/model.safetensors", "B": "case/B/model.safetensors"}
print("=== hashes ===")
hs = {m: sha256(p) for m, p in paths.items()}
for m, h in hs.items():
    print(m, h)
print("byte_identical", hs["A"] == hs["B"])

print("=== safetensors ===")
fs = {m: safe_open(p, framework="pt", device="cpu") for m, p in paths.items()}
ks = {m: set(fs[m].keys()) for m in fs}
for m in fs:
    print(m, "metadata", fs[m].metadata())
    print(m, "num_keys", len(ks[m]))
print("A-only", sorted(ks["A"] - ks["B"]))
print("B-only", sorted(ks["B"] - ks["A"]))
common = sorted(ks["A"] & ks["B"])
print("common", len(common))
print("first_keys", common[:20])

def prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n

print("total_params", sum(prod(fs["A"].get_slice(k).get_shape()) for k in common))

rows = []
for k in common:
    a = fs["A"].get_tensor(k)
    b = fs["B"].get_tensor(k)
    rec = {
        "key": k, "shape": tuple(a.shape), "dtype": str(a.dtype),
        "max": None, "rel": None, "srt": None, "rn": None, "cn": None,
        "ident": None, "ndiff": None, "note": "",
    }
    if a.shape != b.shape or a.dtype != b.dtype:
        rec.update(max=float("inf"), rel=float("inf"), ident=False,
                   note=f"B={tuple(b.shape)}/{b.dtype}")
        rows.append(rec)
        continue

    af = a.float()
    bf = b.float()
    d = (af - bf).abs()

    rec["max"] = d.max().item() if d.numel() else 0.0
    rec["rel"] = (d.norm().item() / (bf.norm().item() + 1e-12)) if d.numel() else 0.0
    rec["ident"] = torch.equal(a, b)
    rec["ndiff"] = int((af != bf).sum().item())

    if d.numel():
        sa = torch.sort(af.flatten()).values
        sb = torch.sort(bf.flatten()).values
        rec["srt"] = (sa - sb).abs().max().item()

    if a.ndim == 2:
        rn_a = torch.sort(af.norm(dim=1)).values
        rn_b = torch.sort(bf.norm(dim=1)).values
        cn_a = torch.sort(af.norm(dim=0)).values
        cn_b = torch.sort(bf.norm(dim=0)).values
        rec["rn"] = (rn_a - rn_b).abs().max().item()
        rec["cn"] = (cn_a - cn_b).abs().max().item()

    rows.append(rec)

def fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    return f"{x:.3e}"

rows.sort(key=lambda r: r["max"] if r["max"] is not None else float("inf"), reverse=True)
print("=== top tensor diffs ===")
for r in rows[:40]:
    print(
        f"{r['key']:60s} {str(r['shape']):20s} {r['dtype']:12s} "
        f"max={fmt(r['max'])} rel={fmt(r['rel'])} srt={fmt(r['srt'])} "
        f"rn={fmt(r['rn'])} cn={fmt(r['cn'])} ident={r['ident']} "
        f"ndiff={r['ndiff']} {r['note']}"
    )

print("=== summary ===")
print("identical", sum(1 for r in rows if r["ident"] is True), "/", len(rows))
print("nonidentical", sum(1 for r in rows if r["ident"] is False))
print("max_abs_diff", fmt(rows[0]["max"]) if rows else "-")
PY
```

Interpretation keys:

- `byte_identical True` → the two safetensors files are exactly the same.
- Large `max` but `srt`, `rn`, `cn` near zero → `B` is likely a permutation/reordering of `A`’s weights.
- Differences concentrated in a few tensors → likely a targeted weight edit/fine-tune.
- Uniform small differences everywhere → ordinary fine-tune.