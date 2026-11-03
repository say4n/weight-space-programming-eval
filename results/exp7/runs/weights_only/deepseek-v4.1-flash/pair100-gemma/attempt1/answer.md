```bash
python3 - <<'PY'
import json, struct, mmap, os, numpy as np
from collections import defaultdict

paths = {'A': 'case/A/model.safetensors', 'B': 'case/B/model.safetensors'}

def parse(path):
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hlen = struct.unpack('<Q', mm[:8])[0]
    header = json.loads(mm[8:8+hlen])
    base = 8 + hlen
    return f, mm, header, base

fa, ma, ha, basea = parse(paths['A'])
fb, mb, hb, baseb = parse(paths['B'])
ka = {k for k in ha if k != '__metadata__'}
kb = {k for k in hb if k != '__metadata__'}
print('keys A/B:', len(ka), len(kb), 'same set:', ka == kb)

rows = []
same = 0
for k in sorted(ka):
    ia, ib = ha[k], hb[k]
    sa, ea = ia['data_offsets']
    sb, eb = ib['data_offsets']
    ba = ma[basea+sa:basea+ea]
    bb = mb[baseb+sb:baseb+eb]
    if ba == bb:
        same += 1
        continue
    numel = 1
    for d in ia['shape']:
        numel *= d
    arr_a = np.frombuffer(ma, dtype='<u2', count=numel, offset=basea+sa)
    arr_b = np.frombuffer(mb, dtype='<u2', count=numel, offset=baseb+sb)
    va = (arr_a.astype(np.uint32) << 16).view(np.float32)
    vb = (arr_b.astype(np.uint32) << 16).view(np.float32)
    d = np.abs(va - vb)
    ndiff = int(np.count_nonzero(d))
    rows.append((k, ia['shape'], numel, ndiff, float(d.max()), float(d.mean()),
                 float(np.abs(va).max()), float(np.abs(vb).max())))

print('identical tensors:', same, 'differing tensors:', len(rows))

with open('diff_tensors.tsv', 'w') as out:
    out.write('key\tshape\tnumel\tn_diff\tmax_abs_diff\tmean_abs_diff\tmax_abs_A\tmax_abs_B\n')
    for k, shape, numel, nd, mx, mean, maa, mab in rows:
        out.write(f'{k}\t{shape}\t{numel}\t{nd}\t{mx:.9g}\t{mean:.9g}\t{maa:.9g}\t{mab:.9g}\n')

print('\nTop differing tensors by max_abs_diff:')
for r in sorted(rows, key=lambda x: x[4], reverse=True)[:40]:
    k, shape, numel, nd, mx, mean, maa, mab = r
    print(f'{k:60s} shape={shape} ndiff={nd}/{numel} max={mx:.6g} mean={mean:.6g} maxA={maa:.6g} maxB={mab:.6g}')

# layer-level summary
lay = defaultdict(lambda: [0, 0, 0.0, 0])
other = [0, 0, 0.0, 0]
for r in rows:
    k, shape, numel, nd, mx, mean, maa, mab = r
    parts = k.split('.')
    target = other
    if len(parts) > 2 and parts[0] == 'model' and parts[1] == 'layers':
        target = lay[int(parts[2])]
    target[0] += 1
    target[1] += nd
    target[2] = max(target[2], mx)
    target[3] += numel
print('\nLayer summary: diffs / changed_elems / total_elems / max_abs_diff')
for i in sorted(lay):
    ndiff_t, changed, mx, total = lay[i]
    print(f'layer {i:2d}: {ndiff_t:2d} tensors, {changed}/{total} elems, max={mx:.6g}')
if other[0]:
    ndiff_t, changed, mx, total = other
    print(f'other    : {ndiff_t:2d} tensors, {changed}/{total} elems, max={mx:.6g}')
PY
```