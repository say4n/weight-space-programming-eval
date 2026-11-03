Run 1 establishes that the two checkpoints share a byte-identical config and the same inventory: both are Gemma3-270M (`gemma3_text`, 18 layers, hidden 640, 4 heads / 1 KV head, head_dim 256, vocab 262144, tied embeddings, float32). The open question is whether the **weights** are the same. Run 2 locates both `model.safetensors` files, parses the safetensors headers directly (no torch/safetensors dependency), and hashes every tensor so any differing tensors are named explicitly.

```bash
python3 - <<'PY'
import os, sys, json, struct, hashlib

PRUNE = {'.git','node_modules','__pycache__','.venv','venv','.mypy_cache','.Trash',
         'Library','.cache','site-packages','dist-packages','.tox','.idea','.vscode'}

def find_model_dirs(root, max_depth=8, max_dirs=200000):
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    out = []
    count = 0
    for dp, dns, fns in os.walk(root):
        count += 1
        if count > max_dirs:
            break
        keep = []
        for d in dns:
            if d in PRUNE:
                continue
            pd = os.path.join(dp, d)
            try:
                if os.path.islink(pd) and os.path.isdir(pd) and \
                   os.path.isfile(os.path.join(pd, 'model.safetensors')):
                    out.append(os.path.abspath(pd))
                    continue
            except OSError:
                pass
            keep.append(d)
        dns[:] = keep
        depth = dp.rstrip(os.sep).count(os.sep) - base
        if depth > max_depth:
            dns[:] = []
            continue
        if 'model.safetensors' in fns:
            out.append(os.path.abspath(dp))
    return out

def load_cfg(m):
    try:
        with open(os.path.join(m, 'config.json')) as f:
            return json.load(f)
    except Exception:
        return {}

start = os.getcwd()
print('cwd:', start)

hints = {}
for k, v in os.environ.items():
    if any(s in k.upper() for s in ('MODEL','TASK','PAIR','GEMMA')):
        hints[k] = v[:500]
print('env hints:', hints)

print('\n[work dir]', start)
try:
    for e in sorted(os.listdir(start)):
        p = os.path.join(start, e)
        try:
            sz = '' if os.path.isdir(p) else os.path.getsize(p)
        except OSError:
            sz = '?'
        print(' ', e + ('/' if os.path.isdir(p) else ''), sz)
except OSError as ex:
    print(' error:', ex)

parent = os.path.dirname(start)
print('\n[parent dir]', parent)
try:
    for e in sorted(os.listdir(parent)):
        p = os.path.join(parent, e)
        try:
            sz = '' if os.path.isdir(p) else os.path.getsize(p)
        except OSError:
            sz = '?'
        print(' ', e + ('/' if os.path.isdir(p) else ''), sz)
except OSError as ex:
    print(' error:', ex)

print('\n[small text/py files in work+parent]')
for base in (start, parent):
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        continue
    for e in entries:
        p = os.path.join(base, e)
        ext = os.path.splitext(e)[1].lower()
        try:
            if os.path.isfile(p) and ext in ('.py','.sh','.json','.txt','.md','.yaml','.yml') \
               and os.path.getsize(p) < 200_000:
                print('\n--- %s ---' % p)
                print(open(p, errors='replace').read()[:5000])
        except OSError:
            pass

proj = start
d = start
for _ in range(12):
    nd = os.path.dirname(d)
    if nd == d:
        break
    d = nd
    if os.path.isdir(os.path.join(d, 'results')):
        proj = d
        break
print('\n[project root]', proj)

roots = [proj]
for cand in ['/Users/sayan/Projects/weight-space-programming','/models','/data','/workspace',
             '/mnt','/tmp', os.path.expanduser('~/models'), os.path.expanduser('~/data'),
             os.path.expanduser('~/datasets'), os.path.expanduser('~/Downloads')]:
    if not os.path.isdir(cand):
        continue
    ac = os.path.abspath(cand)
    if all(os.path.abspath(r) != ac for r in roots):
        roots.append(cand)

models = []
seen = set()
for r in roots:
    try:
        for m in find_model_dirs(r, 8):
            if m not in seen:
                seen.add(m)
                models.append(m)
    except Exception as e:
        print('search error', r, repr(e))
models.sort()

print('\n[model dirs with model.safetensors]', len(models))
for m in models[:40]:
    cfg = load_cfg(m)
    f = os.path.join(m, 'model.safetensors')
    print(' ', m, 'type=%s' % cfg.get('model_type'),
          'hidden=%s' % cfg.get('hidden_size'),
          'layers=%s' % cfg.get('num_hidden_layers'),
          'dtype=%s' % cfg.get('dtype'),
          'bytes=%d' % os.path.getsize(f))
if len(models) > 40:
    print('  ... (%d more)' % (len(models) - 40))

def is_gemma270(cfg):
    return (cfg.get('model_type') == 'gemma3_text'
            and cfg.get('hidden_size') == 640
            and cfg.get('num_hidden_layers') == 18)

gemma = [m for m in models if is_gemma270(load_cfg(m))]
f32 = [m for m in gemma if load_cfg(m).get('dtype') == 'float32']
if len(f32) >= 2:
    gemma = f32
print('\n[gemma3-270 candidates]', len(gemma))
for m in gemma[:60]:
    print(' ', m)
if len(gemma) > 60:
    print('  ... (%d more)' % (len(gemma) - 60))

def under_parent(m):
    try:
        return os.path.commonpath([os.path.abspath(m), os.path.abspath(parent)]) == os.path.abspath(parent)
    except ValueError:
        return False

chosen = list(gemma)
prefs = [
    under_parent,
    lambda m: ('pair100' in m.lower() or 'pair_100' in m.lower() or 'pair-100' in m.lower()),
    lambda m: '/results/' not in m,
]
for f in prefs:
    sel = [m for m in chosen if f(m)]
    if len(sel) >= 2:
        chosen = sel
        if len(sel) == 2:
            break
gemma = chosen

if len(gemma) != 2:
    print('\n[CANNOT ISOLATE PAIR] gemma3-270 candidates=%d' % len(gemma))
    sys.exit(0)

a, b = gemma
print('\n[selected A]', a)
print('[selected B]', b)

for m in (a, b):
    cfgp = os.path.join(m, 'config.json')
    if os.path.isfile(cfgp):
        with open(cfgp, 'rb') as f:
            print(' config sha256', m, hashlib.sha256(f.read()).hexdigest())

def read_header(path):
    with open(path, 'rb') as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError('file too short: ' + path)
        n = struct.unpack('<Q', first)[0]
        hdr = json.loads(f.read(n))
    return hdr, n

def hash_range(F, off, length, chunk=8 << 20):
    F.seek(off)
    h = hashlib.sha256()
    rem = length
    while rem > 0:
        c = F.read(min(chunk, rem))
        if not c:
            raise IOError('unexpected EOF')
        h.update(c)
        rem -= len(c)
    return h.digest()

fa = os.path.join(a, 'model.safetensors')
fb = os.path.join(b, 'model.safetensors')
print('\n[sizes] A=%d B=%d' % (os.path.getsize(fa), os.path.getsize(fb)))

try:
    ha, na = read_header(fa)
    hb, nb = read_header(fb)
except Exception as ex:
    print('header read error:', repr(ex))
    sys.exit(0)

ka = {k for k in ha if k != '__metadata__'}
kb = {k for k in hb if k != '__metadata__'}
print('[headers] A_tensors=%d B_tensors=%d onlyA=%d onlyB=%d' %
      (len(ka), len(kb), len(ka - kb), len(kb - ka)))
if ka - kb:
    print(' only A:', sorted(ka - kb)[:20])
if kb - ka:
    print(' only B:', sorted(kb - ka)[:20])

shared = sorted(ka & kb)
shape_bad = [k for k in shared
             if ha[k].get('shape') != hb[k].get('shape')
             or ha[k].get('dtype') != hb[k].get('dtype')]
print('[shared] %d  shape/dtype mismatches=%d' % (len(shared), len(shape_bad)))
if shape_bad:
    print(' mismatched:', shape_bad[:20])

same = 0
diff = []
with open(fa, 'rb') as Fa, open(fb, 'rb') as Fb:
    for k in shared:
        ta, tb = ha[k], hb[k]
        oa = ta['data_offsets'][0]
        sa = ta['data_offsets'][1] - oa
        ob = tb['data_offsets'][0]
        sb = tb['data_offsets'][1] - ob
        if sa != sb or k in shape_bad:
            diff.append((k, ta.get('dtype'), str(ta.get('shape')), sa, sb))
            continue
        da = hash_range(Fa, 8 + na + oa, sa)
        db = hash_range(Fb, 8 + nb + ob, sb)
        if da == db:
            same += 1
        else:
            diff.append((k, ta.get('dtype'), str(ta.get('shape')), sa, sb))

print('\n[tensor results] identical=%d/%d differing=%d' % (same, len(shared), len(diff)))
for x in diff[:80]:
    print(' DIFF:', x)
if len(diff) > 80:
    print(' ... (%d more diffs)' % (len(diff) - 80))

identical = (len(diff) == 0 and len(shape_bad) == 0
             and len(ka) == len(kb) and not (ka - kb))
print('\nALL_WEIGHT_TENSORS_IDENTICAL:', identical)
PY
```

If it prints `ALL_WEIGHT_TENSORS_IDENTICAL: True`, the two checkpoints have byte-identical weight tensors. Otherwise the `DIFF:` lines name exactly which tensors differ (and their dtypes/shapes), which can be quantified numerically in a follow-up run.