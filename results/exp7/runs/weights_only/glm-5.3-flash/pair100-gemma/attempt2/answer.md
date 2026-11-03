No prior context to preserve — the run-1 output gives a partial picture but the headline metrics (||B−A||_F = 2.44 over 235 tensors, Δ concentrated in gate/up/down with cos≈1.0000 printed) contain one suspicious ambiguity I must resolve before concluding anything: whether B−A is truly **parallel to A** (multiplicative edit) or whether that cos column was computed against the wrong buffer. Also the reported Δ norms look like a *uniform per-entry-RMS field* (~2.5e-4 across q/k/v/o/gate/up/down) which is inconsistent with per-tensor scaling unless the cos is real — so the next run has to decide cleanly between "Δ ∝ A (rescale edit)" and "Δ is an independent structured update (training-style)", plus identify the 31 ~zero tensors.

Run 2:

```bash
python3 - <<'PY'
import os, math, re, json, time
import numpy as np
from safetensors import safe_open
T0=time.time()
try:
    import torch as T; HAVE=True
except Exception:
    HAVE=False
if not HAVE:
    import ml_dtypes

def f32(t):
    if HAVE:
        return t.detach().float().numpy() if isinstance(t, T.Tensor) else np.asarray(t, np.float32)
    return np.asarray(t, np.float32)

class Store:
    def __init__(self, d, fw):
        self.h=[]; self.e={}
        for fn in sorted(os.listdir(d)):
            p=os.path.join(d,fn)
            if fn.endswith('.safetensors'):
                h=safe_open(p, framework=fw); self.h.append(h)
                for k in h.keys(): self.e[k]=(h,k)
            elif fn.endswith(('.pt','.pth','.bin')):
                sd = T.load(p, map_location='cpu') if HAVE else None
                if sd is not None:
                    for k,v in sd.items(): self.e[k]=('mem',v)
    def keys(self): return sorted(self.e.keys())
    def raw(self,k):
        x=self.e[k]
        if len(x)==2 and x[0]=='mem': return x[1]
        return self.e[k][0].get_tensor(self.e[k][1])

fw = 'pt' if HAVE else 'numpy'
Astore=Store('case/A',fw); Bstore=Store('case/B',fw)
keys=[k for k in Astore.keys() if k in Bstore.e]
def parse(k):
    m=re.match(r'model\.layers\.(\d+)\.(.+)$',k)
    if m: return int(m.group(1)), m.group(2)
    return -1, ('final_norm' if 'norm' in k else k)

rows=[]
for k in keys:
    la,ty=parse(k)
    ra=Astore.raw(k); rb=Bstore.raw(k)
    ex = f32(ra).tobytes()==f32(rb).tobytes()
    a=f32(ra); b=f32(rb)
    dt = ('b16' if str(ra.dtype).find('bf')>=0 or (HAVE and ra.dtype==T.bfloat16) else str(ra.dtype))
    n=a.size
    d=b-a; dn=float(np.linalg.norm(d.ravel())); an=float(np.linalg.norm(a.ravel()))
    x=d.ravel(); y=a.ravel()
    dot=float(x@y)
    cos=dot/(dn*an+1e-30) if dn>0 else float('nan')
    c=dot/(an*an+1e-30) if an>0 else float('nan')
    pf=float(np.linalg.norm(x-c*y)/dn)**2 if dn>0 else float('nan')
    xm=x-x.mean(); ym=y-y.mean()
    corr=float(xm@ym)/ (np.linalg.norm(xm)*np.linalg.norm(ym)+1e-30) if dn>0 else float('nan')
    xa=np.abs(x); ya=np.abs(y)
    xam=xa-xa.mean(); yam=ya-ya.mean()
    absc=float(xam@yam)/ (np.linalg.norm(xam)*np.linalg.norm(yam)+1e-30) if dn>0 else float('nan')
    nz=np.abs(y)>0
    sa=float(np.mean(np.sign(x[nz])==np.sign(y[nz]))) if dn>0 and nz.sum()>0 else float('nan')
    rowcos=rowcv=colcv=float('nan')
    if a.ndim==2 and dn>0:
        rn=np.linalg.norm(a,axis=1); dnr=np.linalg.norm(d,axis=1)
        ok=rn>0
        rowcos=float(np.mean(((a*d).sum(1))[ok]/(rn[ok]*dnr[ok]+1e-30)))
        rr=dnr[ok]/rn[ok]; rowcv=float(rr.std()/(rr.mean()+1e-30))
        cn=np.linalg.norm(a,axis=0); dnc=np.linalg.norm(d,axis=0); okc=cn>0
        rc2=dnc[okc]/cn[okc]; colcv=float(rc2.std()/(rc2.mean()+1e-30))
    rows.append(dict(k=k,lay=la,ty=ty,ex=ex,dt=dt,dn=dn,an=an,rel=dn/(an+1e-30),cos=cos,c=c,
                     pf=pf,corr=corr,absc=absc,sa=sa,rowcos=rowcos,rowcv=rowcv,colcv=colcv,
                     rho=dn/math.sqrt(n),n=n,amax=float(np.abs(d).max())))

print('n tensors:',len(rows),'| dtypes:',sorted(set(r['dt'] for r in rows)))
print('bit-identical count:',sum(r['ex'] for r in rows))

print('\n--- near-zero-delta tensors (sorted by d) ---')
z=[r for r in rows if r['ex'] or r['dn']<1e-4]
print('count(dn<1e-4 or bit-identical):',len(z))
for r in sorted(z,key=lambda r:r['dn'])[:40]:
    print('%-52s ex=%d d=%.2e rel=%.2e maxd=%.2e cos=%+.3f'%(r['k'],r['ex'],r['dn'],r['rel'],r['amax'],r['cos']))

from collections import defaultdict
by=defaultdict(list)
for r in rows: by[r['ty']].append(r)
print('\n--- per-type medians: rel | cos | corr | absc | pf | sa | c | rho(2D) | rowcv | colcv ---')
for ty in sorted(by):
    g=by[ty]; med=lambda f: float(np.median([r[f] for r in g if r[f]==r[f]]))
    print('%-30s n=%3d rel=%.2e cos=%+.4f corr=%+.4f absc=%+.3f pf=%.3f sa=%.3f c=%.2e rho=%.2e rcv=%.3f ccv=%.3f'%(
        ty,len(g),med('rel'),med('cos'),med('corr'),med('absc'),med('pf'),med('sa'),med('c'),med('rho'),med('rowcv'),med('colcv')))

lin=[r for r in rows if r['n']>2000]
sx=np.array([math.log10(math.sqrt(r['n'])) for r in lin]); sy=np.array([math.log10(r['dn']+1e-30) for r in lin])
sA=np.array([math.log10(r['an']+1e-30) for r in lin])
for nm,pred in [('d ~ sqrt(numel)',sx),('d ~ |A|',sA)]:
    p=np.polyfit(pred,sy,1); r2=1-np.var(sy-np.polyval(p,pred))/np.var(sy)
    print('loglog fit %s: slope=%.3f R2=%.3f'%(nm,p[0],r2))

print('\n--- spectra: A vs Delta for top-8 deltas ---')
top=sorted([r for r in rows if r['dn']>0],key=lambda r:-r['dn'])[:8]
def load(x): return f32(Astore.raw(x['k'])), f32(Bstore.raw(x['k']))
for r in top:
    k=r['k']; a,b=load(r); d=b-a
    try:
        Ua,Sa,Vta=np.linalg.svd(a,full_matrices=False)
        Sd=np.linalg.svd(d,compute_uv=False)
        print('%-52s'%k); print('  svA/S0:',np.round(Sa[:8]/Sa[0],3).tolist(),'| svD/S0:',np.round(Sd[:8]/max(Sd[0],1e-30),3).tolist())
        print('  ratio svD/svA:',np.round(Sd[:8]/(Sa[:8]+1e-30),3).tolist())
        Vd=np.linalg.svd(d,full_matrices=False)[2]
        M=Vd[:16]@Vta[:16].T
        so=np.linalg.svd(M,compute_uv=False)
        E16=((d@Vta[:16].T)**2).sum()/ (r['dn']**2)
        print('  sub-overlap(top16) s1=%.3f s2=%.3f | DeltaE in A-top16=%.3f'%(so[0],so[1],E16))
    except Exception as e:
        Sd=np.linalg.svd(d,compute_uv=False); print('%-52s svD only err=%s'%(k,repr(e)[:40]))

print('\n--- cross-layer Delta-direction coherence ---')
for ty in ['mlp.gate_proj.weight','mlp.up_proj.weight','mlp.down_proj.weight','self_attn.q_proj.weight']:
    g=[r for r in rows if r['ty']==ty and r['ty'] in ('x',) or r['ty']==ty]
    g=[r for r in rows if r['ty']==ty]
    if len(g)<4: continue
    shapes=set()
    D=[]
    for r in sorted(g,key=lambda r:r['lay']):
        a=f32(Astore.raw(r['k'])); b=f32(Bstore.raw(r['k'])); shapes.add(a.shape)
        D.append((b-a).ravel())
    if len(shapes)!=1: print(ty,'shape mismatch',shapes); continue
    X=np.stack(D); Xn=X/ (np.linalg.norm(X,axis=1,keepdims=True)+1e-30)
    Gn=Xn@Xn.T
    off=Gn[~np.eye(len(Gn),dtype=bool)]
    print('%-26s cross-layer cos(Δ,Δ): mean=%.3f med=%.3f min=%.3f max=%.3f'%(ty,off.mean(),np.median(off),off.min(),off.max()))
    Aa=np.stack([f32(Astore.raw(r['k'])).ravel() for r in sorted(g,key=lambda r:r['lay'])])
    An=Aa/(np.linalg.norm(Aa,axis=1,keepdims=True)+1e-30)
    Ga=An@An.T; ofa=Ga[~np.eye(len(Ga),dtype=bool)]
    print('%-26s cross-layer cos(A,A):  mean=%.3f med=%.3f min=%.3f max=%.3f'%(ty,ofa.mean(),np.median(ofa),ofa.min(),ofa.max()))

if HAVE:
    print('\n--- bf16 rescale reconstruction test (top sensors) ---')
    for r in top[:3]:
        ra=Astore.raw(r['k']); rb=Bstore.raw(r['k'])
        a=f32(ra); braw=rb
        s=r['c']
        rec=(a*(1.0+s))
        if hasattr(rec,'dtype') and HAVE:
            rb2=rb.detach() if isinstance(rb,T.Tensor) else None
            if rb2 is not None:
                recb=rec if rec.dtype==rb2.dtype else rec.to(rb2.dtype) if isinstance(rec,T.Tensor) else T.from_numpy(rec).to(rb2.dtype)
                m=float((recb==rb2).float().mean())
                print('%-52s s=%.4e match=%.4f'%(r['k'],s,m))
print('elapsed %.1fs'%(time.time()-T0))
json.dump(rows,open('stats_run2.json','w'))
print('saved stats_run2.json')
PY
```

What this will disambiguate for the final verdict:

1. **cos vs corr separation** — if median per-tensor `corr(Δ, A)` ≈ 1, B is a per-tensor rescale of A (and run-1's cos was real); if corr ≈ 0 while run-1's "cos" printed 1.0000, that column was buggy (likely ⟨Δ,Δ⟩) and Δ is an independent structured update. `pf` (orthogonal-energy fraction) gives the same answer independently.
2. **The 31 zeros** — exact bit-identity listing will reveal which tensors B left untouched (likely final norm + a subset of RMSNorms), which is a strong recipe fingerprint for an injected/scalar edit vs. full continued training (where nothing is bit-identical).
3. **Per-entry Δ RMS (rho)** — if it collapses to a single constant (~2.5e-4) across *all* tensor types, the update has uniform step size (Adam-like training or a norm-normalized injection); combined with (1) this tells whether it's parallel-structured or independent.
4. **Spectral ratio svD/svA constant ⇒ Δ ∝ A**; A-top-16 explained energy of Δ tells whether SGD "spectral-bias"-style alignment is present (typical for real fine-tuning).
5. **Cross-layer Δ-Δ coherence vs A-A coherence** — a shared update direction across layers (mean cos >> 0 with Δ-Δ ≈ A-A) means Δ is inherited from the weights themselves (rescale); a shared direction *unrelated to* A-A suggests a single global training signal (data-driven update); near-zero cross-layer coherence means per-layer independent noise.
6. **bf16 rescale reconstruction** — if `bf16(A·(1+s))` reproduces B bit-exactly on the top tensors, the transformation is literally a per-tensor scalar multiplication, and we can then recover the exact per-tensor s values in run 3.