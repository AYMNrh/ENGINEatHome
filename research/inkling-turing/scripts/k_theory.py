#!/usr/bin/env python3
"""Per-token byte and FLOP budget of Inkling 2 Mini from the real GGUF tensor sizes and the 48 layer visits.
usage: k_theory.py <gguf> <read_bw_GBs> <int8_tops> <fp16_tflops>   (prints a JSON summary + markdown tables)"""
import json, re, struct, sys

path, BW, INT8, FP16 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
f = open(path, 'rb')
def rd(fmt):
    s = struct.calcsize(fmt); return struct.unpack(fmt, f.read(s))
def rstr():
    n, = rd('<Q'); return f.read(n).decode('utf8', 'replace')
def skip(t):
    if t == 8: rstr(); return
    if t == 9:
        et, = rd('<I'); n, = rd('<Q')
        for _ in range(n): skip(et)
        return
    f.read({0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[t])
def val(t):
    if t in (8, 9):
        if t == 8: return rstr()
        skip(9) if False else None
    fm = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
    return rd(fm[t])[0]
f.read(4); ver, = rd('<I'); nt, = rd('<Q'); nkv, = rd('<Q')
kv = {}; align = 32
for _ in range(nkv):
    k = rstr(); t, = rd('<I')
    if t == 9:
        skip(9); continue
    kv[k] = val(t)
align = kv.get('general.alignment', 32)
tens = []
for _ in range(nt):
    n = rstr(); nd, = rd('<I'); dims = rd('<' + 'Q'*nd); ty, = rd('<I'); off, = rd('<Q')
    tens.append([n, dims, ty, off])
hdr_end = f.tell(); data_start = (hdr_end + align - 1)//align*align
import os
fsize = os.path.getsize(path)
order = sorted(range(len(tens)), key=lambda i: tens[i][3])
for j, i in enumerate(order):
    nxt = tens[order[j+1]][3] if j + 1 < len(order) else fsize - data_start
    tens[i].append(nxt - tens[i][3])     # bytes incl. alignment padding (<= 31 B)

A = kv['general.architecture']
core0, core1, npass = kv[f'{A}.loop.start'], kv[f'{A}.loop.end'], kv[f'{A}.loop.count']
nphys = kv[f'{A}.block_count']
visits = {l: (npass if core0 <= l < core1 else 1) for l in range(nphys)}
n_exp, k_exp = kv[f'{A}.expert_count'], kv[f'{A}.expert_used_count']

def numel(d):
    p = 1
    for x in d: p *= x
    return p

cat = {}   # category -> [bytes per token, flops per token (2*MAC), count]
def add(c, b, fl):
    e = cat.setdefault(c, [0.0, 0.0]); e[0] += b; e[1] += fl
for n, d, ty, off, nb in tens:
    m = re.match(r'blk\.(\d+)\.(.*)', n)
    if not m:
        if n.startswith('output.'):
            add('lm head', nb, 2*numel(d))
        elif n.startswith('token_embd.'):
            add('token embedding (1 row)', nb / d[1], 0)
        else:
            add('global vectors', nb, 0)
        continue
    l, rest = int(m.group(1)), m.group(2)
    v = visits[l]
    if rest.startswith('loop_'):
        # controllers run in passes 2..npass of the core layers only
        add('loop controllers (F32)', nb * (npass - 1), 2*numel(d)*(npass - 1) if 'query' not in rest else 0)
    elif '_exps.' in rest and rest.endswith('.weight'):
        frac = k_exp / n_exp
        add('routed experts (top-8 of 128)', nb * frac * v, 2*numel(d)*frac*v)
    elif rest.startswith('ffn_gate_inp'):
        add('router (F32)', nb * v, 2*numel(d)*v)
    elif rest.startswith('attn_') and rest.endswith('.weight') and len(d) == 2 and d[0] >= 1024 and d[1] >= 24:
        add('attention projections', nb * v, 2*numel(d)*v)
    else:
        add('norms / conv taps / small vectors', nb * v, 0)

total_b = sum(e[0] for e in cat.values()); total_f = sum(e[1] for e in cat.values())
# KV (q8_0: 34 B per 32 values), K and V of 8 heads x 128, per visit
kv_tok = 2 * 1024 * 34 / 32
full = [l for l in range(nphys) if (l + 1) % 6 == 0]
n_full = sum(visits[l] for l in full); n_swa = sum(visits.values()) - n_full
res = dict(bandwidth_GBs=BW, int8_tops=INT8, fp16_tflops=FP16, visits=sum(visits.values()), full_attn_visits=n_full,
           categories={c: dict(MB=round(e[0]/1e6, 2), GFLOP=round(e[1]/1e9, 3)) for c, e in cat.items()},
           weights_MB_per_token=round(total_b/1e6, 1), linear_GFLOP_per_token=round(total_f/1e9, 3))
rows = []
for ctx in (0, 8192, 32768):
    kvb = kv_tok * (n_full * ctx + n_swa * min(ctx, 1024 + 256))
    b = total_b + kvb
    rows.append(dict(ctx=ctx, kv_MB=round(kvb/1e6, 1), total_MB=round(b/1e6, 1), decode_ceiling_tok_s=round(BW*1e9/b, 1)))
res['decode'] = rows
# prefill: int8 MMQ for quantized linears (~50% of measured INT8), attention ~ fp16 FA
att_flops = lambda L: 4 * n_full * 24 * 128 * (L/2) + 4 * n_swa * 24 * 128 * min(L/2, 1024)
res['prefill'] = [dict(L=L, flops_per_token_G=round((total_f + att_flops(L))/1e9, 3),
                       ceiling_tok_s=round(0.5*INT8*1e12/(total_f + att_flops(L)*(INT8/FP16)), 0)) for L in (512, 8192, 32768)]
print(json.dumps(res, indent=1))
