import struct, sys, re
f = open(sys.argv[1], 'rb')
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
    if t == 8: return rstr()
    if t == 9: skip9(); return None
    fm = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t]; return rd(fm)[0]
def skip9():
    et, = rd('<I'); n, = rd('<Q')
    for _ in range(n): skip(et)
f.read(4); ver, = rd('<I'); nt, = rd('<Q'); nkv, = rd('<Q')
kv = {}
for _ in range(nkv):
    k = rstr(); t, = rd('<I'); v = val(t)
    if t != 9: kv[k] = v
for k in sorted(kv):
    if k.startswith('tokenizer.') : continue
    print('KV', k, '=', str(kv[k])[:80])
names = []
for _ in range(nt):
    n = rstr(); nd, = rd('<I'); dims = rd('<' + 'Q'*nd); ty, = rd('<I'); off, = rd('<Q'); names.append((n, dims, ty))
blk = sorted({int(m.group(1)) for n, _, _ in names for m in [re.match(r'blk\.(\d+)\.', n)] if m})
print('tensors', nt, 'blocks present', blk[0], '..', blk[-1], 'count', len(blk))
for n, d, ty in names:
    if re.match(r'blk\.(6)\.', n) or not n.startswith('blk.'): print('T', n, d, 'type', ty)
print([n for n,_,_ in names if not n.startswith('blk.')][:30])
