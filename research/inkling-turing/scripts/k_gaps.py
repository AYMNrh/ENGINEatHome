import sqlite3, sys, statistics, collections
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
ks = c.execute("select start, end, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL join StringIds s on s.id=shortName order by start").fetchall() if False else None
ks = c.execute("select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
# decode window = from the first logits D2H copy minus a token to the end
d2h = [r[0] for r in c.execute("select start from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind=2 order by start")]
t0 = d2h[0] - 30_000_000
print('decode window from', (ks[-1][1]-t0)/1e6, 'ms before end;', len(d2h), 'D2H logits copies')
w = [k for k in ks if k[0] >= t0]
tok = len(d2h)
gaps = [(b[0]-a[1], a[2], b[2]) for a, b in zip(w, w[1:]) if b[0] > a[1]]
bins = [(0,1),(1,5),(5,20),(20,60),(60,200),(200,1000),(1000,10**9)]
print(f'kernels in window {len(w)}  tokens {tok}')
for lo, hi in bins:
    g = [x for x in gaps if lo*1000 <= x[0] < hi*1000]
    print(f'  gap {lo:5d}-{hi if hi<10**9 else "inf"} us: n={len(g):6d} ({len(g)/tok:6.1f}/token) total {sum(x[0] for x in g)/1e6:7.1f} ms ({sum(x[0] for x in g)/1e6/tok:5.2f} ms/token)')
big = collections.Counter((x[1][:40], x[2][:40]) for x in gaps if x[0] > 20000)
print('most common (prev kernel -> next kernel) for gaps > 20us:')
for (a, b), n in big.most_common(8): print(f'  {n:5d}  {a} -> {b}')
mc = c.execute("select copyKind, count(*), sum(bytes), min(bytes), max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where start >= ? group by copyKind", (t0,)).fetchall()
print('memcpy in decode window (kind,count,bytes,min,max):', mc)
