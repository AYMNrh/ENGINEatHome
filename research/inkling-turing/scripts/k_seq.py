import sqlite3, sys, re
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
ks = c.execute("select k.start, k.end, s.value, k.gridX, k.gridY from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName order by k.start").fetchall()
idx = [i for i, k in enumerate(ks) if k[2].startswith('softcap')]
a, b = idx[40], idx[41]            # one full token
tok = ks[a+1:b+1]
print('kernels in token:', len(tok), 'span %.2f ms' % ((tok[-1][1]-tok[0][0])/1e6))
# visit boundaries: find repeated flash_attn_ext_vec occurrences
fa = [i for i, k in enumerate(tok) if k[2].startswith('flash_attn_ext_vec')]
print('flash_attn positions', fa[:12])
s0, s1 = fa[10]-20, fa[11]-20       # a core-loop visit (period between consecutive attention kernels)
print('visit length', s1-s0)
for i in range(s0, s1):
    k = tok[i]; print(f'{i-s0:3d} {(k[1]-k[0])/1e3:6.1f}us grid=({k[3]},{k[4]}) {k[2][:70]}')
