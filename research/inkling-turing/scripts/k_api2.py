import sqlite3, sys, statistics
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
print('kernel cols:', ' '.join(cols))
mc = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_MEMCPY)")]
print('memcpy cols:', ' '.join(mc))
print('memcpy by kind (copyKind 1=H2D 2=D2H 8=D2D 10=P2P): count, bytes, total us')
for r in c.execute("select copyKind, count(*), sum(bytes), sum(end-start)/1e3, min(bytes), max(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind"):
    print('  ', r)
ks = c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
span = ks[-1][1] - ks[0][0]; busy = sum(b - a for a, b in ks)
print(f'GPU timeline: span {span/1e6:.1f} ms, kernel-busy {busy/1e6:.1f} ms -> idle {100*(1-busy/span):.0f}%  ({len(ks)} kernels)')
gaps = [b[0] - a[1] for a, b in zip(ks, ks[1:]) if b[0] > a[1]]
big = [g for g in gaps if g > 100000]
small = [g for g in gaps if g <= 100000]
print(f'gaps <=100us: n={len(small)} median {statistics.median(small)/1e3:.2f} us total {sum(small)/1e6:.1f} ms | gaps >100us: n={len(big)} total {sum(big)/1e6:.1f} ms (median {statistics.median(big)/1e3:.0f} us)')
