import sqlite3, sys
db = sqlite3.connect(sys.argv[1]); c = db.cursor()
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
rt = 'CUPTI_ACTIVITY_KIND_RUNTIME'
print('runtime API calls (count, total ms, avg us):')
for n, cnt, tot in c.execute(f"select s.value, count(*), sum(r.end-r.start) from {rt} r join StringIds s on s.id=r.nameId group by s.value order by sum(r.end-r.start) desc limit 12"):
    print(f'  {n[:60]:60s} {cnt:8d} {tot/1e6:9.1f} ms {tot/cnt/1e3:8.1f} us')
# gaps between consecutive kernels on the GPU for the busiest stream
ks = c.execute("select start, end, streamId, graphNodeId, graphId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
print('kernels', len(ks), 'with graphId', sum(1 for k in ks if k[4]))
gaps = []; busy = 0
for a, b in zip(ks, ks[1:]):
    g = b[0] - a[1]
    if g > 0: gaps.append(g)
span = ks[-1][1] - ks[0][0]
busy = sum(k[1] - k[0] for k in ks)
print(f'span {span/1e6:.1f} ms, sum kernel {busy/1e6:.1f} ms, idle {100*(1-busy/span):.0f}%')
import statistics
print(f'inter-kernel gaps: n={len(gaps)} median {statistics.median(gaps)/1e3:.2f} us mean {sum(gaps)/len(gaps)/1e3:.2f} us')
big = [g for g in gaps if g > 50000]
print(f'gaps >50us: n={len(big)} total {sum(big)/1e6:.1f} ms')
