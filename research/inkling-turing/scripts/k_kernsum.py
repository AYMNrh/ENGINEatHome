import csv, sys, re
f = sys.argv[1]; top = int(sys.argv[2]) if len(sys.argv) > 2 else 22
rows = []
with open(f, newline='') as fh:
    txt = fh.read()
txt = txt[txt.index('Time (%)'):]
rd = csv.DictReader(txt.splitlines())
for r in rd:
    rows.append((float(r['Total Time (ns)']), int(r['Instances']), float(r['Avg (ns)']), r['Name']))
tot = sum(r[0] for r in rows)
print(f'total kernel time {tot/1e6:.1f} ms over {sum(r[1] for r in rows)} launches')
for t, n, avg, name in sorted(rows, reverse=True)[:top]:
    name = re.sub(r'\s+', ' ', name)[:118]
    print(f'{100*t/tot:5.1f}%  {t/1e6:8.2f} ms  n={n:6d}  avg={avg/1e3:8.1f} us  {name}')
