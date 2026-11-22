import csv, sys, re, collections
txt = open(sys.argv[1], newline='').read(); txt = txt[txt.index('Time (%)'):]
rows = []
for r in csv.DictReader(txt.splitlines()):
    rows.append((float(r['Total Time (ns)']), int(r['Instances']), float(r['Avg (ns)']), re.sub(r'\s+', ' ', r['Name'])))
TOK = 129
tot = sum(r[0] for r in rows); n = sum(r[1] for r in rows)
print(f'all: {n/TOK:.0f} launches/token, {tot/1e6/TOK:.2f} ms/token')
def fam(name):
    m = re.match(r'(?:void )?(\w+)<?', name); f = m.group(1) if m else name
    t = re.search(r'\(ggml_type\)(\d+)', name)
    return f + (f'[t{t.group(1)}]' if t and f.startswith('mul_mat') else '')
fams = collections.defaultdict(lambda: [0, 0.0])
for t, c, avg, name in rows:
    f = fam(name); fams[f][0] += c; fams[f][1] += t
print('per-token launches and ms by kernel family (top 30):')
for f, (c, t) in sorted(fams.items(), key=lambda x: -x[1][1])[:30]:
    print(f'  {f:42s} {c/TOK:7.1f} launches  {t/1e6/TOK:6.3f} ms  avg {t/c/1e3:6.1f} us')
for lo, hi in [(0, 4), (4, 8), (8, 20), (20, 1e9)]:
    s = [(t, c) for t, c, avg, name in rows if lo*1e3 <= avg < hi*1e3]
    print(f'kernels with avg {lo}-{hi if hi<1e9 else "inf"} us: {sum(c for t,c in s)/TOK:7.0f} launches/token, {sum(t for t,c in s)/1e6/TOK:6.2f} ms/token')
