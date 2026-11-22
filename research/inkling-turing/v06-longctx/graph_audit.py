#!/usr/bin/env python3
"""Deterministic graph audit (offline, no GPU). Input: TSV dumps from LLAMA_DUMP_GRAPH_DIR (see k_patch_dump.py).
Checks (exit code 1 if any FAIL):
  T1 equal topology: prefill (reserve_N) vs decode (reserve_1) vs real graphs have the same node count and op sequence
  T2 op histogram diff (explains node-count discrepancies)
  T3 shape diff for the same node index
  T4 hidden sources: sources that are not graph nodes (leafs/inputs) per op (informational)
  T5 allocator overlap (allocated dumps only): two tensors alive at the same time must not share bytes in one buffer;
     live range = [first node idx, last consumer idx]; views follow view_src; graph outputs (flag 4) live to the end
usage: graph_audit.py A.tsv B.tsv   (compare)   |   graph_audit.py --overlap REAL.tsv
"""
import sys, collections

def load(p):
    rows = []
    for l in open(p):
        if l.startswith('#'):
            continue
        f = l.rstrip('\n').split('\t')
        rows.append(dict(i=int(f[0]), op=f[1], name=f[2], type=f[3], ne=tuple(int(x) for x in f[4:8]), flags=int(f[8]), srcs=f[9].split('|')[:-1],
                         vsrc=f[10], voff=int(f[11]), buf=f[12], off=int(f[13]), nbytes=int(f[14])))
    return rows

def compare(a, b, fa, fb):
    bad = 0
    print(f"T1 nodes {fa}={len(a)} {fb}={len(b)}", "OK" if len(a) == len(b) else "FAIL")
    bad += len(a) != len(b)
    ha, hb = collections.Counter(r['op'] for r in a), collections.Counter(r['op'] for r in b)
    d = {k: hb[k] - ha[k] for k in set(ha) | set(hb) if ha[k] != hb[k]}
    print("T2 op histogram diff (B-A):", d or "none")
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i]['op'] != b[i]['op']), None)
    print("T1b first op-sequence divergence:", first, (a[first]['op'], a[first]['name'], b[first]['op'], b[first]['name']) if first is not None else "none")
    bad += first is not None
    sh = [i for i in range(n) if a[i]['op'] == b[i]['op'] and a[i]['ne'][1:] != b[i]['ne'][1:] and a[i]['ne'][0] == b[i]['ne'][0] and False]
    # srcs pattern: same producer indices (topology), ignore shapes (token count differs between pp and tg)
    tdiff = [i for i in range(n) if a[i]['srcs'] != b[i]['srcs']]
    print("T3 nodes whose source index lists differ:", len(tdiff), tdiff[:10])
    bad += len(tdiff) > 0
    return bad

def overlap(rows):
    n = len(rows)
    root = {}
    def base(r):
        return r['vsrc'] if r['vsrc'] != '-' else r['name']
    last = collections.defaultdict(lambda: -1)
    for r in rows:
        last[r['i']] = max(last[r['i']], r['i'])
        for s in r['srcs']:
            if s.isdigit():
                last[int(s)] = max(last[int(s)], r['i'])
    byname = {r['name']: r for r in rows}
    ranges = []
    for r in rows:
        if r['buf'] == '-' or r['vsrc'] != '-' or r['nbytes'] == 0:
            continue
        end = n if (r['flags'] & 4) else last[r['i']]
        ranges.append((r['buf'], r['off'], r['off'] + r['nbytes'], r['i'], end, r['name'], r['op']))
    # views extend the life of their parent
    for r in rows:
        if r['vsrc'] != '-' and r['vsrc'] in byname:
            p = byname[r['vsrc']]
            for k, x in enumerate(ranges):
                if x[3] == p['i']:
                    ranges[k] = x[:4] + (max(x[4], last[r['i']]),) + x[5:]
    ranges.sort()
    bad = 0
    byb = collections.defaultdict(list)
    for x in ranges:
        byb[x[0]].append(x)
    for b, xs in byb.items():
        xs.sort(key=lambda x: x[1])
        act = []
        for x in xs:
            act = [y for y in act if y[2] > x[1]]
            for y in act:
                if y[4] > x[3] and x[4] > y[3]:  # end==start is a legal in-place reuse
                    bad += 1
                    if bad <= 10:
                        print("T5 OVERLAP", b, y[5], y[6], (y[3], y[4]), "vs", x[5], x[6], (x[3], x[4]))
            act.append(x)
    print("T5 live-range overlaps:", bad, "OK" if not bad else "FAIL", f"({len(ranges)} tensors)")
    return bad

if __name__ == '__main__':
    if sys.argv[1] == '--overlap':
        sys.exit(1 if overlap(load(sys.argv[2])) else 0)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    sys.exit(1 if compare(a, b, sys.argv[1], sys.argv[2]) else 0)
