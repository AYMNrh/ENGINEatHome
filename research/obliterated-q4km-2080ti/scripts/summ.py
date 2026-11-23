#!/usr/bin/env python3
import json, sys, statistics as st, collections
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "ledger.jsonl")]
g = collections.OrderedDict()
for r in rows:
    if r.get("failed_to_start"):
        g.setdefault(r["label"], []).append(r); continue
    g.setdefault(r["label"].split(":")[0], []).append(r)
med = lambda xs: round(st.median(xs), 2) if xs else None
print(f"{'exp':16s}{'n':>3}{'agg_tps':>8}{'cold_pf':>8}{'tail_pf':>8}{'dec_med':>8}{'dec_min':>8}{'dec_max':>8}{'step_ms':>8}{'acc':>6}{'t/step':>7}{'depth':>6}{'Pw':>6}{'SMmhz':>6}{'gpu%':>5}{'mem%':>5}{'VRAM':>7}")
for k, rs in g.items():
    if rs[0].get("failed_to_start"):
        print(f"{k:16s} FAILED TO START"); continue
    cold = [r["prefill_tps"] for r in rs if r["turn"] == 0 and r.get("prefill_tps")]
    tail = [r["prefill_ms"] for r in rs if r["turn"] > 0]
    tailn = [r["prompt_n"] for r in rs if r["turn"] > 0]
    tailtps = med([n / (ms / 1000) for n, ms in zip(tailn, tail) if n and ms])
    d = [r["decode_tps"] for r in rs]
    print(f"{k:16s}{len(rs):>3}{round(sum(r['completion_tokens'] for r in rs)/(sum(r['decode_ms'] for r in rs)/1000),2):>8}{med(cold) or 0:>8}{tailtps or 0:>8}{med(d):>8}{min(d):>8}{max(d):>8}{med([r['step_ms'] for r in rs]):>8}"
          f"{med([r['accept_rate'] for r in rs if r['accept_rate'] is not None]) or 0:>6}{med([r['tok_per_step'] for r in rs if r['tok_per_step']]) or 0:>7}"
          f"{med([r['eff_depth'] for r in rs if r['eff_depth']]) or 0:>6}{med([r.get('power_mean_w', 0) for r in rs]):>6}{med([r.get('sm_mhz_mean', 0) for r in rs]):>6}"
          f"{med([r.get('gpu_util_mean', 0) for r in rs]):>5}{med([r.get('mem_util_mean', 0) for r in rs]):>5}{max(r.get('vram_peak_mib', 0) for r in rs):>7.0f}"
          + ("  LOOP" if any(r["loop_flag"] for r in rs) else ""))
