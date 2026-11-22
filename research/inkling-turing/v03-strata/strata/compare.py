#!/usr/bin/env python3
"""Compare teacher-forced logits: CPU FP32 reference vs GPU batched forward, at a spec on/off divergence point.
usage: compare.py CASE   (needs results/ref_cpu_CASE.bin, results/ref_gpu_CASE.bin, results/caseCASE.json)"""
import json, os, struct, sys
import numpy as np
R = os.path.expanduser("~/llm/experiments/strata-ab/results")
case = sys.argv[1]
meta = json.load(open("%s/case%s.json" % (R, case)))
def load(p):
    with open(p, "rb") as f:
        n, V = struct.unpack("<ii", f.read(8)); ids = np.frombuffer(f.read(4 * n), dtype=np.int32)
        lg = np.frombuffer(f.read(4 * n * V), dtype=np.float32).reshape(n, V)
    return ids, lg
ids_c, cpu = load("%s/ref_cpu_%s.bin" % (R, case))
have_gpu = os.path.exists("%s/ref_gpu_%s.bin" % (R, case))
a, b = meta["off_tok"], meta["on_tok"]
last = cpu[-1]
top = np.argsort(-last)[:5]
print("case", case, "n", len(ids_c), "divergence position d =", meta["d"], "| M=1 (spec-off) token", a, "spec-on token", b)
print("  CPU fp32 reference at divergence: argmax", int(top[0]), "top5", [(int(t), round(float(last[t]), 4)) for t in top])
print("  logit(off tok) - logit(on tok) =", round(float(last[a] - last[b]), 5), "| reference prefers", "OFF(M=1) token" if last[a] > last[b] else "ON(spec) token")
if have_gpu:
    ids_g, gpu = load("%s/ref_gpu_%s.bin" % (R, case))
    lg = gpu[-1]
    print("  GPU batched forward at divergence: logit(off) - logit(on) =", round(float(lg[a] - lg[b]), 5), "argmax", int(np.argmax(lg)))
    err = np.abs(gpu - cpu)
    topk = np.argsort(-cpu, axis=1)[:, :20]
    e_top = np.take_along_axis(err, topk, axis=1)
    print("  GPU-vs-CPU abs logit error over each position's top-20 tokens: mean %.4f  p99 %.4f  max %.4f" % (e_top.mean(), np.percentile(e_top, 99), e_top.max()))
    am_c, am_g = cpu.argmax(1), gpu.argmax(1)
    mism = (am_c != am_g)
    srt = np.sort(cpu, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    print("  argmax mismatch GPU batched vs CPU fp32 over %d positions: %d (%.2f%%); positions with CPU top1-top2 gap < 0.05: %d (%.2f%%), < 0.2: %d" % (len(am_c), mism.sum(), 100 * mism.mean(), (gap < 0.05).sum(), 100 * (gap < 0.05).mean(), (gap < 0.2).sum()))
    if mism.any():
        print("  mismatching positions' CPU gaps:", [round(float(g), 4) for g in gap[mism]][:12])
gap_here = float(np.sort(last)[-1] - np.sort(last)[-2])
print("  CPU top1-top2 gap at the divergence position:", round(gap_here, 5))
