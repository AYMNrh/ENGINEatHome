"""Charts for the ENGINEatHome guide.
Every number is from the repo (docs/, research/) or the 2026-11-24 deep-opt baseline."""
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(OUT, exist_ok=True)
INK, MUTE, GRID = "#1f2328", "#6b7280", "#e5e7eb"
C1, C2, C3, C4, RED = "#2563eb", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTE,
                     "xtick.color": MUTE, "ytick.color": MUTE, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 100, "savefig.dpi": 200, "axes.titleweight": "bold", "axes.titlesize": 11,
                     "axes.titlelocation": "left", "text.color": INK})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def box(ax, x, y, w, h, text, fc="#eff6ff", ec=C1, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, p, q, color=MUTE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.4))


def f_quant():
    fig, ax = plt.subplots(figsize=(7, 3.3))
    names = ["16-bit (F16)\ncomputed", "8-bit (Q8_0)\ncomputed", "Q4_K_M\nObliterated, real file", "IQ3_S\nSwift, real file"]
    gb = [27 * 16 / 8, 27 * 8.5 / 8, 16.81, 12.12]
    cols = [MUTE, MUTE, C2, C3]
    ax.barh(names[::-1], gb[::-1], color=cols[::-1], height=0.62)
    for i, v in enumerate(gb[::-1]):
        ax.text(v + 0.6, i, f"{v:.1f} GB", va="center", fontsize=10, fontweight="bold")
    ax.axvline(23.6, color=RED, lw=1.6, ls="--")
    ax.text(24.2, 3.4, "RTX 2080 Ti: 22 GB card", color=RED, fontsize=9, va="top")
    ax.set_xlim(0, 62)
    ax.set_xlabel("file size of a ~27-billion-parameter model (GB)")
    ax.set_title("Same brain, four sizes: only the squeezed ones fit on the card")
    grid(ax, "x")
    save(fig, "quant")


def f_kv():
    ctx = [8192, 32768, 86016, 131072, 262144]
    f16 = [0.5, 2, 5.25, 8, 16]
    q8 = [0.266, 1.0625, 2.789, 4.25, 8.5]
    q4 = [0.141, 0.5625, 1.477, 2.25, 4.5]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for y, c, lab in [(f16, C4, "f16 (exact reference)"), (q8, C1, "q8_0 (about half)"), (q4, C3, "q4_0 (about a quarter)")]:
        ax.plot([k / 1024 for k in ctx], y, "-o", color=c, lw=2.2, label=lab)
        ax.text(266, y[-1], f"{y[-1]:g} GiB", color=c, va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 305)
    ax.set_xlabel("context window (thousand tokens)")
    ax.set_ylabel("KV cache (GiB)")
    ax.set_title("The KV cache grows in a straight line with the conversation")
    ax.legend(frameon=False, loc="upper left")
    grid(ax)
    save(fig, "kv")


def f_phases():
    fig, ax = plt.subplots(figsize=(7, 2.9))
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.text(0, 5.6, "PREFILL: read the whole prompt at once (parallel, compute-bound)", fontsize=10, fontweight="bold", color=C1)
    for i in range(12):
        box(ax, 0.1 + i * 1.12, 3.9, 1.0, 1.0, f"t{i + 1}", fc="#dbeafe", ec=C1, fs=8)
    ax.text(0, 3.45, "all tokens processed together: hundreds of tokens per second", fontsize=8.5, color=MUTE)
    ax.text(0, 2.65, "DECODE: write the answer one token at a time (serial, memory-bound)", fontsize=10, fontweight="bold", color=C2)
    for i in range(6):
        box(ax, 0.1 + i * 2.1, 0.9, 1.3, 1.0, f"new {i + 1}", fc="#fef3c7", ec=C2, fs=8)
        if i < 5:
            arrow(ax, (1.45 + i * 2.1, 1.4), (2.15 + i * 2.1, 1.4), C2)
    ax.text(0, 0.35, "every token re-reads ALL the weights from memory: tens of tokens per second", fontsize=8.5, color=MUTE)
    save(fig, "phases")


def f_ceiling():
    ctx = [16384, 65536, 131072, 185099, 242134]
    meas = [55.88, 45.80, 36.81, 30.5, 26.8]
    ceil = [546.8 / (3.216 + 0.113 + 8 * c * 2176 / 1e9) for c in ctx]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = [c / 1024 for c in ctx]
    ax.plot(x, ceil, "--", color=MUTE, lw=2, label="bandwidth ceiling (547 GB/s, bytes read per token)")
    ax.plot(x, meas, "-o", color=C1, lw=2.4, label="measured decode (Inkling 2 Mini, 2080 Ti)")
    ax.fill_between(x, meas, ceil, color=C1, alpha=0.07)
    for xi, m, c in zip(x, meas, ceil):
        ax.text(xi, m - 9, f"{m:.1f}", ha="center", color=C1, fontsize=9, fontweight="bold")
        ax.text(xi, c + 3, f"{c:.0f}", ha="center", color=MUTE, fontsize=8.5)
    ax.text(150, 62, "the gap is latency, not bandwidth:\nmeasured is ~37% of the ceiling everywhere", color=C1, fontsize=9, ha="center")
    ax.set_ylim(0, 175)
    ax.set_xlabel("tokens already in context (thousands)")
    ax.set_ylabel("decode tokens / second")
    ax.set_title("Longer context means slower answers; we sit well under the physical limit")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    grid(ax)
    save(fig, "ceiling")


def f_spec():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})
    labs = ["no MTP", "MTP depth 2\n(q4 KV)", "full stack\n(depth 3, f16 KV,\nRAM cache)"]
    v = [16.0, 24.10, 29.50]
    a.bar(labs, v, color=[MUTE, C2, C3], width=0.62)
    for i, y in enumerate(v):
        a.text(i, y + 0.6, f"{y:.1f}", ha="center", fontweight="bold")
    a.set_ylim(0, 35)
    a.set_ylabel("decode tok/s at 65K")
    a.set_title("Obliterated 27B: drafting\nnearly doubles decode")
    grid(a)
    a.tick_params(axis="x", labelsize=8)
    d = [3, 4, 5, 7]
    y = [80.4, 97.2, 90.2, 106.2]
    b.plot(d, y, "-o", color=C4, lw=2.4)
    for xi, yi in zip(d, y):
        b.text(xi, yi + 2.5, f"{yi:.0f}", ha="center", fontweight="bold", color=C4, fontsize=9)
    b.set_xticks(d)
    b.set_xlabel("draft depth (tokens guessed per step)")
    b.set_ylim(60, 118)
    b.set_title("DFlash2 on dense 27B:\nbest depth is not monotonic")
    grid(b)
    fig.tight_layout()
    save(fig, "spec")


def f_cache():
    fig, ax = plt.subplots(figsize=(7, 3.4))
    rows = [("Obliterated, 20K\nafter a side request", 44.1, 4.3), ("Obliterated, 52K\nafter a side request", 131.6, 5.9),
            ("Obliterated, 64K\nfollow-up", 158, 0.51), ("Inkling, 242K\nfollow-up", 556.1, 1.3)]
    y = list(range(len(rows)))[::-1]
    for yi, (n, cold, warm) in zip(y, rows):
        ax.barh(yi + 0.19, cold, height=0.34, color=RED, alpha=0.85)
        ax.barh(yi - 0.19, warm, height=0.34, color=C3)
        ax.text(cold * 1.15, yi + 0.19, f"{cold:g} s", va="center", fontsize=8.5)
        ax.text(warm * 1.15, yi - 0.19, f"{warm:g} s", va="center", fontsize=8.5, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 2500)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlabel("time to first token (seconds, log scale)")
    ax.set_title("A prompt cache turns minutes of waiting into about a second")
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=RED, alpha=.85), Rectangle((0, 0), 1, 1, color=C3)],
              labels=["no cache (re-read everything)", "cache hit"], frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    grid(ax, "x")
    save(fig, "cache")


def f_profile():
    items = [("attention (GQA decode kernel)", 33.9, C1), ("expert/projection mat-vec: IQ2_S", 10.0, C2),
             ("Q8_0 projections", 6.3, C2), ("expert mat-vec: NVFP4", 5.0, C2), ("small Q8_0 pair kernels", 4.7, C2),
             ("expert mat-vec: IQ3_XXS", 4.7, C2), ("sliding-window attention", 3.6, C1),
             ("everything else (glue, norms, conv, router)", 31.8, MUTE)]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    n = [i[0] for i in items][::-1]
    v = [i[1] for i in items][::-1]
    c = [i[2] for i in items][::-1]
    ax.barh(n, v, color=c, height=0.62)
    for i, yv in enumerate(v):
        ax.text(yv + 0.5, i, f"{yv:.1f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 42)
    ax.set_xlabel("share of GPU time per decoded token (nsys, 128K context)")
    ax.set_title("Profile at 128K: attention alone is a third of every token")
    grid(ax, "x")
    ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "profile")


def f_stairs():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    a.bar(["old\nruntime", "new fork\n(pure)", "+ fusions\n(CP3)"], [39.4, 57.0, 62.4], color=[MUTE, C2, C3], width=0.6)
    for i, y in enumerate([39.4, 57.0, 62.4]):
        a.text(i, y + 2, f"{y}", ha="center", fontweight="bold")
    a.axhline(170, color=RED, ls="--", lw=1.2)
    a.text(2.4, 160, "ceiling 170", color=RED, ha="right", fontsize=8.5)
    a.set_ylim(0, 185)
    a.set_title("Decode at 0K (tok/s)")
    grid(a)
    pv = [600, 1222, 1361, 1401]
    b.bar(["old\nruntime", "pure\nfork", "round 4", "+ NVFP4\ntile"], pv, color=[MUTE, C2, C3, C3], width=0.6)
    for i, y in enumerate(pv):
        b.text(i, y + 30, f"{y}", ha="center", fontweight="bold")
    b.set_ylim(0, 1700)
    b.set_title("Prefill pp512 (tok/s)")
    grid(b)
    fig.tight_layout()
    save(fig, "stairs")


def f_reject():
    items = [("grouped-expert verify kernel (N=4)", -28), ("IQ3_XXS fragment-direct dequant (pp)", -13),
             ("FA query tile 16 cols (32K)", -12), ("FA query tile 16 cols (8K)", -7),
             ("GQA per-head FA packing (32K)", -2.7), ("IQ3_XXS lookup split (pp)", -1),
             ("NVFP4 register prefetch", 1), ("IQ3_XXS half-K tile", 0)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    n = [i[0] for i in items][::-1]
    v = [i[1] for i in items][::-1]
    ax.barh(n, v, color=[RED if x < 0 else MUTE for x in v], height=0.6)
    for i, y in enumerate(v):
        ax.text(y + (0.8 if y >= 0 else -0.8), i, f"{y:+g}%", va="center", ha="left" if y >= 0 else "right",
                fontsize=9, fontweight="bold")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-34, 8)
    ax.set_xlabel("change vs the control (%)")
    ax.set_title("Most ideas lose: measured, rejected, kept on file")
    grid(ax, "x")
    ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "reject")


def f_fagqa():
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    a = [45.6, 36.7]
    b = [47.0, 38.5]
    x = np.arange(2)
    ax.bar(x - 0.18, a, 0.34, color=MUTE, label="default (mode 1, bit-exact)")
    ax.bar(x + 0.18, b, 0.34, color=C3, label="candidate (mode 2, not bit-exact)")
    for i in range(2):
        ax.text(x[i] - 0.18, a[i] + .6, f"{a[i]:.1f}", ha="center", fontsize=9)
        ax.text(x[i] + 0.18, b[i] + .6, f"{b[i]:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(0.18, 52, "+3.0%", ha="center", color=C3, fontweight="bold")
    ax.text(1.18, 43, "+4.8%", ha="center", color=C3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["64K context", "128K context"])
    ax.set_ylim(0, 60)
    ax.set_ylabel("decode tok/s")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Newest experiment: a faster attention split, not yet trusted")
    grid(ax)
    save(fig, "fagqa")


def f_hybrid():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.axis("off")
    ax.set_xlim(0, 64)
    ax.set_ylim(-3.4, 5.2)
    for i in range(64):
        att = (i % 4 == 3)
        ax.add_patch(Rectangle((i + 0.06, 2.8), 0.88, 1.6, fc=C2 if att else "#bfdbfe", ec="white"))
    ax.text(0, 4.75, "Qwen3.8-27B (Swift, Obliterated): 64 blocks = 16 'attention' (orange) + 48 'recurrent' (blue)",
            fontsize=9.2, fontweight="bold")
    seq = list(range(0, 6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    ax.text(0, 1.55, "Inkling 2 Mini: 24 stored layers, run 48 times; core layers 6-17 are looped three times",
            fontsize=9.2, fontweight="bold")
    for k, L in enumerate(seq):
        full = L in (5, 11, 17, 23)
        fc = C2 if full else ("#c4b5fd" if 6 <= L <= 17 else "#bfdbfe")
        ax.add_patch(Rectangle((k * 64 / 48 + 0.04, -0.3), 64 / 48 - 0.08, 1.4, fc=fc, ec="white"))
    ax.text(0, -1.3, "orange = full-attention visit (8 per token)   purple = looped core   blue = other visits (sliding window)",
            fontsize=8, color=MUTE)
    save(fig, "hybrid")


def f_moe():
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis("off")
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 7)
    random.seed(3)
    act = set(random.sample(range(128), 8))
    for i in range(128):
        r, c = divmod(i, 32)
        ax.add_patch(Rectangle((6 + c * 0.5, 4.5 - r * 0.65), 0.42, 0.55, fc=C2 if i in act else "#e5e7eb", ec="white"))
    box(ax, 0.2, 2.6, 3.6, 1.4, "token\n(one word-piece)", fc="#eff6ff", ec=C1)
    box(ax, 4.0, 2.6, 1.6, 1.4, "router", fc="#f3e8ff", ec=C4)
    arrow(ax, (3.8, 3.3), (4.0, 3.3))
    arrow(ax, (5.6, 3.3), (6.3, 3.3))
    ax.text(6, 6.4, "128 experts per layer. The router wakes only 8 for each token.", fontsize=10, fontweight="bold")
    ax.text(6, 0.5, "Only 8 of 128 expert banks are read per token (6%), so a 26B model\ndecodes more like a much smaller one.",
            fontsize=8.5, color=MUTE)
    save(fig, "moe")


def f_platform():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 13)
    box(ax, 0.3, 9.2, 4.6, 2.4, "OpenCode\nHermes\nPython scripts", fc="#eff6ff", ec=C1)
    box(ax, 0.3, 4.5, 4.6, 2.4, "Home Assistant\ndashboard", fc="#fef3c7", ec=C2)
    box(ax, 7.2, 3.0, 7.2, 8.9, "", fc="#f0fdf4", ec=C3)
    ax.text(10.8, 11.35, "Gateway (port 8080)", ha="center", fontweight="bold", color=C3)
    for i, t in enumerate(["API keys + roles", "session leases", "wrong model name: 409",
                           "usage log, metadata only", "profile apply: idle only,\nreal probe, auto-restore"]):
        box(ax, 7.6, 9.3 - i * 1.55, 6.4, 1.15, t, fc="white", ec=C3, fs=8)
    box(ax, 17.2, 7.8, 6.4, 3.4, "llama-server\n(one model at a time,\nloopback only)", fc="#fff7ed", ec=C2)
    box(ax, 17.2, 3.0, 6.4, 3.4, "RTX 2080 Ti 22 GB\n175 W limit", fc="#f3e8ff", ec=C4, bold=True)
    arrow(ax, (4.9, 10.4), (7.2, 10.0))
    arrow(ax, (4.9, 5.7), (7.2, 6.0))
    arrow(ax, (14.4, 9.5), (17.2, 9.5))
    arrow(ax, (20.4, 7.8), (20.4, 6.4))
    ax.text(12, 1.4, "Profiles: Swift (default), Inkling Balanced, Inkling Maximum, Obliterated.\n'Select' never switches the model; 'Apply' does.",
            ha="center", fontsize=8.5, color=MUTE)
    save(fig, "platform")


def f_coding():
    fig, ax = plt.subplots(figsize=(7, 3.3))
    rows = [("Strata Coder, 6 issues x 3 (guided)", 18, 18, C3), ("Strata Coder, guided 4 tasks", 4, 4, C3),
            ("Swift on RTX 5070 Ti, 4 tasks", 4, 4, C3), ("Swift on RTX 2080 Ti, 4 tasks", 3, 4, C2),
            ("Baseline in guided 4-task run", 2, 4, C2), ("Edge0-145B Q3_K at 16K", 0, 18, RED)]
    for i, (n, p, t, c) in enumerate(rows[::-1]):
        ax.barh(i, 1, color="#f3f4f6", height=0.58)
        ax.barh(i, p / t, color=c, height=0.58)
        ax.text(1.02, i, f"{p}/{t}", va="center", fontweight="bold", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.set_title("Strict coding passes: wins, partial passes and honest failures")
    ax.spines["bottom"].set_visible(False)
    save(fig, "coding")


def f_pyramid():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    lv = [("4  Kernel-level engineering", "#7c3aed", 0.0, 2.0), ("3  How the speed is won", "#2563eb", 2.0, 4.0),
          ("2  The core concepts", "#0d9488", 4.0, 6.0), ("1  The gist", "#f59e0b", 6.0, 7.8)]
    for t, c, y0, y1 in lv:
        def half(y):
            return 5 * (1 - y / 12.0)
        ax.fill([5 - half(y0), 5 + half(y0), 5 + half(y1), 5 - half(y1)], [y0 + 0.04, y0 + 0.04, y1 - 0.04, y1 - 0.04], color=c)
        ax.text(5, (y0 + y1) / 2, t, ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax.text(5, 7.9, "start here", ha="center", fontsize=9, color=MUTE)
    save(fig, "pyramid")


if __name__ == "__main__":
    for fn in [f_quant, f_kv, f_phases, f_ceiling, f_spec, f_cache, f_profile, f_stairs, f_reject, f_fagqa,
               f_hybrid, f_moe, f_platform, f_coding, f_pyramid]:
        fn()
    print("ok", len(os.listdir(OUT)))
