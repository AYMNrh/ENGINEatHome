"""Render README cards from published measurements; optional dependency: Pillow."""
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'docs/assets'
W, H, SCALE = 1600, 1080, 2
BG, PANEL, BORDER = '#090f12', '#111d22', '#273b43'
WHITE, MUTED, GREEN, CYAN, AMBER = '#edf5f6', '#a6bac2', '#b5f04c', '#58d9e3', '#ffc16b'


def load(name):
    return json.loads((ROOT / 'docs/results' / name).read_text(encoding='utf-8-sig'))


class Card:
    def __init__(self, title, subtitle, number, date='10 NOV 2026'):
        self.image = Image.new('RGB', (W*SCALE, H*SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = {}
        self.rect((24, 24, 1576, 1056), BG, BORDER, 20)
        self.text(52, 42, 'ENGINEatHome', 22, GREEN, True)
        self.text(1270, 42, date+' / '+number, 21, MUTED)
        self.text(52, 86, title, 46, WHITE, True)
        self.text(52, 146, subtitle, 24, MUTED)

    def font(self, size, bold):
        key = size, bold
        if key not in self.fonts:
            filename = 'consolab.ttf' if bold else 'consola.ttf'
            candidates = [Path('C:/Windows/Fonts') / filename,
                          Path('/usr/share/fonts/truetype/dejavu') /
                          ('DejaVuSansMono-Bold.ttf' if bold else 'DejaVuSansMono.ttf')]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise RuntimeError('Install Consolas or DejaVu Sans Mono to render cards')
            self.fonts[key] = ImageFont.truetype(str(path), size*SCALE)
        return self.fonts[key]

    def text(self, x, y, value, size=24, color=WHITE, bold=False, width=None):
        value = str(value)
        font = self.font(size, bold)
        bounds = self.draw.textbbox((0, 0), value, font=font)
        if (bounds[2]-bounds[0])/SCALE > (W-x-40 if width is None else width):
            raise ValueError('Text exceeds its card region: '+value)
        self.draw.text((x*SCALE, y*SCALE), value, font=font, fill=color)

    def rect(self, box, fill=PANEL, outline=None, radius=12):
        self.draw.rounded_rectangle(tuple(int(v*SCALE) for v in box), radius*SCALE,
                                    fill=fill, outline=outline, width=SCALE)

    def metric(self, index, label, value, detail, color=GREEN):
        x = 52+380*index
        self.rect((x, 198, x+356, 330), outline=BORDER)
        self.text(x+18, 211, label, 20, MUTED, width=320)
        self.text(x+18, 241, value, 40, color, True, width=320)
        self.text(x+18, 297, detail, 19, MUTED, width=320)

    def bar(self, x, y, width, value, maximum, color, label):
        if not 0 <= value <= maximum:
            raise ValueError('Bar value outside its explicit zero-based scale')
        self.rect((x, y, x+width, y+25), '#20323a', radius=4)
        self.rect((x, y, x+max(2, width*value/maximum), y+25), color, radius=4)
        self.text(x, y-29, label, 20, MUTED, width=width)

    def footer(self, lines):
        self.rect((52, 925, 1548, 1035), '#252219', '#6c5431')
        for i, line in enumerate(lines):
            self.text(70, 939+i*27, line, 21, AMBER, width=1460)

    def save(self, name):
        path = ASSETS / name
        self.image.resize((W, H), Image.Resampling.LANCZOS).save(path, optimize=True)
        print(path.relative_to(ROOT))


def coder_card():
    r = load('strata-coder-filled64k-20261110.json')
    resources, requests = r['resources'], r['requests']
    c = Card('CODER / FILLED 64K', 'RTX 5070 Ti 16GB + 64GB RAM / Strata 0.1.59 / synthetic probe', '01')
    c.metric(0, 'ACTUAL INPUT', f'{r["actual_input_tokens"]:,}', '65,536 capacity', CYAN)
    c.metric(1, 'COLD DECODE', f'{requests[0]["decode_tps"]:.1f} tok/s', 'Thinking OFF')
    c.metric(2, 'CACHED DECODE', f'{max(x["decode_tps"] for x in requests):.1f} tok/s', 'Best of 2 warm requests')
    checks = sum(x['canary_retrieval_passed'] and x['syntax_passed'] for x in requests)
    c.metric(3, 'MARKERS + SYNTAX', f'{checks}/{len(requests)}', 'Not coding-agent grades', CYAN)
    for x, title in [(52, 'DECODE / tok/s'), (824, 'FRESH PREFILL / tok/s')]:
        c.rect((x, 354, x+724, 654), outline=BORDER)
        c.text(x+22, 373, title, 25, WHITE, True)
    for i, request in enumerate(requests):
        label = ['Cold / no reuse', 'Cached 1 / 57,344 reused', 'Cached 2 / 57,344 reused'][i]
        y = 452+i*73
        for x, key, limit, color in [(76, 'decode_tps', 160, GREEN), (848, 'fresh_prefill_tps', 4000, CYAN)]:
            c.bar(x, y, 520, request[key], limit, color, label)
            c.text(x+535, y-2, f'{request[key]:,.1f}', 25, color, True, width=146)
    c.text(76, 625, 'Scale: 0 to 160 / accepted output', 19, MUTED)
    c.text(848, 625, 'Scale: 0 to 4,000 / fresh input only', 19, MUTED)
    values = [('PEAK VRAM', f'{resources["device_vram_peak_sampled_bytes"]/2**30:.2f} GiB'),
              ('NATIVE PEAK RSS', f'{resources["process_peak_rss_bytes"]/2**30:.2f} GiB'),
              ('MIN HOST RAM FREE', f'{resources["host_min_available_bytes"]/2**30:.2f} GiB'),
              ('GPU ENERGY', f'{resources["gpu_energy_wh"]:.2f} Wh')]
    for i, (label, value) in enumerate(values):
        x = 70+i*380
        c.text(x, 696, label, 19, MUTED)
        c.text(x, 729, value, 34, WHITE, True, width=340)
    c.text(70, 794, f'GPU: {resources["gpu_power_mean_watts"]:.0f} W mean / {resources["gpu_power_peak_watts"]:.0f} W peak / '
           f'{resources["gpu_temperature_peak_c"]:.1f} C peak', 24, MUTED)
    c.text(70, 839, '256 experts/layer / 10 active / retained weights 3.5 bpw', 24, CYAN)
    c.text(70, 878, 'INT8 KV / MTP4 / full PLE / hybrid GPU + RAM + SSD', 23, MUTED)
    c.footer(['3 requests. Repeated repository text; 3 canaries + Python compile, no code execution.',
              'Expert-tail skip 7 is approximate. Separate thinking-on agent validation is pending.',
              'GPU-only device-wide energy includes startup; RSS can include shared/mapped pages.'])
    c.save('coder-filled64k.png')


def quality_card():
    off = load('strata-iq3-hermes18-20261109.json')['counts']
    on = load('strata-iq3-thinking18-20261110.json')
    oc = load('strata-iq3-opencode18-20261110.json')
    report = (ROOT/'docs/strata-iq3-opencode18-20261110.md').read_text()
    decode = float(re.search(r'decode: \*\*([\d.]+) tok/s', report)[1])
    c = Card('CODING AGENTS / MEASURED QUALITY', 'Original IQ3_S / 64K capacity / six selected issues x three attempts', '02')
    c.metric(0, 'OPENCODE STRICT', f'{oc["validation"]["strict_passes"]}/18', 'Correct patch + clean stop')
    c.metric(1, 'PASSING PATCHES', f'{oc["validation"]["patch_passes"]}/18', 'Independent held-out tests', CYAN)
    c.metric(2, 'ACCEPTED DECODE', f'{decode:.1f} tok/s', 'Includes reasoning')
    c.metric(3, 'MAX ACTUAL INPUT', f'{oc["summary"]["actual_max_prompt_tokens"]:,}', 'Capacity is not prompt size', CYAN)
    c.rect((52, 354, 1548, 704), outline=BORDER)
    c.text(74, 372, 'COMPLETED MATRICES / counts out of 18', 25, WHITE, True)
    c.text(1050, 375, 'STRICT', 21, GREEN, True)
    c.text(1220, 375, 'PATCH', 21, CYAN, True)
    rows = [('Hermes / OFF', off['passes'], off['patch_grade_passes']),
            ('Hermes / ON', on['summary']['tasks_passed'], on['patch_passes']),
            ('OpenCode / 1K', oc['validation']['strict_passes'], oc['validation']['patch_passes'])]
    for i, (label, strict, patch) in enumerate(rows):
        y = 432+i*86
        c.text(78, y+10, label, 27, WHITE, True, width=310)
        for offset, value, color in [(0, strict, GREEN), (34, patch, CYAN)]:
            c.rect((430, y+offset, 1360, y+offset+22), '#20323a', radius=3)
            c.rect((430, y+offset, 430+930*value/18, y+offset+22), color, radius=3)
            c.text(1380, y+offset-4, f'{value}/18', 24, color, True, width=140)
    summary, resources = oc['summary'], oc['resources']
    for i, (label, value) in enumerate([
        ('MEDIAN EPISODE', f'{summary["median_task_seconds"]/60:.2f} min'),
        ('CACHE REUSE', f'{summary["cache_reuse_fraction"]*100:.1f}%'),
        ('TOOLS / ERRORS', f'{summary["tool_calls"]} / {summary["opencode_tool_error_states"]}'),
        ('GPU ENERGY', f'{resources["gpu_energy_wh"]:.1f} Wh')]):
        x = 70+i*380
        c.text(x, 743, label, 19, MUTED)
        c.text(x, 775, value, 32, WHITE, True, width=340)
    c.text(70, 837, 'OpenCode 1.18.16 / high effort / requested thinking 1K / output 8K', 23, MUTED)
    c.text(70, 875, 'Strict completion = graded patch + normal finish within the agent limits.', 23, CYAN)
    c.footer(['Historical OpenCode completion scores are affected by the shell-timeout unit bug.',
              'Harnesses had different step/time/output limits. This is not a full SWE-bench score.',
              'RAM reserve failed; tail skip 7 is approximate. Coder agent quality is not established.'])
    c.save('coding-quality.png')


def stability_card():
    r = load('strata-iq3-efficiency-20261110.json')
    s, a, b = r['stability'], r['profiles']['tail7'], r['profiles']['tail0']
    c = Card('STABILITY / EXPERT-TAIL COMPARISON', 'Original IQ3_S / corrected tool harness / separate measured experiments', '03')
    c.metric(0, 'STABILITY', 'PASS' if s['passed'] else 'FAIL', f'{s["elapsed_seconds"]/60:.0f} minutes / review')
    c.metric(1, 'REVIEW TURNS', str(s['review_turns']), 'Not new coding tasks', CYAN)
    c.metric(2, 'COMPLETED TOOLS', str(s['completed_tools']), 'Real tool execution')
    c.metric(3, 'COMPACTIONS', str(s['recorded_compactions']), 'Recorded native events', CYAN)
    c.rect((52, 354, 776, 752), outline=BORDER)
    c.text(76, 375, 'ONE-HOUR REVIEW / PASSED', 27, GREEN, True)
    for i, label in enumerate(['Compaction + exact memory recall', 'Restart + persisted message history',
                               'Tools working after both checks', 'Existing patch preserved',
                               f'Held-out grading: {s["grade"]["tests"]}/{s["grade"]["tests"]} tests']):
        c.text(76, 431+i*47, '+ '+label, 25, WHITE, width=675)
    c.text(76, 696, f'{s["resources"]["gpu_energy_wh"]:.1f} GPU Wh / review + checks', 23, MUTED)
    c.rect((824, 354, 1548, 752), outline=BORDER)
    c.text(848, 375, 'FOUR CODING ATTEMPTS / PROFILE', 26, WHITE, True)
    for i, (profile, data, color) in enumerate([('Skip 7 / ON', a, GREEN), ('Skip 0 / OFF', b, CYAN)]):
        y = 455+i*76
        c.bar(848, y, 500, data['decode_tps'], 120, color, profile+' / decode tok/s')
        c.text(1365, y-2, f'{data["decode_tps"]:.1f}', 28, color, True, width=150)
    c.text(848, 595, '                    ON       OFF', 21, MUTED)
    c.text(848, 630, f'Strict passes       {a["strict_passes"]}/4      {b["strict_passes"]}/4', 25)
    c.text(848, 666, f'Passing patches     {a["patch_passes"]}/4      {b["patch_passes"]}/4', 25)
    c.text(848, 706, f'Coding GPU Wh      {a["matched_task_gpu_wh"]:.1f}     {b["matched_task_gpu_wh"]:.1f}', 24, MUTED)
    c.text(70, 790, 'ON/OFF: Pytest x3 + Sphinx x1. All five strict failures hit 20 minutes.', 24, MUTED)
    c.text(70, 832, 'Observed coding decode includes reasoning. Different generated trajectories.', 23, MUTED)
    c.text(70, 875, 'Original IQ3_S RAM reserve: FAILED / 128K promotion: NOT PASSED', 25, AMBER, True)
    c.footer(['Stability tested an existing patch, not new coding throughput or filled-64K retrieval.',
              'Four coding attempts/profile cannot establish quality equivalence. No retries removed.',
              'Coding energy excludes both review runs. All energy is device-wide GPU-only.'])
    c.save('stability-and-experts.png')


def context_card():
    r = load('strata-coder-long-context-20261110.json')
    a, b = r['probes']
    c = Card('CODER / 128K + 256K CONTEXT', 'RTX 5070 Ti 16GB + 64GB RAM / Strata 0.1.59 / cold synthetic probes', '04')
    c.metric(0, '128K ACTUAL INPUT', f'{a["input_tokens"]:,}', '131,072 capacity', CYAN)
    c.metric(1, '256K ACTUAL INPUT', f'{b["input_tokens"]:,}', '262,144 capacity', CYAN)
    c.metric(2, 'DEPTH RETRIEVAL', '3/3 + 3/3', 'Markers at 10 / 50 / 90%')
    c.metric(3, 'OUTPUT PER PROBE', '54 tokens', 'Thinking OFF / no reuse', AMBER)
    for x, title in [(52, 'DECODE / tok/s'), (824, 'FRESH PREFILL / tok/s')]:
        c.rect((x, 354, x+724, 610), outline=BORDER)
        c.text(x+22, 373, title, 25, WHITE, True)
    for i, p in enumerate(r['probes']):
        y = 455+i*85
        for x, key, limit, color in [(76, 'decode_tps', 120, GREEN), (848, 'fresh_prefill_tps', 4000, CYAN)]:
            c.bar(x, y, 510, p[key], limit, color, f'{p["capacity"]//1024}K / cold request')
            c.text(x+525, y-2, f'{p[key]:,.1f}', 25, color, True, width=150)
    c.text(76, 577, 'Scale: 0 to 120 / short output sample', 19, MUTED)
    c.text(848, 577, 'Scale: 0 to 4,000 / zero cached input', 19, MUTED)
    c.text(76, 646, 'MEASURED PER RUN', 27, WHITE, True)
    c.text(785, 646, '128K', 27, WHITE, True)
    c.text(1190, 646, '256K', 27, WHITE, True)
    rows = [('Request latency', 'elapsed_seconds', 's'), ('Peak VRAM', 'vram_gib', 'GiB'),
            ('Native peak RSS', 'rss_gib', 'GiB'), ('Minimum host RAM free', 'ram_free_gib', 'GiB'),
            ('GPU energy incl. startup', 'gpu_wh', 'Wh')]
    for i, (label, key, unit) in enumerate(rows):
        y = 697+i*38
        c.text(76, y, label, 24, MUTED, width=620)
        c.text(785, y, f'{a[key]:.2f} {unit}', 25, WHITE, True, width=300)
        c.text(1190, y, f'{b[key]:.2f} {unit}', 25, WHITE, True, width=300)
    c.footer(['One cold request/context; repeated repository filler; 54 output tokens is not sustained speed.',
              'INT8 KV / MTP4 / full PLE / hybrid GPU + RAM / approximate expert-tail skip 7.',
              'Three-marker smoke only, not agent quality or promotion. Energy: device-wide GPU-only.'])
    c.save('coder-long-context.png')


def guided_card():
    r = load('strata-coder-workflow-20261111.json')
    measured = r['guided18']
    baseline, guided = (r['matched4'][key] for key in ('baseline', 'guided'))
    leader = r['ranking'][0]
    c = Card('GUIDED CODER / #1 IN CODING TASKS', 'RTX 5070 Ti 16GB + 64GB RAM / OpenCode / six selected issues x three', '05', '11 NOV 2026')
    c.metric(0, 'STRICT PASSES', f'{leader["strict_passes"]}/{leader["attempts"]}', '18/18 correct patches')
    c.metric(1, 'ACCEPTED DECODE', f'{measured["decode_tps"]:.1f} tok/s', 'Includes reasoning', CYAN)
    c.metric(2, 'FRESH PREFILL', f'{measured["fresh_prefill_tps"]:,.0f} tok/s', '43,633 max actual input', CYAN)
    c.metric(3, 'MIN HOST RAM FREE', f'{measured["min_free_ram_gib"]:.2f} GiB', '15.27 GiB peak VRAM')
    for x, title, key, limit in [(52, 'MATCHED 4 / TASK MINUTES', 'minutes', 60),
                                 (824, 'MATCHED 4 / GPU Wh', 'gpu_wh', 50)]:
        c.rect((x, 354, x+724, 630), outline=BORDER)
        c.text(x+24, 373, title, 25, WHITE, True)
        for y, data, label, color in [(455, baseline, 'Baseline / 2 strict, 3 patches', CYAN),
                                      (540, guided, 'Guidance / 4 strict, 4 patches', GREEN)]:
            c.bar(x+24, y, 510, data[key], limit, color, label)
            c.text(x+550, y-2, f'{data[key]:.2f}', 25, color, True, width=140)
        c.text(x+24, 596, f'Zero-based scale: 0 to {limit}', 20, MUTED)
    c.rect((52, 657, 1548, 904), outline=BORDER)
    c.text(76, 680, 'STABILITY GATE FAILED / 128K AGENT WORK SKIPPED', 29, AMBER, True)
    c.text(76, 736, 'Compaction recall passed initially. Recall after restart returned unknown.', 23, WHITE)
    c.text(76, 783, 'All 452 saved messages survived. Recent summaries omitted the marker.', 23, MUTED)
    c.text(76, 830, f'Coding 18: {measured["gpu_wh"]:.2f} GPU Wh / no agent timeouts / median {measured["median_task_seconds"]:.0f}s', 23, CYAN)
    c.footer(['Ranking applies to these six selected issues. Harness limits differ across configurations.',
              '64K capacity, no filler. INT8 KV / MTP4 / approximate expert-tail skip 7.',
              'Device-wide GPU energy only. Stability energy is separate from coding.'])
    c.save('coder-guided18.png')


def handoff_card():
    r = load('strata-coder-stability-handoff-20261111.json')
    resources = r['resources']
    c = Card('CODER / ONE-HOUR STABILITY PASS', 'RTX 5070 Ti 16GB + 64GB RAM / OpenCode / verified read then summary', '06', '11 NOV 2026')
    seconds = round(r['elapsed_seconds'])
    c.metric(0, 'STABILITY TIME', f'{seconds//60}m {seconds%60:02d}s', '64K capacity / no filler')
    c.metric(1, 'VERIFIED HANDOFFS', str(r['review_pairs']), 'One read, then summary', CYAN)
    c.metric(2, 'RECALL CHECKS', f'{r["recall_passes"]}/3', 'Compaction, restart, final')
    c.metric(3, 'HELD-OUT TESTS', f'{r["held_out_tests_passed"]}/{r["held_out_tests"]}', 'Existing patch unchanged', CYAN)
    c.rect((52, 354, 776, 740), outline=BORDER)
    c.text(76, 375, 'CHECKPOINTS / VERIFIED', 26, WHITE, True)
    for index, label in enumerate(['Compaction and exact recall',
        f'Restart retained {r["messages_retained_on_restart"]:,} messages', 'Tools worked after both checkpoints',
        'Final recall and patch grading passed', f'Max {r["max_dispatches_per_turn"]}/{r["dispatch_limit"]} requests per turn; no cap hit']):
        c.text(76, 430+index*53, '+ '+label, 23, GREEN, width=670)
    c.rect((824, 354, 1548, 740), outline=BORDER)
    c.text(848, 375, 'MEASURED / FULL HOUR', 26, WHITE, True)
    for index, label in enumerate([f'{r["decode_tps"]:.1f} tok/s accepted decode',
        f'{r["fresh_prefill_tps"]:,.0f} tok/s fresh prefill / {r["actual_max_prompt"]:,} max input',
        f'{resources["device_vram_peak_sampled_bytes"]/2**30:.2f} GiB VRAM peak / {resources["host_min_available_bytes"]/2**30:.2f} GiB RAM free min',
        f'{resources["gpu_energy_wh"]:.2f} GPU Wh / {resources["gpu_power_mean_watts"]:.0f} W mean',
        f'{resources["gpu_power_peak_watts"]:.1f} W peak / {resources["gpu_temperature_peak_c"]:.1f} C peak']):
        c.text(848, 430+index*53, label, 22, CYAN, width=670)
    c.text(70, 782, '128K CAPACITY CODING STAGE UNLOCKED', 30, GREEN, True)
    c.text(70, 840, 'Separate coding run. Measure actual input before making longer-context claims.', 23, MUTED)
    c.footer(['Review of an existing patch, not new coding throughput or filled-context retrieval.',
              'INT8 KV / MTP4 / approximate expert-tail skip 7. Earlier failed checks remain recorded.',
              'Device-wide GPU-only energy. Decode includes reasoning. Resource peaks are sampled.'])
    c.save('coder-stability-handoff.png')


def agent128_card():
    r = load('strata-coder-agent128-20261111.json')
    resources = r['resources']
    c = Card('CODER / 128K CAPACITY CODING', 'RTX 5070 Ti 16GB + 64GB RAM / guided OpenCode / six selected issues', '07', '11 NOV 2026')
    c.metric(0, 'STRICT PASSES', f'{r["strict_passes"]}/{r["attempts"]}', '6/6 correct patches')
    c.metric(1, 'MAX ACTUAL INPUT', f'{r["actual_max_prompt"]:,}', '131,072 capacity', AMBER)
    c.metric(2, 'ACCEPTED DECODE', f'{r["decode_tps"]:.1f} tok/s', 'Includes reasoning', CYAN)
    c.metric(3, 'FRESH PREFILL', f'{r["fresh_prefill_tps"]:,.0f} tok/s', '97.3% cache reuse', CYAN)
    c.rect((52, 354, 776, 893), outline=BORDER)
    c.text(76, 375, 'SIX ISSUES / ALL PASSED', 26, WHITE, True)
    for index, task in enumerate(r['tasks']):
        y = 456 + index*66
        c.bar(76, y, 515, task['seconds'], 600, GREEN, task['issue'])
        c.text(608, y-2, f'{task["seconds"]:.1f}s', 24, GREEN, True, width=150)
    c.text(76, 856, 'Zero-based scale: 0 to 600 seconds', 20, MUTED)
    c.rect((824, 354, 1548, 893), outline=BORDER)
    c.text(848, 375, 'ACTUAL CONTEXT / CAPACITY', 26, WHITE, True)
    c.bar(848, 456, 660, r['actual_max_prompt'], r['capacity'], AMBER, 'Longest actual input: 55,948 tokens')
    c.rect((1177, 454, 1179, 486), MUTED, radius=0)
    c.text(848, 495, '0', 19, MUTED)
    c.text(1160, 495, '64K', 19, MUTED)
    c.text(1462, 495, '128K', 19, MUTED)
    c.text(848, 540, 'Zero prompts exceeded 64K.', 25, AMBER, True)
    for index, label in enumerate([f'{r["task_minutes"]:.2f} task minutes / 0 agent timeouts',
        f'{resources["gpu_energy_wh"]:.2f} GPU Wh / {resources["gpu_power_mean_watts"]:.0f} W mean',
        f'{resources["device_vram_peak_sampled_bytes"]/2**30:.2f} GiB VRAM peak / {resources["host_min_available_bytes"]/2**30:.2f} GiB RAM free min',
        'Full one-hour 64K stability gate passed first.']):
        c.text(848, 612+index*58, label, 22, CYAN, width=670)
    c.footer(['Six selected issues. Capacity configured at 128K; coding above 64K remains untested.',
              'No filler or subagents. INT8 KV / MTP4 / approximate expert-tail skip 7.',
              'Device-wide GPU-only energy. Stability energy is excluded from these coding figures.'])
    c.save('coder-agent128.png')


def dense_iq3s_cards():
    data = load('dense-iq3s-20261112.json')
    rows = data['extended_64k_sweep']
    context64, context128, context256 = data['context_probes']
    c = Card('DENSE QWEN3.8-27B / 64K DECODE',
             'RTX 5070 Ti 16GB / Unsloth UD-IQ3_S / llama.cpp CUDA', '08', '12 NOV 2026')
    c.metric(0, 'PEAK EDIT DECODE', f'{rows[2]["requests"]["edit"]["decode_tps"]:.1f} tok/s', 'MTP5 / Q4 KV')
    c.metric(1, 'PEAK FRESH CODE', f'{rows[1]["requests"]["decode"]["decode_tps"]:.1f} tok/s', 'MTP4 / Q4 KV')
    c.metric(2, 'BEST FRESH PREFILL', f'{rows[4]["requests"]["decode"]["fresh_prefill_tps"]:,.0f} tok/s',
             'DFlash2 / depth 3', CYAN)
    c.metric(3, 'ACTUAL INPUT', f'{rows[0]["requests"]["decode"]["input_tokens"]:,}', '512 output per prompt', AMBER)
    c.rect((52, 354, 1548, 893), outline=BORDER)
    c.text(76, 375, 'PROFILE', 25, WHITE, True)
    c.text(410, 375, 'FRESH CODE / TOK/S', 25, WHITE, True)
    c.text(963, 375, 'CODE EDIT / TOK/S', 25, WHITE, True)
    labels = ('MTP3', 'MTP4', 'MTP5', 'MTP7', 'DFlash2 depth 3', 'DSpine3 (WSL)', 'MTP3 control (WSL)')
    for index, (label, profile) in enumerate(zip(labels, rows)):
        y = 432 + 55*index
        c.text(76, y, label, 22, MUTED if index >= 5 else WHITE, width=300)
        for x, task, color in ((410, 'decode', GREEN), (963, 'edit', CYAN)):
            value = profile['requests'][task]['decode_tps']
            c.rect((x, y+2, x+415, y+23), '#20323a', radius=4)
            c.rect((x, y+2, x+415*value/100, y+23), color, radius=4)
            c.text(x+434, y-1, f'{value:.1f}', 22, color, True, width=82)
    c.text(76, 851, 'Scale 0 to 100 tok/s / two cold prompts per profile / GPU layer and cache placement', 20, MUTED)
    c.footer(['Q4 target KV and Q8 draft KV throughout this chart. Repeated repository filler.',
              'DFlash2 finished both cold requests in about 59s; MTP5 took 61 to 62s.',
              'Speed probes only. Draft and KV settings still need coding and retrieval checks.'])
    c.save('dense-iq3s-64k.png')

    c = Card('DENSE QWEN3.8-27B / CONTEXT FIT',
             'RTX 5070 Ti 16GB / Unsloth UD-IQ3_S / MTP off', '09', '12 NOV 2026')
    c.metric(0, '64K PREFILL', f'{context64["fresh_prefill_tps"]:,.0f} tok/s', '64,000 fresh input', CYAN)
    c.metric(1, '128K PREFILL', f'{context128["fresh_prefill_tps"]:,.0f} tok/s', '130,816 fresh input', CYAN)
    c.metric(2, '128K DECODE', f'{context128["decode_tps"]:.1f} tok/s', '128 output tokens')
    c.metric(3, '256K RESULT', 'Stopped', 'Shared GPU memory', AMBER)
    c.rect((52, 354, 776, 893), outline=BORDER)
    c.text(76, 375, 'MEASURED DECODE / TOK/S', 25, WHITE, True)
    for index, (label, result) in enumerate((('64K / Q8 KV / 32 output', context64),
                                             ('128K / Q4 KV / 128 output', context128))):
        y = 473 + index*146
        c.bar(76, y, 520, result['decode_tps'], 50, GREEN, label)
        c.text(610, y-3, f'{result["decode_tps"]:.1f}', 26, GREEN, True)
        c.text(76, y+43, f'{result["peak_device_vram_mib"]/1024:.2f} GiB peak device VRAM', 21, MUTED)
    c.text(76, 838, 'Zero-based scale: 0 to 50 tok/s', 20, MUTED)
    c.rect((824, 354, 1548, 893), outline=BORDER)
    c.text(848, 375, '256K / STOPPED AT PLACEMENT', 25, WHITE, True)
    for index, attempt in enumerate(context256['attempts']):
        y = 473 + index*146
        c.bar(848, y, 520, attempt['shared_mib'], 1500, AMBER, f'Microbatch {attempt["ubatch"]} / shared GPU memory')
        c.text(1382, y-3, f'{attempt["shared_mib"]:,.0f} MiB', 22, AMBER, True)
    c.text(848, 767, 'GPU residency check refused both loads.', 21, AMBER, True)
    c.text(848, 838, 'Zero-based scale: 0 to 1,500 MiB', 20, MUTED)
    c.footer(['64K and 128K use different KV precision and output lengths.',
              '256K attempted Q4 KV at both microbatch sizes. No prefill or decode was run.',
              'Synthetic speed probes only. Agent and retrieval quality are unscored.'])
    c.save('dense-iq3s-context.png')


if __name__ == '__main__':
    ASSETS.mkdir(parents=True, exist_ok=True)
    coder_card()
    quality_card()
    stability_card()
    context_card()
    guided_card()
    handoff_card()
    agent128_card()
    dense_iq3s_cards()
