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


def dense_dflash2_cards():
    data = load('dense-iq3s-dflash2-20261112.json')
    sweep, replay = data['depth_sweep'], data['replay']
    code, edit = (sweep[-1]['requests'][task] for task in ('decode', 'edit'))
    c = Card('DENSE 27B / DFLASH2 DEPTH SWEEP',
             'RTX 5070 Ti 16GB / Qwen3.8-27B UD-IQ3_S / llama.cpp CUDA', '10', '12 NOV 2026')
    c.metric(0, 'FRESH CODE DECODE', f'{code["decode_tps"]:.1f} tok/s', 'DFlash2 depth 7')
    c.metric(1, 'CODE EDIT DECODE', f'{edit["decode_tps"]:.1f} tok/s', 'DFlash2 depth 7')
    c.metric(2, 'FRESH PREFILL', f'{(code["prefill_tps"]+edit["prefill_tps"])/2:,.0f} tok/s', 'Depth 7 / two prompts', CYAN)
    c.metric(3, 'ACTUAL INPUT', f'{code["input_tokens"]:,}', '512 output per prompt', AMBER)
    c.rect((52, 354, 1548, 893), outline=BORDER)
    c.text(76, 375, 'DEPTH', 25, WHITE, True)
    c.text(290, 375, 'FRESH CODE / TOK/S', 25, WHITE, True)
    c.text(929, 375, 'CODE EDIT / TOK/S', 25, WHITE, True)
    for index, row in enumerate(sweep):
        y = 458 + 96*index
        c.text(76, y, f'{row["depth"]} drafts', 24, WHITE)
        for x, task, color in ((290, 'decode', GREEN), (929, 'edit', CYAN)):
            value = row['requests'][task]['decode_tps']
            c.rect((x, y+3, x+474, y+29), '#20323a', radius=4)
            c.rect((x, y+3, x+474*value/120, y+29), color, radius=4)
            c.text(x+491, y, f'{value:.1f}', 24, color, True)
    c.text(76, 842, 'Scale 0 to 120 tok/s / Q4 target KV / Q8 draft KV / GPU layer and cache placement', 20, MUTED)
    c.footer(['One cold sample per prompt and depth. Synthetic repository text with repeated filler.',
              'Depth 7 leads both prompts and advances to the OpenCode history replay.',
              'Throughput probes only. Coding and retrieval quality are unscored.'])
    c.save('dense-iq3s-dflash2-depths.png')

    cold, warm = (row['dflash2-7'] for row in replay[-2:])
    c = Card('DENSE 27B / OPENCODE HISTORY REPLAY',
             'RTX 5070 Ti 16GB / Qwen3.8-27B UD-IQ3_S / one saved Sphinx session', '11', '12 NOV 2026')
    c.metric(0, '64K COLD DECODE', f'{cold["decode_tps"]:.1f} tok/s', '63,529 fresh input')
    c.metric(1, '64K CACHED DECODE', f'{warm["decode_tps"]:.1f} tok/s', '64,000 actual input')
    c.metric(2, '64K FRESH PREFILL', f'{cold["prefill_tps"]:,.0f} tok/s', 'DFlash2 depth 7', CYAN)
    c.metric(3, '64K CACHED REQUEST', f'{warm["elapsed_seconds"]:.2f}s', '63,525 reused tokens', AMBER)
    c.rect((52, 354, 1548, 893), outline=BORDER)
    c.text(76, 375, 'ACTUAL INPUT / CACHE', 25, WHITE, True)
    c.text(432, 375, 'DFLASH2 DEPTH 7 / TOK/S', 25, WHITE, True)
    c.text(987, 375, 'MTP5 / TOK/S', 25, WHITE, True)
    for index, row in enumerate(replay):
        y = 432 + 64*index
        label = 'cached' if row['phase'] == 'warm' else 'cold'
        c.text(76, y, f'{row["dflash2-7"]["input_tokens"]:,} / {label}', 23, WHITE)
        for x, profile, color in ((432, 'dflash2-7', GREEN), (987, 'mtp5', CYAN)):
            value = row[profile]['decode_tps']
            c.rect((x, y+3, x+395, y+25), '#20323a', radius=4)
            c.rect((x, y+3, x+395*value/180, y+25), color, radius=4)
            c.text(x+413, y, f'{value:.1f}', 23, color, True)
    c.text(76, 843, 'Scale 0 to 180 tok/s / 256 output per request / Q4 target KV / Q8 draft KV', 20, MUTED)
    c.footer(['Reconstructed turns and tool outputs. Four saved source files extend the longest pair.',
              'DFlash2 peak device VRAM 14.95 GiB. MTP5 peak 14.08 GiB.',
              'Single-stream speed only. Tools were not executed and patches were not graded.'])
    c.save('dense-iq3s-dflash2-replay.png')


def dense_live_card():
    data = load('dense-iq3s-live-20261112.json')
    dense, coder = data['dense'], data['coder']
    c = Card('DENSE 27B VS STRATA CODER / LIVE AGENT',
             'RTX 5070 Ti 16GB / 64K capacity / four matched guided OpenCode attempts', '12', '12 NOV 2026')
    c.metric(0, 'DENSE STRICT PASSES', f'{dense["strict"]}/{dense["attempts"]}', '4/4 held-out patch passes', AMBER)
    c.metric(1, 'STRATA STRICT PASSES', f'{coder["strict"]}/{coder["attempts"]}', '4/4 held-out patch passes')
    c.metric(2, 'DENSE DECODE', f'{dense["decode_tps"]:.1f} tok/s', 'Includes reasoning', CYAN)
    c.metric(3, 'DENSE RETRIEVAL', '3/3 each', 'DFlash2 on and off')
    for x, field, scale, title, unit in (
        (52, 'task_minutes', 40, 'TOTAL CODING TIME', 'min'),
        (824, 'gpu_wh', 120, 'CODING ENERGY / GPU ONLY', 'Wh'),
    ):
        c.rect((x, 354, x+724, 893), outline=BORDER)
        c.text(x+24, 375, title, 25, WHITE, True)
        for index, (model, label, color) in enumerate((
            (dense, 'Dense IQ3_S + DFlash2 depth 7', CYAN),
            (coder, 'Strata Flash-Next GSQ-RCO Coder', GREEN),
        )):
            y = 483 + index*137
            value = model[field]
            c.bar(x+24, y, 490, value, scale, color, label)
            c.text(x+532, y-2, f'{value:.1f} {unit}', 25, color, True, width=166)
        if field == 'task_minutes':
            c.text(x+24, 736, f'{dense["tool_calls"]} dense tool calls / {coder["tool_calls"]} Strata', 22, MUTED)
            c.text(x+24, 783, 'Final dense Pytest hit the 100-step cap.', 21, AMBER)
        else:
            c.text(x+24, 736, 'Device-wide GPU energy / sampled', 22, MUTED)
            c.text(x+24, 783, 'Coding windows exclude retrieval.', 21, MUTED)
        c.text(x+24, 841, f'Scale 0 to {scale} {unit}', 20, MUTED)
    c.footer(['Four frozen attempts per profile. Both produced four passing patches.',
              'Dense retrieval: exact matches at 10%, 50%, and 90% of 64,000 fresh tokens.',
              'Dense and Strata use different models and runtimes. Coding prompts use no filler.'])
    c.save('dense-iq3s-live.png')


def dense_supervision_card():
    data = load('dense-iq3s-supervision-20261112.json')
    leader, _, guarded = data['ranked']
    hybrid = data['hybrid_partial']
    c = Card('64K CODING / GUARDED AND HYBRID QWEN',
             'RTX 5070 Ti 16GB / guided OpenCode / four frozen attempts', '13', '12 NOV 2026')
    c.metric(0, 'RANK 1 / STRATA CODER', f'{leader["strict"]}/4 strict', f'{leader["minutes"]:.1f} task minutes')
    c.metric(1, 'GUARDED QWEN', f'{guarded["strict"]}/4 strict', f'{guarded["minutes"]:.1f} task minutes', CYAN)
    c.metric(2, 'HYBRID / PARTIAL', f'{hybrid["strict"]}/{hybrid["graded"]} strict', '2/2 passing patches', AMBER)
    c.metric(3, 'HYBRID / GRADED TIME', f'{hybrid["task_minutes"]:.1f} min', 'Two graded attempts', AMBER)
    c.rect((52, 354, 776, 893), outline=BORDER)
    c.text(76, 375, 'COMPLETED PROFILES / TASK TIME', 24, WHITE, True)
    labels = ('Strata Flash-Next Coder', 'Dense IQ3_S + DFlash2', 'Dense IQ3_S + guard')
    for i, (profile, row) in enumerate(zip(labels, data['ranked'])):
        y = 473 + 121*i
        color = GREEN if i == 0 else CYAN
        label = f'{row["rank"]}. {profile} / {row["strict"]} strict, {row["patch"]} patch'
        c.bar(76, y, 540, row['minutes'], 40, color, label)
        c.text(630, y-3, f'{row["minutes"]:.1f}m', 24, color, True)
    c.text(76, 846, 'Scale 0 to 40 min / four attempts per profile', 19, MUTED)
    c.rect((824, 354, 1548, 893), outline=BORDER)
    c.text(848, 375, 'CODEX + QWEN / STOPPED BY USER', 24, WHITE, True)
    usage = hybrid['supervisor_usage']
    for i, label in enumerate((
        'Pytest: strict pass and held-out patch pass',
        'Sphinx: step cap, held-out patch pass',
        'One interrupted trial / one never started',
        f'{hybrid["supervisor_calls"]} Codex calls / {hybrid["supervisor_seconds"]:.0f}s on graded cases',
        f'{usage["input_tokens"]:,} input / {usage["output_tokens"]:,} output tokens',
        f'{hybrid["takeovers"]} takeovers / Qwen wrote both patches',
    )):
        c.text(848, 451+64*i, label, 22, CYAN if i >= 3 else MUTED, width=665)
    c.footer(['Ranking uses strict passes, then held-out patch passes, then total task time.',
              'Hybrid is unranked. Its time and token figures above cover the two graded cases.',
              'One Pytest trial passed after a guard pause. Guarded Sphinx timed out.'])
    c.save('dense-iq3s-supervision.png')


def qwen36_hardest_context_card():
    data = load('qwen36-hardest-context-20261114.json')
    p64, p92, p128 = data['profiles']
    c = Card('QWEN3.6 / HARDEST TASK WINNER',
             '35B-A3B IQ3_XXS / OpenCode / Q4 KV / Flash Attention / speculation off',
             '14', '14 NOV 2026')
    c.metric(0, 'HISTORICAL BASELINE', '0/9 strict', 'Sphinx #10449 / 2 patches', AMBER)
    c.metric(1, '128K RESULT', '1/1 strict', '21/21 held-out tests')
    c.metric(2, '128K DECODE', f'{p128["decode_tokens_per_second"]:.1f} tok/s',
             '37,333 max actual prompt', CYAN)
    c.metric(3, '92K HEADROOM', f'{p92["minimum_free_device_vram_mib"]:.0f} MiB',
             '72,937 max actual prompt', CYAN)

    c.rect((52, 354, 776, 893), outline=BORDER)
    c.text(76, 375, 'AGENT RESULT / SAME HARDEST ISSUE', 24, WHITE, True)
    for index, profile in enumerate(data['profiles']):
        y = 471 + index*131
        capacity = profile['context_capacity'] // 1024
        result = 'STRICT PASS' if profile['strict_pass'] else 'PATCH PASS / STEP CAP'
        color = GREEN if profile['strict_pass'] else AMBER
        c.text(76, y-52, f'{capacity}K / {result}', 21, color, True, width=650)
        c.bar(76, y+18, 490, profile['decode_tokens_per_second'], 180, color,
              f'{profile["maximum_actual_prompt_tokens"]:,} max actual prompt')
        c.text(582, y+16, f'{profile["decode_tokens_per_second"]:.1f}', 24, color, True)
    c.text(76, 844, 'Bars: aggregate accepted decode / scale 0-180 tok/s', 19, MUTED)

    c.rect((824, 354, 1548, 893), outline=BORDER)
    c.text(848, 375, 'GPU MEMORY / STATIC CONTEXT ALLOCATION', 24, WHITE, True)
    for index, profile in enumerate(data['profiles']):
        y = 471 + index*131
        capacity = profile['context_capacity'] // 1024
        free = profile['minimum_free_device_vram_mib']
        c.text(848, y-52, f'{capacity}K / {profile["explicit_gpu_buffers_mib"]:,.0f} MiB explicit',
               21, CYAN, True, width=650)
        c.bar(848, y+18, 490, free, 600, CYAN,
              f'Q4 KV {profile["gpu_kv_buffer_mib"]:,.0f} MiB / minimum free')
        c.text(1354, y+16, f'{free:.0f} MiB', 24, CYAN, True, width=160)
    c.text(848, 844, 'Peak includes CUDA/runtime and other device users.', 19, MUTED)
    c.footer(['Provisional hardest-task winner: first strict pass after 0/9 historical attempts.',
              'Strata Coder retains the broader 18/18 consistency result.',
              'Single attempts; 128K strict pass peaked at 37,333 actual prompt tokens.'])
    c.save('qwen36-hardest-context.png')


def bonsai_cards():
    data = load('bonsai-dsh-256k-20261117.json')
    card = Card('Coding quality, runtime and throughput',
                'Same six frozen issues / three attempts each / model and harness settings vary',
                'CODING', date='17 NOV 2026')
    card.text(54, 207, 'Configuration / capacity', 22, MUTED)
    card.text(530, 207, 'Strict / patch (out of 18)', 19, MUTED)
    card.text(850, 207, 'Task time (min)', 19, MUTED)
    card.text(1125, 207, 'Fresh prefill', 19, MUTED)
    card.text(1370, 207, 'Decode', 19, MUTED)
    card.text(1125, 231, 'tok/s', 17, MUTED)
    card.text(1370, 231, 'tok/s', 17, MUTED)
    for i, row in enumerate(data['matched_issue_matrices']):
        y = 260 + 87*i
        card.text(54, y+4, row['label'], 21, WHITE, True)
        card.text(54, y+32, f"{row['capacity']//1024}K capacity", 19, MUTED)
        for offset, key, color in [(0, 'strict', GREEN), (26, 'patches', CYAN)]:
            value = row[key]
            card.rect((530, y+offset, 755, y+offset+17), '#20323a', radius=3)
            if value:
                card.rect((530, y+offset, 530+225*value/18, y+offset+17), color, radius=3)
            card.text(765, y+offset-4, f'{value:2d}/18', 19, color)
        minutes = row['task_minutes']
        card.rect((850, y+10, 1020, y+31), '#20323a', radius=3)
        card.rect((850, y+10, 850+170*minutes/250, y+31), AMBER, radius=3)
        card.text(1030, y+10, f'{minutes:.1f}', 20, AMBER)
        card.text(1125, y+8, f"{row['prefill_tps']:,.1f}", 24, CYAN, True)
        card.text(1370, y+8, f"{row['decode_tps']:,.1f}", 24, GREEN, True)
    card.footer(['Strict pass = normal completion and passing held-out tests. Task time includes failures.',
                 'Prefill counts fresh tokens. Decode includes reasoning. Both are weighted by engine time.'])
    card.save('bonsai-coding-comparison.png')

    card = Card('Bonsai DSH / all 18 attempts', '256K capacity / 20-minute agent budget per attempt',
                'TRIALS', date='17 NOV 2026')
    for x, label, color in [(54, 'Strict pass', GREEN), (270, 'Patch only', CYAN), (478, 'Patch failed', AMBER)]:
        card.text(x, 194, label, 22, color, True)
    card.text(463, 231, '0', 19, MUTED)
    card.text(1132, 231, '20 min', 19, MUTED)
    for i, row in enumerate(data['bonsai_256k']['cases']):
        y = 263 + 34*i
        task = row['task'].split('__')[1].replace('dev__', '')
        card.text(55, y-3, f"{i+1:02d}  {task} / trial {row['trial']}", 19)
        color = GREEN if row['strict'] else CYAN if row['patch'] else AMBER
        card.rect((463, y, 1180, y+17), '#20323a', radius=3)
        card.rect((463, y, 463+717*row['seconds']/1220, y+17), color, radius=3)
        tick = 463+717*1200/1220
        card.rect((tick, y, tick+1, y+17), WHITE, radius=0)
        card.text(1198, y-3, f"{row['seconds']/60:5.1f} min  {row['finish']}", 19, color)
    card.footer(['Five timeouts and four token-limit exits. Four of the timeouts left passing patches.',
                 'Task time includes model startup and grading, so some bars exceed 20 minutes.'])
    card.save('bonsai-dsh-attempts.png')

    card = Card('Long-context throughput and correctness',
                'RTX 5070 Ti / Q4 KV / Flash Attention / no speculation / synthetic prompts',
                'CONTEXT', date='17 NOV 2026')
    card.text(54, 195, 'Fresh prefill (tok/s)', 23, MUTED)
    card.text(820, 195, 'Decode (tok/s)', 23, MUTED)
    for i, row in enumerate(data['context_speed']):
        y = 259 + 68*i
        color = GREEN if row['model'] == 'bonsai-pq2' else CYAN
        model = 'Bonsai' if color == GREEN else 'Ornith'
        label = f"{model} {row['capacity']//1024}K / {row['input_tokens']:,} input"
        for x, key, maximum in [(54, 'prefill_tps', 1400), (820, 'decode_tps', 110)]:
            value = row[key]
            card.bar(x, y, 560, value, maximum, color, label)
            card.text(x+576, y, f'{value:,.1f}', 21, color, True)
    card.text(54, 683, '260,000-token correctness / cold and cached', 24, WHITE, True)
    card.text(600, 724, 'Needle 10%', 20, MUTED)
    card.text(775, 724, 'Needle 50%', 20, MUTED)
    card.text(950, 724, 'Needle 90%', 20, MUTED)
    card.text(1125, 724, 'Dependency', 20, MUTED)
    card.text(1300, 724, 'Tool turns', 20, MUTED)
    observed = {(row['model'], row['task']): row for row in data['context_correctness']}
    for i, model in enumerate(('bonsai-pq2', 'ornith-ud-iq3xxs')):
        y = 766 + i*50
        card.text(54, y, 'Bonsai PQ2' if i == 0 else 'Ornith UD-IQ3_XXS', 22)
        for index, task in enumerate(('needle-10', 'needle-50', 'needle-90', 'dependency', 'tool-turns')):
            row = observed.get((model, task))
            if not row:
                value, color = ('Interrupted' if task == 'needle-50' else 'No result'), MUTED
            elif task == 'tool-turns':
                turns = row['tool_turns']
                value, color = f"{sum(turn['passed'] for turn in turns)}/{len(turns)} passed", GREEN
            else:
                value = '/'.join('Pass' if row[mode]['passed'] else 'Fail' for mode in ('cold', 'cached'))
                color = GREEN if value == 'Pass/Pass' else AMBER
            card.text(600+index*175, y, value, 19, color)
    card.footer(['One sample per speed setting or correctness case. Tool turns use simulated responses.',
                 'Bonsai dependency trace returned 103 (expected 100). Ornith stopped during the 50% needle.'])
    card.save('bonsai-ornith-context.png')


def strata_iq3s_prefill_card():
    data = load('strata-iq3s-128k-prefill-20261117.json')
    large, small = data['runs']
    ratio = large['cold_prefill_tps'] / small['cold_prefill_tps']
    card = Card(data['model'], '128K capacity / RTX 5070 Ti 16 GB + 64 GB RAM / Strata 0.1.60',
                'SPEED', '17 NOV 2026')
    card.metric(0, 'COLD PREFILL / 4K', f'{large["cold_prefill_tps"]:,.1f} tok/s', 'Fastest measured')
    card.metric(1, 'COLD PREFILL / 1K', f'{small["cold_prefill_tps"]:,.1f} tok/s', 'All experts resident', CYAN)
    card.metric(2, 'PREFILL SPEEDUP / 4K', f'{ratio:.2f}x', 'Same synthetic prompt')
    card.metric(3, 'ACTUAL INPUT', f'{data["input_tokens"]:,}', '128 output tokens', CYAN)
    card.rect((52, 365, 1548, 575), outline=BORDER)
    card.text(74, 383, 'COLD PREFILL / tok/s (0 to 3,000)', 26, WHITE, True)
    for run, color, y in ((large, GREEN, 461), (small, CYAN, 532)):
        label = f'{run["chunk_tokens"]:,}-token chunks'
        card.bar(76, y, 1210, run['cold_prefill_tps'], 3000, color, label)
        card.text(1315, y-2, f'{run["cold_prefill_tps"]:,.1f}', 25, color, True)

    card.rect((52, 603, 1548, 886), outline=BORDER)
    columns = (76, 404, 669, 963, 1220)
    labels = ('PREFILL CHUNK', 'COLD TOTAL', 'COLD DECODE', 'CACHED DECODE', 'SSD EXPERT READS')
    for x, label in zip(columns, labels):
        card.text(x, 630, label, 20, MUTED)
    for run, color, y in ((large, GREEN, 692), (small, CYAN, 754)):
        values = (f'{run["chunk_tokens"]:,} tokens', f'{run["cold_total_s"]:.1f} s',
                  f'{run["cold_decode_tps"]:.1f} tok/s', f'{run["cached_decode_tps"]:.1f} tok/s',
                  str(run['expert_file_reads']))
        for x, value in zip(columns, values):
            card.text(x, y, value, 25, color, True)
    card.text(76, 828, 'Both: INT8 KV / MTP4 / expert-tail skip 7 / ~14 GiB peak device VRAM', 22, MUTED)
    card.footer(['One cold request and one replay per setup. Synthetic speed test.',
                 '1K covered all 486 borrowed GPU slots in RAM. The PLE lookup table stays SSD-backed.'])
    card.save('strata-iq3s-128k-prefill.png')


def sensenova_phase2_cards():
    evidence = ROOT / 'bench/sensenova/evidence/quality-256k-phase2/results.json'
    result = json.loads(evidence.read_text(encoding='utf-8'))
    summary, tests = result['summary'], result['tests']
    subtitle = 'SenseNova-Loop-9B Q4_K_M / RTX 5070 Ti / 256K / Q8 KV / 1,024-token microbatch'
    date = '18 NOV 2026'

    quality = Card('256K multi-hop reasoning', subtitle, 'QUALITY', date)
    quality.metric(0, 'STRICT JSON', f'{summary["correct"]}/{len(tests)}', 'Standalone JSON', AMBER)
    quality.metric(1, 'CONTENT CORRECT', f'{summary["manual_content_review"]["correct"]}/{len(tests)}',
                   'One JSON fence allowed', CYAN)
    quality.metric(2, 'CONTEXT UTILIZATION', f'{tests[0]["context_utilization_percent"]:.2f}%',
                   f'{result["context_capacity"]:,}-token allocation')
    quality.metric(3, 'MAX ACTUAL INPUT', f'{max(test["input_tokens"] for test in tests):,}',
                   'Four synthetic questions', CYAN)
    quality.rect((52, 360, 1548, 886), outline=BORDER)
    for x, title in [(76, 'REASONING CASE'), (330, 'STRICT'), (470, 'CONTENT'), (640, 'OBSERVED ERROR')]:
        quality.text(x, 382, title, 22, MUTED)
    errors = {
        'temporal': 'Chose a policy signed after activation and missed the retention delay.',
        'dependency': 'Selected the staging ABI chain for a production deployment.',
        'overrides': 'Kept the regional endpoint despite a higher-precedence tenant override.',
        'synthesis': 'Returned every field correctly inside a Markdown code fence.',
    }
    for i, test in enumerate(tests):
        y = 439 + i*107
        category = test['category']
        quality.rect((67, y-11, 1533, y+64), fill=BG)
        quality.text(76, y+8, category.title(), 25, WHITE, True)
        for x, passed in [(330, test['correct']), (470, test['content_review']['correct'])]:
            quality.text(x, y+8, 'PASS' if passed else 'FAIL', 24, GREEN if passed else AMBER, True)
        quality.text(640, y+9, errors[category], 20, WHITE)
    quality.footer(['Phase 1 retrieval scored 6/6. Phase 2 used one test per category and one shared synthetic archive.',
                    'Greedy decoding / seed 42 / thinking disabled.'])
    quality.save('sensenova-phase2-quality.png')

    cold, cached = (summary['by_cache_mode'][mode] for mode in ('cold', 'cached'))
    timing = Card('256K multi-hop timing', subtitle, 'TIMING', date)
    timing.metric(0, 'COLD TTFT', f'{cold["median_ttft_s"]:.2f} s', '1 request')
    timing.metric(1, 'CACHED TTFT', f'{cached["median_ttft_s"]:.3f} s', 'Median of 3 requests', CYAN)
    timing.metric(2, 'CACHED FRESH INPUT', f'{cached["median_fresh_tokens"]:,} tokens', 'Median of 3 requests', CYAN)
    peak = max(test['peak_device_vram_mib'] for test in tests)
    timing.metric(3, 'PEAK DEVICE VRAM', f'{peak/1024:.2f} GiB', 'Whole GPU, sampled')
    timing.rect((52, 360, 1548, 886), outline=BORDER)
    columns = (76, 284, 433, 590, 724, 1134, 1337)
    headers = ('CASE', 'ACTUAL INPUT', 'REUSED', 'TTFT / s', 'FRESH PREFILL / tok/s', 'DECODE / tok/s', 'VRAM / GiB')
    for x, title in zip(columns, headers):
        timing.text(x, 382, title, 19, MUTED)
    for i, test in enumerate(tests):
        y = 444 + i*100
        color = GREEN if test['mode'] == 'cold' else CYAN
        values = (test['category'].title(), f'{test["input_tokens"]:,}', f'{test["cached_tokens"]:,}',
                  f'{test["ttft_s"]:.3f}', f'{test["prefill_tok_s"]:,.1f}',
                  f'{test["decode_tok_s"]:.1f}', f'{test["peak_device_vram_mib"]/1024:.2f}')
        for x, value in zip(columns, values):
            timing.text(x, y, value, 23, color, True)
        timing.bar(724, y+39, 350, test['prefill_tok_s'], 1200, color, '')
    timing.text(724, 846, 'Bar scale 0 to 1,200 tok/s', 18, MUTED)
    timing.footer(['Temporal ran cold. The other three requests reused the shared prefix.',
                   'Prefill counts fresh tokens. Cold and cached timings come from different questions.'])
    timing.save('sensenova-phase2-timing.png')


def swift_qwen_cards():
    data = load('swift-qwen-20261118.json')
    swift, original = (data['agent'][key] for key in ('swift-iq3s', 'qwen38-iq3s'))
    card = Card('Swift 1.5 Qwen3.8-27B / Coding',
                'IQ3_S / RTX 5070 Ti / 64K Q4 KV / xhigh / unrestricted reasoning / MTP off',
                'CODING', '18 NOV 2026')
    card.metric(0, 'SWIFT STRICT', f'{swift["strict"]}/{swift["tasks"]}', 'Held-out patch + clean finish')
    card.metric(1, 'ORIGINAL STRICT', f'{original["strict"]}/{original["tasks"]}', 'Matched OpenCode settings', CYAN)
    card.metric(2, 'TASK TIME REDUCTION', f'{1-swift["seconds"]/original["seconds"]:.1%}', 'Across four attempts')
    card.metric(3, 'GENERATED TOKEN REDUCTION', f'{1-swift["generated"]/original["generated"]:.1%}',
                'Native output incl. reasoning', CYAN)
    card.rect((52, 360, 1548, 892), outline=BORDER)
    for x, label in [(76, 'TOTAL TASK TIME / MIN'), (800, 'HELD-OUT'), (1010, 'GENERATED'), (1285, 'TOOL CALLS')]:
        card.text(x, 385, label, 23, MUTED)
    for i, (label, row, color) in enumerate((('Swift 1.5', swift, GREEN), ('Original Qwen3.8', original, CYAN))):
        y = 485 + 134*i
        minutes = row['seconds']/60
        card.bar(76, y, 520, minutes, 60, color, label)
        for x, value in [(614, f'{minutes:.2f}'), (800, f'{row["patch"]}/{row["tasks"]}'),
                         (1010, f'{row["generated"]:,}'), (1285, row['tools'])]:
            card.text(x, y-5, value, 30, color, True)
    card.text(76, 689, 'Time scale 0 to 60 min', 20, MUTED)
    card.text(76, 752, 'Sphinx #10323', 24, WHITE, True)
    sphinx = next(case for case in swift['results'] if case['task'] == 'sphinx-doc__sphinx-10323')
    card.text(76, 802, f'Swift passed in {sphinx["seconds"]/60:.2f} min. Original Qwen timed out and failed grading.', 25)
    card.footer(['Pytest #6202 x3 and Sphinx #10323 x1 per model. Same grader and 16K total output allowance.',
                 'Strict requires a passing held-out patch and a clean agent finish.'])
    card.save('swift-qwen-coding.png')

    profiles = {(row['context'], row['kv'], row['mtp_depth'], row.get('batch', 512), row.get('microbatch', 128)): row
                for row in data['speed'] if row['status'] == 'measured'}
    selections = [
        ('64K Q4 / MTP off', (65536, 'q4_0', 0, 512, 128)),
        ('64K Q8 / MTP off', (65536, 'q8_0', 0, 512, 128)),
        ('92K Q4 / MTP off', (92160, 'q4_0', 0, 512, 128)),
        ('64K Q4 / MTP3', (65536, 'q4_0', 3, 128, 32)),
        ('32K Q4 / MTP4', (32768, 'q4_0', 4, 128, 32)),
    ]
    rows = [(label, profiles[key]) for label, key in selections]
    card = Card('Swift 1.5 Qwen3.8-27B / Inference',
                'GSQ-RCO IQ3_S MTP checkpoint / RTX 5070 Ti / filled synthetic retrieval',
                'SPEED', '18 NOV 2026')
    card.metric(0, '64K COLD PREFILL', f'{rows[0][1]["requests"]["cold"]["timings"]["prompt_per_second"]:,.0f} tok/s',
                'Q4 KV, MTP off')
    card.metric(1, '64K CACHED TTFT', f'{rows[0][1]["requests"]["cached"]["ttft_seconds"]*1000:.0f} ms',
                '64,508 input tokens reused', CYAN)
    card.metric(2, 'MAX VERIFIED INPUT', f'{rows[2][1]["requests"]["cold"]["actual_input_tokens"]:,}',
                '92,160 capacity, Q4 KV')
    card.metric(3, 'PEAK 32K DECODE', f'{rows[-1][1]["requests"]["cold"]["timings"]["predicted_per_second"]:.1f} tok/s',
                'MTP4, cold request', CYAN)
    card.rect((52, 360, 1548, 892), outline=BORDER)
    columns = (76, 380, 568, 755, 927, 1110)
    for x, label in zip(columns, ('PROFILE', 'ACTUAL INPUT', 'PREFILL / tok/s', 'DECODE / tok/s',
                                  'TTFT / s', 'PEAK VRAM / GiB')):
        card.text(x, 385, label, 19, MUTED)
    for i, (label, row) in enumerate(rows):
        request = row['requests']['cold']
        timings = request['timings']
        y = 456 + 75*i
        values = (label, f'{request["actual_input_tokens"]:,}', f'{timings["prompt_per_second"]:,.0f}',
                  f'{timings["predicted_per_second"]:.1f}', f'{request["ttft_seconds"]:.2f}',
                  f'{request["resources"]["device_vram_peak_sampled_bytes"]/2**30:.2f}')
        for x, value in zip(columns, values):
            card.text(x, y, value, 23, CYAN if row['mtp_depth'] else GREEN, x == columns[0])
        card.bar(1330, y+2, 175, timings['predicted_per_second'], 100, CYAN if row['mtp_depth'] else GREEN, '')
    card.text(76, 846, 'Cold rates shown. Decode bars use a 0 to 100 tok/s scale.', 19, MUTED)
    card.footer(['MTP rows use a smaller batch and Q4 draft KV. All listed profiles passed exact retrieval.',
                 '128K stopped before generation at the VRAM guard. VRAM is sampled device-wide.'])
    card.save('swift-qwen-inference.png')


def swift_2080ti_cards():
    data = load('swift-2080ti-overview-20261119.json')
    profiles = data['filled']['profiles']
    subtitle = 'Swift 1.5 Qwen3.8-27B GSQ-RCO IQ3_S / full CUDA / Q4 target + draft KV / xhigh'
    card = Card('RTX 2080 Ti 22GB / Filled MTP', subtitle, 'MTP', '19 NOV 2026')
    maximum = max(row['input'] for row in profiles)
    checks = sum(row['needle_passes'] for row in profiles)
    total = sum(row['needle_checks'] for row in profiles)
    fastest = max(row['sustained'] for row in profiles)
    card.metric(0, 'MAX ACTUAL INPUT', f'{maximum:,}', '262,144 total capacity', CYAN)
    card.metric(1, 'SUSTAINED DECODE', f'{fastest:.2f} tok/s', '128K / MTP3')
    card.metric(2, 'RETRIEVAL FIELDS', f'{checks}/{total}', 'Cold + cached / 3 records')
    card.metric(3, '256K / MTP3', '19.31 tok/s', '19.21 GiB peak GPU', CYAN)
    card.rect((52, 354, 1548, 902), outline=BORDER)
    columns = (76, 238, 335, 540, 738, 948, 1157)
    for x, label in zip(columns, ('CAPACITY', 'MTP', 'ACTUAL INPUT', 'PREFILL tok/s', 'DECODE tok/s', 'VRAM GiB', 'RETRIEVAL')):
        card.text(x, 376, label, 20, MUTED)
    for i, row in enumerate(profiles):
        y = 424 + i*49
        color = GREEN if row['depth'] == 3 else CYAN if row['depth'] == 5 else WHITE
        values = (f'{row["context"]//1024}K', 'off' if not row['depth'] else row['depth'],
                  f'{row["input"]:,}', f'{row["prefill"]:.1f}', f'{row["sustained"]:.2f}',
                  f'{row["peak"]/2**30:.2f}', '3/3 + 3/3' if row['cold_correct'] and row['cached_correct'] else 'FAIL')
        for x, value in zip(columns, values):
            card.text(x, y, value, 22, color, row['depth'] == 3)
        card.bar(1332, y+2, 175, row['sustained'], 30, color, '')
    card.text(76, 875, 'Decode bars 0 to 30 tok/s / fixed 512-token generation', 18, MUTED)
    card.footer(['One cold request + replay per profile. Same input IDs across depths. Sampled device-wide VRAM.',
                 'MTP3 is the balanced choice. MTP5 at 256K was 1.1% faster and used 300 MiB more.',
                 'Retrieval screening uses synthetic archive text. MTP coding quality remains unmeasured.'])
    card.save('swift-2080ti-filled-mtp.png')

    headroom = data['headroom']['summary']
    cold, cached = headroom['cold'], headroom['cached']
    card = Card('RTX 2080 Ti 22GB / Output Headroom', subtitle, 'OUTPUT', '19 NOV 2026')
    card.metric(0, 'TOTAL CAPACITY', f'{headroom["capacity"]:,}', '131,072 actual input', CYAN)
    card.metric(1, 'OUTPUT ALLOWANCE', f'{headroom["total_output_allowance"]:,}', 'Reasoning + final content')
    card.metric(2, 'NATURAL OUTPUT', f'{cold["output_tokens"]:,}', 'EOS respected / thinking ON')
    card.metric(3, 'DECODE', f'{cold["timings"]["predicted_per_second"]:.2f} tok/s',
                'Q4 KV / MTP3', CYAN)
    card.rect((52, 354, 1548, 590), outline=BORDER)
    for x, label in zip((76, 375, 685, 985, 1265), ('REQUEST', 'PREFILL tok/s', 'TTFT', 'GENERATED', 'WALL TIME')):
        card.text(x, 377, label, 22, MUTED)
    for i, (label, request, color) in enumerate((('Cold', cold, GREEN), ('Cached', cached, CYAN))):
        y = 439 + i*68
        values = (label, f'{request["timings"]["prompt_per_second"]:.2f}',
                  f'{request["ttft_seconds"]:.3f} s', f'{request["output_tokens"]:,}',
                  f'{request["wall_seconds"]/60:.2f} min')
        for x, value in zip((76, 375, 685, 985, 1265), values):
            card.text(x, y, value, 27, color, True)
    card.rect((52, 615, 776, 895), outline=BORDER)
    card.text(76, 638, 'FIXED KV RESERVATION', 24, WHITE, True)
    card.text(76, 690, '4,608 MiB main + 288 MiB draft', 27, CYAN, True)
    card.text(76, 744, 'VRAM after long generation / 19.21 GiB', 23)
    card.text(76, 792, 'Startup -> after generation / +244 MiB', 22, MUTED)
    card.text(76, 835, 'KV allocated at context initialization', 21, MUTED)
    card.rect((800, 615, 1548, 895), outline=BORDER)
    card.text(824, 638, 'REASONING ACROSS TOOLS', 24, WHITE, True)
    card.text(824, 690, '2 tool calls / 3 thinking turns', 27, GREEN, True)
    card.text(824, 744, 'Prior traces retained / 1 of 1, then 2 of 2', 22)
    card.text(824, 792, 'Final archive + tool-result recall / PASS', 22, GREEN)
    card.text(824, 835, 'Large chat max_tokens accepted / 130,559', 21, MUTED)
    card.footer(['30,154 output tokens include reasoning. Cold and cached outputs had identical token IDs.',
                 'One controlled tool conversation. New-user-query and compaction retention remain untested.',
                 'Cached prefill is four fresh tokens. Original 128K Q8 server was restored on port 8080.'])
    card.save('swift-2080ti-headroom.png')

    homelab, reference = data['coding'], data['reference_5070ti']
    card = Card('RTX 2080 Ti 22GB / Coding Comparison',
                'Swift 1.5 Qwen3.8-27B IQ3_S / OpenCode 1.18.16 / 64K Q4 KV / xhigh / MTP off',
                'CODING', '19 NOV 2026')
    card.metric(0, '2080 Ti STRICT', f'{homelab["strict"]}/{homelab["graded"]}', '3 held-out patch passes', CYAN)
    card.metric(1, '5070 Ti STRICT', f'{reference["strict"]}/{reference["tasks"]}', '4 held-out patch passes')
    card.metric(2, '2080 Ti TASK TIME', f'{homelab["task_seconds"]/60:.2f} min', f'{homelab["tools"]} tool calls', CYAN)
    card.metric(3, '5070 Ti TASK TIME', f'{reference["seconds"]/60:.2f} min', f'{reference["tools"]} tool calls')
    card.rect((52, 354, 1548, 635), outline=BORDER)
    for i, (label, seconds, color) in enumerate((('RTX 2080 Ti 22GB', homelab['task_seconds'], CYAN),
                                               ('RTX 5070 Ti 16GB', reference['seconds'], GREEN))):
        y = 445+i*100
        card.bar(76, y, 1130, seconds/60, 60, color, label)
        card.text(1250, y-7, f'{seconds/60:.2f} min', 32, color, True)
    card.text(76, 601, 'Scale 0 to 60 minutes / sum of four attempts', 19, MUTED)
    card.rect((52, 659, 1548, 895), outline=BORDER)
    for x, label in ((76, 'ISSUE'), (770, '2080 Ti'), (1160, '5070 Ti')):
        card.text(x, 682, label, 22, MUTED)
    card.text(76, 738, 'Pytest #6202 / three fresh attempts', 25)
    card.text(770, 738, '3/3 strict', 25, GREEN)
    card.text(1160, 738, '3/3 strict', 25, GREEN)
    card.text(76, 805, 'Sphinx #10323 / one attempt', 25)
    card.text(770, 805, '20m timeout', 25, AMBER)
    card.text(1160, 805, '18.27m pass', 25, GREEN)
    card.footer(['Same frozen images, grader, guidance, 20-minute agent limit and 16K total output allowance.',
                 'Sphinx cleanup failed on Linux. Its retained patch failed offline grading.',
                 'Remaining attempts used corrected process cleanup. Runtime builds and host CPUs differ.'])
    card.save('swift-2080ti-coding.png')


def inkling_turing_cards():
    data = load('inkling-turing-20261122.json')
    runs = {run['id']: run for run in data['runs']}
    r128, r185, r242, r128b = runs['ub512-128k'], runs['ub512-185k'], runs['ub256-242k'], runs['ub256-128k']
    head = data['headline']
    sub = 'Inkling 2 Mini UD-IQ3_XXS / RTX 2080 Ti 22GB 175 W / Q8_0 KV / runtime 7d3e5b1 / research engine'
    date = '22 NOV 2026'
    label = {'ub512-128k': '128K / ub512', 'ub512-185k': '185K / ub512', 'ub256-128k': '128K / ub256', 'ub256-242k': '242K / ub256'}
    order = ('ub512-128k', 'ub512-185k', 'ub256-128k', 'ub256-242k')

    def short(run):
        return f'{run["snapshot_tokens"]:,}'

    # 1 overview
    card = Card('Inkling-Turing / Long-Context Overview', sub, 'INKLING 1', date)
    card.metric(0, 'MAX ACTUAL INPUT', f'{head["max_snapshot_tokens"]:,}', 'repository snapshot tokens', CYAN)
    card.metric(1, 'CONFIGURED CAPACITY', f'{head["configured_capacity"]:,}', 'allocated, not input', WHITE)
    card.metric(2, 'HIDDEN TESTS', head['hidden_tests'], 'two-file fix / rules followed')
    card.metric(3, 'FOLLOW-UP CHECKS', head['followup_checks'], 'retrieval / recall / rules', CYAN)
    card.rect((52, 354, 1548, 640), outline=BORDER)
    for x, text in ((76, 'RUN'), (330, 'SNAPSHOT'), (520, 'CONTEXT AFTER'), (760, 'COLD PREFILL'), (1010, 'DECODE'), (1220, 'PEAK VRAM')):
        card.text(x, 374, text, 20, MUTED)
    for i, key in enumerate(order):
        run = runs[key]
        y = 418 + i*54
        color = GREEN if key == 'ub256-242k' else CYAN if key == 'ub512-185k' else WHITE
        vals = (label[key], short(run), f'{run["final_context_tokens"]:,}', f'{run["cold_prefill"]["seconds"]:.1f} s',
                f'{run["decode_tok_per_s"]["fix_turn"]:.1f} tok/s', f'{run["vram_mib"]["peak_polled"]:,} MiB')
        for x, value in zip((76, 330, 520, 760, 1010, 1220), vals):
            card.text(x, y, value, 23, color, key == 'ub256-242k')
    card.rect((52, 664, 776, 902), outline=BORDER)
    card.text(76, 686, 'RUNTIME', 24, WHITE, True)
    card.text(76, 730, 'Validated baseline  7d3e5b1', 24, GREEN, True)
    card.text(76, 774, 'Fallbacks  e4a7ea6 / e0be55a', 22)
    card.text(76, 814, 'GPU-side relative positions, Q8_0 KV', 22)
    card.text(76, 854, 'No CPU offload / bit-exact default', 22, MUTED)
    card.rect((800, 664, 1548, 902), outline=BORDER)
    card.text(824, 686, 'STATUS', 24, WHITE, True)
    card.text(824, 730, 'Separately validated research engine', 24, CYAN, True)
    card.text(824, 774, 'Swift 1.5 stays the production backend', 22)
    card.text(824, 814, f'{head["scheduler_replans"]} scheduler re-plans in the 242K session', 22, GREEN)
    card.text(824, 854, 'Nothing merged upstream', 22, MUTED)
    card.footer([f'Actual input is {r242["snapshot_tokens"]:,} snapshot tokens ({r242["first_request_prompt_tokens"]:,} in the first request, {r242["final_context_tokens"]:,} after tool use), never 256K.',
                 'Controlled synthetic-repository qualification, one run per size, not SWE-bench and not a reliability estimate.',
                 'Runs differ in ubatch and capacity: ub512 at 204,800, ub256 at 262,144. See the report for provenance.'])
    card.save('inkling-turing-overview.png')

    # 2 decode
    card = Card('Inkling-Turing / Long-Context Decode', sub, 'INKLING 2', date)
    for i, key in enumerate(('ub512-128k', 'ub512-185k', 'ub256-242k', 'ub256-128k')):
        run = runs[key]
        card.metric(i, label[key].upper(), f'{run["decode_tok_per_s"]["fix_turn"]:.1f} tok/s', f'{run["snapshot_tokens"]:,} tokens',
                    GREEN if key == 'ub256-242k' else CYAN)
    card.rect((52, 354, 1548, 760), outline=BORDER)
    for x, text in ((76, 'RUN'), (300, 'SNAPSHOT'), (470, 'CONFIG'), (700, 'CONTEXT DURING FIX TURN')):
        card.text(x, 376, text, 20, MUTED)
    for i, key in enumerate(order):
        run = runs[key]
        y = 440 + i*78
        color = GREEN if key == 'ub256-242k' else CYAN if key.startswith('ub512') else WHITE
        cfg = run['configuration']
        card.text(76, y, label[key], 23, color, True)
        card.text(300, y, short(run), 23, color)
        card.text(470, y, f'cap {cfg["context_capacity"]:,} Q8_0', 21, MUTED)
        card.text(700, y, f'{run["first_request_prompt_tokens"]:,} -> {run["fix_turn"]["context_tokens_after"]:,}', 21, MUTED, width=330)
        card.bar(1050, y+2, 330, run['decode_tok_per_s']['fix_turn'], 40, color, '')
        card.text(1400, y, f'{run["decode_tok_per_s"]["fix_turn"]:.1f}', 23, color, True)
    card.text(76, 735, 'Bars 0 to 40 tok/s / mean decode rate over the fix-turn generations', 18, MUTED)
    card.rect((52, 784, 1548, 902), outline=BORDER)
    card.text(76, 802, f'Decode falls from {r128["decode_tok_per_s"]["fix_turn"]:.1f} to {r242["decode_tok_per_s"]["fix_turn"]:.1f} tok/s as the snapshot grows from {r128["snapshot_tokens"]//1000}K to {r242["snapshot_tokens"]//1000}K tokens.', 20, WHITE, width=1440)
    card.text(76, 836, 'Rates come from the agent session (thinking on, tools, temperature 0), not from a llama-bench sweep.', 20, MUTED, width=1440)
    card.text(76, 866, f'At 128K the ub512 run measured {r128["decode_tok_per_s"]["fix_turn"]:.1f} and the ub256 run {r128b["decode_tok_per_s"]["fix_turn"]:.1f} tok/s.', 20, MUTED, width=1440)
    card.footer(['Decode is measured on actual tokens in context at the stated size; capacity is only the allocation.',
                 'One successful session per size. Run-to-run variance was not measured.',
                 f'Different seeds and ubatch across runs; 128K decode differs by {abs(r128b["decode_tok_per_s"]["fix_turn"]/r128["decode_tok_per_s"]["fix_turn"]-1)*100:.1f} percent between ub512 and ub256.'])
    card.save('inkling-turing-decode.png')

    # 3 prefill
    card = Card('Inkling-Turing / Cold Prefill vs Context', sub, 'INKLING 3', date)
    for i, key in enumerate(('ub512-128k', 'ub512-185k', 'ub256-128k', 'ub256-242k')):
        run = runs[key]
        card.metric(i, label[key].upper(), f'{run["cold_prefill"]["tok_per_s"]:.0f} tok/s', f'{run["cold_prefill"]["seconds"]:.1f} s cold prefill',
                    GREEN if key == 'ub256-242k' else CYAN if key.startswith('ub512') else WHITE)
    card.rect((52, 354, 1548, 700), outline=BORDER)
    card.text(76, 370, 'COLD PREFILL THROUGHPUT / tok/s (bars 0 to 800)', 20, MUTED)
    for i, key in enumerate(order):
        run = runs[key]
        y = 440 + i*62
        color = CYAN if key.startswith('ub512') else GREEN
        card.text(76, y-6, label[key], 22, color, True)
        card.bar(300, y, 900, run['cold_prefill']['tok_per_s'], 800, color, '')
        card.text(1230, y-6, f'{run["cold_prefill"]["tok_per_s"]:.0f} tok/s', 22, color, True)
        card.text(1400, y-6, f'{run["cold_prefill"]["seconds"]:.0f} s', 22, WHITE)
    ratio = r128b['cold_prefill']['seconds'] / r128['cold_prefill']['seconds']
    card.rect((52, 724, 776, 902), outline=BORDER)
    card.text(76, 744, 'MICROBATCH EFFECT AT 128K', 22, WHITE, True)
    card.text(76, 788, f'ub256 takes {ratio:.2f}x the time of ub512', 24, AMBER, True)
    card.text(76, 832, f'{r128b["cold_prefill"]["seconds"]:.1f} s vs {r128["cold_prefill"]["seconds"]:.1f} s (derived)', 22)
    card.text(76, 868, f'decode {r128b["decode_tok_per_s"]["fix_turn"]:.1f} vs {r128["decode_tok_per_s"]["fix_turn"]:.1f} tok/s', 21, MUTED)
    card.rect((800, 724, 1548, 902), outline=BORDER)
    card.text(824, 744, 'WHY ub256 AT 256K', 22, WHITE, True)
    card.text(824, 788, 'ub512 does not load at 262,144', 24, AMBER, True)
    mem = data['memory']['ub512_at_262144']
    card.text(824, 832, f'cudaMalloc of {mem["failed_allocation_mib"]/1024:.2f} GiB failed', 22)
    card.text(824, 868, 'ub512 loads at 204,800 capacity', 21, MUTED)
    card.footer(['Cyan = ub512 at 204,800 capacity. Green = ub256 at 262,144 capacity. Cold means an empty prefix cache.',
                 'Throughput falls with depth: ' + ', '.join(f'{runs[k]["cold_prefill"]["tok_per_s"]:.0f}' for k in order) + ' tok/s in the order listed above.',
                 'Tokens processed in the first request include system prompt and tool schema (see JSON).'])
    card.save('inkling-turing-prefill.png')

    # 4 intelligence
    card = Card('Inkling-Turing / Context Intelligence', sub, 'INKLING 4', date)
    card.metric(0, 'HIDDEN TESTS', head['hidden_tests'], 'all four runs passed', GREEN)
    card.metric(1, 'RETRIEVAL + RULES', head['followup_checks'], 'retrieval / recall / rules', CYAN)
    card.metric(2, 'FILES FIXED', '2', 'distant bugs, one in legacy/', GREEN)
    card.metric(3, 'DECOYS IGNORED', 'yes', 'outdated docs, staging at 242K', CYAN)
    card.rect((52, 354, 1548, 700), outline=BORDER)
    for x, text in ((76, 'CHECK'), (610, '128K ub512'), (800, '185K ub512'), (1000, '128K ub256'), (1200, '242K ub256')):
        card.text(x, 374, text, 20, MUTED)
    names = (('fix', 'Two-file fix, 7 hidden tests'), ('rules', 'Current rules (# fixed, 4 spaces)'),
             ('retrieval', 'Distant value, decoys ignored'), ('recall', 'Recall first file after tools'), ('rules2', 'Contradictory rules resolved'))
    for i, (key, text) in enumerate(names):
        y = 420 + i*54
        card.text(76, y, text, 22)
        for x, run in zip((610, 800, 1000, 1200), (r128, r185, r128b, r242)):
            if key == 'fix':
                ok = run['fix_turn']['hidden_tests_failed'] == 0
                value = f'{run["fix_turn"]["hidden_tests_passed"]}/{run["fix_turn"]["hidden_tests_total"]}'
            elif key == 'rules':
                ok = run['fix_turn']['rules_ok']
                value = 'PASS' if ok else 'FAIL'
            else:
                follow = {f['name']: f for f in run['followups']}['rules' if key == 'rules2' else key]
                ok = follow['pass']
                value = 'PASS' if ok else 'FAIL'
            card.text(x, y, value, 23, GREEN if ok else AMBER, True)
    var = r242['randomized_variant']
    card.rect((52, 724, 1548, 902), outline=BORDER)
    card.text(76, 744, '242K RUN: FRESH RANDOMIZED CONTENT', 22, WHITE, True)
    card.text(76, 786, f'Current value {var["needle"]} found; decoys {var["decoy"]} (old docs) and {var["staging"]} (staging file) rejected', 21, CYAN, width=1440)
    card.text(76, 822, f'Service name {var["service"]} and seed {var["seed_hex"]} recalled from the first file after tool use', 21, WHITE, width=1440)
    card.text(76, 858, 'Current rules file chosen over the 2024 rules and a never-adopted draft', 21, WHITE, width=1440)
    card.footer(['Controlled synthetic-repository qualification, not a SWE-bench score and not a general coding benchmark.',
                 'One run per context size. Pass/fail here is not a statistical reliability measurement.',
                 'The ub512 runs used fixed needles (seed 31); the ub256 runs used fresh randomized needles (seed 4242).'])
    card.save('inkling-turing-intelligence.png')

    # 5 memory engineering
    host = data['host_cost_32k_prefill_ctx262144_ub256']
    mem = data['memory']
    card = Card('Inkling-Turing / Memory Engineering', sub, 'INKLING 5', date)
    card.metric(0, 'HOST UPLOAD / 32K', f'{host["before"]["uploaded_mb"]:,.0f} MB', f'down to {host["after"]["uploaded_mb"]:.1f} MB', GREEN)
    card.metric(1, 'CPU FILL / 32K', f'{host["before"]["cpu_fill_ms"]:,.0f} ms', f'down to {host["after"]["cpu_fill_ms"]:.1f} ms', CYAN)
    card.metric(2, '256K AT ub256', f'{mem["loaded_mib_7d3e5b1"]["262144:256:q8_0"]:,} MiB', 'loaded, Q8_0 KV', GREEN)
    card.metric(3, 'PEAK VRAM 242K', f'{head["peak_vram_mib"]:,} MiB', 'polled during the session', CYAN)
    card.rect((52, 354, 776, 902), outline=BORDER)
    card.text(76, 374, 'LOADED VRAM / 7d3e5b1 / MiB (bars 0 to 22,528)', 20, MUTED, width=680)
    specs = (('262144:256:q8_0', '256K ub256'), ('262144:128:q8_0', '256K ub128'), ('262144:64:q8_0', '256K ub64'),
             ('131072:512:q8_0', '128K ub512'), ('131072:256:q8_0', '128K ub256'))
    for i, (spec, text) in enumerate(specs):
        y = 450 + i*74
        value = mem['loaded_mib_7d3e5b1'][spec]
        card.text(76, y-30, f'{text}   {value:,}', 21, GREEN if i == 0 else WHITE, i == 0)
        card.bar(76, y, 650, value, 22528, GREEN if i == 0 else CYAN, '')
    card.text(76, 825, f'e4a7ea6 at 256K/ub256: {mem["loaded_mib_e4a7ea6_262144_ub256"]:,} MiB', 20, MUTED, width=680)
    card.text(76, 860, 'e0be55a (unfused chain): did not fit', 20, MUTED, width=680)
    card.rect((800, 354, 1548, 902), outline=BORDER)
    card.text(824, 374, 'WHAT CHANGED', 22, WHITE, True)
    card.text(824, 420, 'Relative-bias mask as one graph node', 21, GREEN, width=700)
    card.text(824, 452, 'no F32 chain intermediates planned', 20, MUTED, width=700)
    card.text(824, 494, 'GPU-side distances from cell and token', 21, GREEN, width=700)
    card.text(824, 526, 'positions, no CPU int32 matrix upload', 20, MUTED, width=700)
    card.text(824, 576, f'Compute buffer at 262,144 / ub256: {mem["compute_buffer_mib_ub256_262144"]:,.0f} MiB', 20, WHITE, width=700)
    card.text(824, 608, f'KV buffers {mem["kv_buffer_mib_ub256_262144"][0]:,.0f} + {mem["kv_buffer_mib_ub256_262144"][1]:,.0f} MiB (Q8_0)', 20, WHITE, width=700)
    card.text(824, 656, 'REJECTED: relbias inside Flash Attention', 21, AMBER, True, width=700)
    card.text(824, 690, 'unit tests bit-exact, model runs unstable:', 20, MUTED, width=700)
    card.text(824, 720, 'prefill and decode op sequences diverge', 20, MUTED, width=700)
    card.text(824, 750, 'at node 1131 (280 nodes differ vs 0)', 20, MUTED, width=700)
    card.text(824, 800, f'ub512 at 262,144 fails: {mem["ub512_at_262144"]["failed_allocation_mib"]/1024:.2f} GiB alloc', 20, WHITE, width=700)
    card.text(824, 840, 'Patches and graph_audit.py kept in repo', 20, CYAN, width=700)
    card.footer(['Host figures: instrumented builds, 32K-token prefill at 262,144 capacity, ub256, 320 calls. End-to-end gain about 1 percent.',
                 'The saving is host work and memory, not speed: the fill overlapped GPU work of the previous micro-batch.',
                 'Rejected experiments and the graph audit that explains them are published with the patches.'])
    card.save('inkling-turing-memory.png')

    # 6 cache and responsiveness
    card = Card('Inkling-Turing / Cache and Responsiveness', sub, 'INKLING 6', date)
    f242 = r242['followups']
    cold = r242['cold_prefill']['seconds']
    solve = r242['fix_turn']['seconds']
    card.metric(0, 'COLD PREFILL 242K', f'{cold/60:.1f} min', f'{cold:.1f} s / empty cache', AMBER)
    card.metric(1, 'CACHED FOLLOW-UP', f'{min(f["ttft_s"] for f in f242):.2f}-{max(f["ttft_s"] for f in f242):.2f} s', 'TTFT / 3 follow-up turns', GREEN)
    card.metric(2, 'WARM FIX TTFT', f'{r242["fix_turn"]["median_warm_ttft_s"]:.2f} s', 'median / tool-call steps', CYAN)
    card.metric(3, 'PREFIX REUSE', f'{f242[0]["cached_tokens"]:,}', f'of {f242[0]["context_tokens"]:,} tokens', GREEN)
    card.rect((52, 354, 1548, 640), outline=BORDER)
    card.text(76, 370, 'TIME TO SOLUTION AT 242K / seconds (bar 0 to 700)', 20, MUTED)
    card.text(76, 410, 'Cold prefill', 22, AMBER, True)
    card.bar(300, 412, 1000, cold, 700, AMBER, '')
    card.text(1330, 410, f'{cold:.1f} s', 22, AMBER, True)
    card.text(76, 470, 'Agent work', 22, CYAN, True)
    card.bar(300, 472, 1000, solve - cold, 700, CYAN, '')
    card.text(1330, 470, f'{solve - cold:.1f} s', 22, CYAN, True)
    card.text(76, 530, 'Follow-up turn', 22, GREEN, True)
    card.bar(300, 532, 1000, max(f242[0]['ttft_s'], 0.1), 700, GREEN, '')
    card.text(1330, 530, f'{f242[0]["ttft_s"]:.2f} s', 22, GREEN, True)
    card.text(76, 590, f'Fix turn {solve:.1f} s = cold prefill + {r242["fix_turn"]["model_calls"]} model calls / {r242["fix_turn"]["tool_calls"]} tools (agent work is derived)', 19, MUTED, width=1440)
    card.rect((52, 664, 1548, 902), outline=BORDER)
    for x, text in ((76, 'RUN'), (400, 'COLD PREFILL'), (700, 'CACHED TTFT'), (1000, 'WARM FIX TTFT'), (1300, 'FIX TURN')):
        card.text(x, 682, text, 20, MUTED)
    for i, key in enumerate(order):
        run = runs[key]
        y = 726 + i*42
        color = GREEN if key == 'ub256-242k' else WHITE
        tt = [f['ttft_s'] for f in run['followups']]
        vals = (label[key], f'{run["cold_prefill"]["seconds"]:.1f} s', f'{min(tt):.2f}-{max(tt):.2f} s',
                f'{run["fix_turn"]["median_warm_ttft_s"]:.2f} s', f'{run["fix_turn"]["seconds"]:.0f} s')
        for x, value in zip((76, 400, 700, 1000, 1300), vals):
            card.text(x, y, value, 21, color, key == 'ub256-242k')
    card.footer(['Cold: first request, empty prefix cache, the whole snapshot is processed once. Warm: later steps reuse it.',
                 f'Cold start at 242K is a one-time {cold/60:.1f} minute cost; cached follow-ups answer in about a second.',
                 'A cache eviction or server restart brings the cold cost back. One session per size.'])
    card.save('inkling-turing-cache.png')


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
    dense_dflash2_cards()
    dense_live_card()
    dense_supervision_card()
    qwen36_hardest_context_card()
    if (ROOT / 'docs/results/bonsai-dsh-256k-20261117.json').is_file():
        bonsai_cards()
    strata_iq3s_prefill_card()
    if (ROOT / 'bench/sensenova/evidence/quality-256k-phase2/results.json').is_file():
        sensenova_phase2_cards()
    if (ROOT / 'docs/results/swift-qwen-20261118.json').is_file():
        swift_qwen_cards()
    if (ROOT / 'docs/results/swift-2080ti-overview-20261119.json').is_file():
        swift_2080ti_cards()
    if (ROOT / 'docs/results/inkling-turing-20261122.json').is_file():
        inkling_turing_cards()
