"""Does the RAM prompt cache restore a conversation after (a) a short side request and (b) an unrelated long prompt?

    python cache_restore_probe.py --variant NAME [--tokens 12000] [--unrelated 9000] --out probe.jsonl

Variants are llama-server launch lines built from the Obliterated profile (see PROFILE below) with one flag changed at a time, run with debug
logging so the server's own cache decisions are saved next to the result. The production gateway must be stopped (one GPU).
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_cache_isolation as iso   # noqa: E402  (reuses its server launcher, corpus and request helpers)

HOME = iso.HOME
TEMPLATE = os.path.expanduser('~/llm/experiments/obliterated-opt/staging/serving/templates/qwen35-tools.jinja')
PROFILE = (f'--ctx-size 65536 --parallel 1 --n-gpu-layers 99 --flash-attn on --cache-type-k {{kv}} --cache-type-v {{kv}} --batch-size 2048 --ubatch-size 512 '
           f'--threads 8 --fit off --no-context-shift --cache-ram {{ram}} --ctx-checkpoints 4 --no-webui --jinja --chat-template-file {TEMPLATE} --no-warmup '
           f'--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-ubatch 64 {{extra}}')


def variant(kv='f16', ram=8192, extra=''):
    return dict(exe=iso.OBL['exe'], model=iso.OBL['model'], args=PROFILE.format(kv=kv, ram=ram, extra=extra) + ' -lv 4')


VARIANTS = {
    'profile-f16-noidle': variant('f16', 8192, '--no-cache-idle-slots'),
    'profile-f16-idle': variant('f16', 8192, ''),
    'profile-q8-noidle': variant('q8_0', 8192, '--no-cache-idle-slots'),
    'profile-f16-noidle-ram16g': variant('f16', 16384, '--no-cache-idle-slots'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    parser.add_argument('--tokens', type=int, default=12000)
    parser.add_argument('--unrelated', type=int, default=9000)
    parser.add_argument('--history', action='store_true', help='replay qualification stages 3-5 first so the pool already holds earlier prompts')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    iso.VARIANTS[args.variant] = (VARIANTS[args.variant], '')
    process, cmd = iso.launch(args.variant)
    rng = random.Random(3)
    corpus = iso.Corpus(rng)
    rows = []

    def step(name, messages, max_tokens=60):
        began = time.time()
        text, stats = iso.ask(messages, max_tokens)
        rows.append(dict(step=name, **stats, seconds=round(time.time() - began, 1)))
        print(json.dumps(rows[-1]), flush=True)
        return text

    try:
        if args.history:
            for size in (4000, 4500, 5000, 16000):
                filler, _ = corpus.document('B', size)
                step(f'HISTORY_{size}', [dict(role='user', content=iso.user_turn(iso.marker(rng), filler))], 120)
        mark = iso.marker(rng)
        doc, _ = corpus.document('A', args.tokens)
        convo = [dict(role='user', content=iso.user_turn(mark, doc))]
        convo.append(dict(role='assistant', content=step('A1_cold', convo).strip()))
        convo.append(dict(role='user', content='Which reference code did I give? One line.'))
        step('A2_followup', convo)
        step('SIDE_title', [dict(role='user', content='Generate a 3-word title for a chat about C++ code review.')], 12)
        step('A3_after_side', convo)
        other, _ = corpus.document('B', args.unrelated)
        step('UNRELATED_long', [dict(role='user', content=iso.user_turn(iso.marker(rng), other))])
        step('A4_after_unrelated', convo)
        step('SIDE_title2', [dict(role='user', content='Generate a 3-word title for a chat about GPUs.')], 12)
        other2, _ = corpus.document('B', args.unrelated)
        step('UNRELATED_long2', [dict(role='user', content=iso.user_turn(iso.marker(rng), other2))])
        step('A5_after_second_unrelated', convo)
    finally:
        process.send_signal(signal.SIGTERM)
        try:process.wait(60)
        except subprocess.TimeoutExpired:process.kill()
        log = f'/tmp/iso_{args.variant}.log'
        keep = [line for line in open(log, errors='ignore') if any(t in line for t in ('prompt cache', 'cache state', 'TAG_', 'restor', 'saving', 'erasing', 'purging', 'selected slot'))]
        open(args.out + '.serverlog', 'w').write(''.join(keep[-400:]))
    with open(args.out, 'a') as handle:
        handle.write(json.dumps(dict(variant=args.variant, cmd=cmd, rows=rows)) + '\n')


if __name__ == '__main__':
    main()
