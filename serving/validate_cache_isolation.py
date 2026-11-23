"""Client-isolation test for llama-server prompt caching (llama.cpp issue #27148: stale conversation state restored into an
unrelated request).  Launches one llama-server variant per configuration on the homelab and sends two independent "clients"
with unique markers, sequentially, interleaved (multi-turn) and concurrently.

Verdict per response: `own` = the response repeats the marker that was in *that* request; `foreign` = it contains a marker or a
distinctive document name belonging to a different client/request. A foreign hit is contamination; a missing own marker is
reported separately (wrong answer, possibly cache corruption). No payload is stored beyond marker bookkeeping and counts.

usage:  python validate_cache_isolation.py --config-name swift-cram8192 [--rounds 8] [--doc-tokens 2000] --out result.jsonl
        (see VARIANTS below; the server is started and stopped by this script, so the production gateway must be stopped first)
"""
import argparse
import glob
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.request

HOME = os.path.expanduser('~/llm')
PORT = 18090
SWIFT = dict(exe=f'{HOME}/llama.cpp/build/bin/llama-server', model=f'{HOME}/models/Swift-1.5-Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf',
             args='--ctx-size 65536 --parallel 1 --n-gpu-layers 99 --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --batch-size 512 '
                  '--ubatch-size 128 --threads 8 --fit off --spec-type draft-mtp --no-context-shift --ctx-checkpoints 4 --no-webui '
                  '--spec-draft-n-max 3 --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-ubatch 64 --jinja '
                  '--override-tensor ^token_embd\\.weight$=CUDA0 --spec-draft-override-tensor ^token_embd\\.weight$=CUDA0 --no-warmup')
OBL = dict(exe=f'{HOME}/llama.cpp/build/bin/llama-server', model=f'{HOME}/models/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf',
           args='--ctx-size 65536 --parallel 1 --n-gpu-layers 99 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 2048 '
                '--ubatch-size 512 --threads 8 --fit off --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6 '
                '--spec-draft-type-k q4_0 --spec-draft-type-v q4_0 --spec-draft-n-ubatch 64 --no-context-shift --ctx-checkpoints 4 '
                '--no-webui --jinja --no-warmup')
# name -> (base, extra flags replacing/adding; later flags win because they are appended)
VARIANTS = {
    'swift-prod-cram0': (SWIFT, '--cache-ram 0'),                                           # what production runs today
    'swift-cram8192': (SWIFT, '--cache-ram 8192'),
    'swift-cram8192-noidle': (SWIFT, '--cache-ram 8192 --no-cache-idle-slots'),
    'swift-cram0-noidle': (SWIFT, '--cache-ram 0 --no-cache-idle-slots'),
    'swift-np2-kvu-cram0-noidle': (SWIFT, '--cache-ram 0 --no-cache-idle-slots --parallel 2 --kv-unified'),
    'obl-cram0': (OBL, '--cache-ram 0'),
    'obl-cram8192': (OBL, '--cache-ram 8192'),
    'obl-cram8192-noidle': (OBL, '--cache-ram 8192 --no-cache-idle-slots'),
    'obl-cram0-noidle': (OBL, '--cache-ram 0 --no-cache-idle-slots'),
    'obl-np2-kvu-cram0-noidle': (OBL, '--cache-ram 0 --no-cache-idle-slots --parallel 2 --kv-unified'),
}


def post(path, body, timeout=900):
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}{path}', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(request, timeout=timeout))


class Corpus:
    """Two disjoint real-code corpora so 'foreign' vocabulary is unmistakable (file-name banner lines carry the signal)."""

    def __init__(self, rng):
        src = os.path.expanduser('~/llm/llama.cpp')
        self.docs = {}
        for name, pattern in (('A', f'{src}/ggml/src/ggml-cuda/*.cu'), ('B', f'{src}/src/*.cpp')):
            files = sorted(glob.glob(pattern))
            self.docs[name] = [(os.path.basename(f), open(f, errors='ignore').read()) for f in files]
        self.rng = rng

    def document(self, client, tokens):
        """~tokens of one client's corpus, starting at a random file, with a banner naming each file."""
        files = self.docs[client]
        start = self.rng.randrange(len(files))
        text, names = '', []
        for i in range(len(files)):
            name, body = files[(start + i) % len(files)]
            names.append(name)
            text += f'// ==== FILE {name} ====\n{body}\n'
            if len(text) > tokens * 3.6:break
        return text[:int(tokens * 3.6)], names


def marker(rng):return 'REF-' + ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(8))


def ask(content_messages, max_tokens=90, thinking=False):
    body = dict(model='x', messages=content_messages, max_tokens=max_tokens, temperature=0, top_k=1, cache_prompt=True,
                chat_template_kwargs=dict(enable_thinking=thinking))
    r = post('/v1/chat/completions', body)
    message = r['choices'][0]['message']
    text = (message.get('content') or '') + ' ' + (message.get('reasoning_content') or '')
    t = r.get('timings', {})
    return text, dict(prompt_tokens=r['usage']['prompt_tokens'], cache_n=t.get('cache_n'), prompt_n=t.get('prompt_n'),
                      prompt_ms=round(t.get('prompt_ms', 0)), predicted_n=r['usage']['completion_tokens'])


def user_turn(mark, doc, first=True):
    head = f'The reference code of this message is {mark}.\n\n' if first else ''
    return head + doc + '\n\nQuestion: state the reference code of this message exactly, then name the first source file mentioned above.'


class Judge:
    def __init__(self):
        self.marks = {}            # mark -> owner label
        self.files = {}            # file name -> set(client letters that legitimately contain it)
        self.rows = []
        self.lock = threading.Lock()

    def record(self, scenario, who, own, names, text, stats, extra=''):
        with self.lock:
            foreign_marks = [m for m, owner in self.marks.items() if owner != who and m in text]
            # a file name only counts as foreign if it exists in the other client's corpus and not in this request's own document
            foreign_files = [n for n in set().union(*[set(v) for v in self.files.values()]) if n in text and n not in names]
            row = dict(scenario=scenario, who=who, own=own in text, foreign_marks=foreign_marks, foreign_files=foreign_files[:3],
                       contaminated=bool(foreign_marks), **stats, note=extra)
            self.rows.append(row)
            return row


def run_rounds(rounds, doc_tokens, out, label, seed, long_generation=0):
    rng = random.Random(seed)
    corpus = Corpus(rng)
    judge = Judge()
    tokenizer_probe = post('/tokenize', dict(content='hello world'))
    assert tokenizer_probe.get('tokens')
    for n in range(rounds):
        docs, marks = {}, {}
        for client in 'AB':
            docs[client] = corpus.document(client, doc_tokens)
            marks[client] = marker(rng)
            judge.marks[marks[client]] = client
            judge.files[client + str(n)] = set(docs[client][1])
        msgs = {c: [dict(role='user', content=user_turn(marks[c], docs[c][0]))] for c in 'AB'}
        # S1: strictly sequential, unrelated, one user
        for c in 'AB':
            text, st = ask(msgs[c])
            judge.record('S1_sequential', c, marks[c], docs[c][1], text, st);msgs[c].append(dict(role='assistant', content=text.strip()))
        # S2: interleaved multi-turn: each follow-up extends its own conversation and must still know its own marker
        for c in 'AB':
            more, names = corpus.document(c, doc_tokens // 4)
            msgs[c].append(dict(role='user', content=f'Follow-up with new material:\n{more}\n\nQuestion: what was the reference code of the first message of this conversation?'))
            text, st = ask(msgs[c]);judge.record('S2_interleaved_followup', c, marks[c], docs[c][1] + names, text, st)
            msgs[c].append(dict(role='assistant', content=text.strip()))
        # S3: concurrent unrelated requests (two clients at once), fresh markers
        fresh = {}
        for c in 'AB':
            doc, names = corpus.document(c, doc_tokens);m = marker(rng);judge.marks[m] = c
            fresh[c] = (m, doc, names);judge.files[c + str(n) + 'c'] = set(names)
        results = {}

        def worker(c):
            results[c] = ask([dict(role='user', content=user_turn(fresh[c][0], fresh[c][1]))])
        threads = [threading.Thread(target=worker, args=(c,)) for c in 'AB']
        [t.start() for t in threads];[t.join() for t in threads]
        for c in 'AB':
            text, st = results[c];judge.record('S3_concurrent', c, fresh[c][0], fresh[c][2], text, st)
        # S4: A continues after a short unrelated side request (a title call) and after B's whole new conversation
        text, st = ask([dict(role='user', content='Write a title of three words for a chat about C++.')], 12)
        judge.record('S4_side_title', 'T', 'x', [], text, st)
        msgs['A'].append(dict(role='user', content='One more question: what was the reference code of the first message of this conversation?'))
        text, st = ask(msgs['A']);judge.record('S4_after_side', 'A', marks['A'], docs['A'][1] + [], text, st)
        # S5 (the issue's single-user repro): a short prompt with a LONG generation, then an unrelated long prompt, strictly sequential.
        if long_generation:
            script_marker = marker(rng);judge.marks[script_marker] = 'S'
            script, script_names = corpus.document('A', 400)
            text, st = ask([dict(role='user', content=f'Reference {script_marker}. Review this code in great detail, quote the reference, and list every possible bug:\n{script}')],
                           long_generation, thinking=True)
            judge.record('S5_long_generation', 'S', script_marker, script_names, text, st)
            other, other_names = corpus.document('B', doc_tokens);other_mark = marker(rng);judge.marks[other_mark] = 'B5'
            text, st = ask([dict(role='user', content=user_turn(other_mark, other))])
            judge.record('S5_unrelated_after_long', 'B5', other_mark, other_names, text, st)
        print(json.dumps(dict(round=n + 1, **{k: sum(1 for r in judge.rows if r['contaminated']) for k in ('contaminated_total',)},
                              own_missing=sum(1 for r in judge.rows if not r['own'] and r['who'] != 'T'))), flush=True)
    for row in judge.rows:open(out, 'a').write(json.dumps(dict(label=label, **row)) + '\n')
    total = [r for r in judge.rows if r['who'] != 'T']
    summary = dict(label=label, requests=len(judge.rows), contaminated=sum(1 for r in judge.rows if r['contaminated']),
                   own_missing=sum(1 for r in total if not r['own']), foreign_file_names=sum(1 for r in judge.rows if r['foreign_files'] and r['who'] != 'T'),
                   cache_hits=sum(1 for r in judge.rows if (r.get('cache_n') or 0) > 16),
                   cache_tokens=sum(r.get('cache_n') or 0 for r in judge.rows), prompt_tokens=sum(r.get('prompt_tokens') or 0 for r in judge.rows))
    open(out, 'a').write(json.dumps(dict(summary=summary)) + '\n')
    print(json.dumps(summary), flush=True)
    return summary


def launch(name):
    base, extra = VARIANTS[name]
    cmd = [base['exe'], '--model', base['model'], '--host', '127.0.0.1', '--port', str(PORT)] + base['args'].split() + extra.split()
    log = open(f'/tmp/iso_{name}.log', 'w')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_ARG_'))}
    env['LD_LIBRARY_PATH'] = os.path.dirname(base['exe']) + ':' + env.get('LD_LIBRARY_PATH', '')
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    for _ in range(180):
        if process.poll() is not None:raise SystemExit(f'server exited {process.returncode}')
        try:
            if b'ok' in urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2).read():return process, cmd
        except Exception:pass
        time.sleep(2)
    process.kill();raise SystemExit('server did not start')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', required=True, choices=sorted(VARIANTS))
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--doc-tokens', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--long-generation', type=int, default=0, help='add scenario S5: tokens of thinking-on generation before an unrelated long prompt')
    parser.add_argument('--out', default='isolation.jsonl')
    args = parser.parse_args()
    if subprocess.run('pgrep -x llama-server', shell=True, capture_output=True).returncode == 0:raise SystemExit('another llama-server is running')
    process, cmd = launch(args.config_name)
    try:
        open(args.out, 'a').write(json.dumps(dict(config=args.config_name, cmd=cmd, doc_tokens=args.doc_tokens, rounds=args.rounds)) + '\n')
        run_rounds(args.rounds, args.doc_tokens, args.out, args.config_name, args.seed, args.long_generation)
    finally:
        process.send_signal(signal.SIGTERM)
        try:process.wait(60)
        except subprocess.TimeoutExpired:process.kill()
