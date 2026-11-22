"""Verify how each profile's thinking-effort setting reaches the model. Run ON the homelab against a gateway whose backend is reachable.

  python validate_reasoning_live.py --base http://127.0.0.1:18090 --backend http://127.0.0.1:18082 [--solve]

1. Template proof: for every advertised level, POST the same kwargs the gateway forwards to the native server's /apply-template and
   show the line the chat template renders (this is the strongest evidence that the value reaches the template, independent of sampling).
2. Gateway proof: a real request through the gateway; the X-Reasoning-Effort header shows what was forwarded.
3. --solve: one moderately hard question per level; reports generated tokens and reasoning size. That shows whether the level changes
   behaviour for this prompt; it is NOT a measurement of the model's internal reasoning effort and a single sample is noisy.
"""
import argparse
import json
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_profiles_live import call, apply, status, chat

PUZZLE = 'How many positive integers below 500 are divisible by 3 or 7 but not by both? Give the final number.'


def native(backend, path, body):
    request = urllib.request.Request(backend + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read()[:200].decode(errors='replace')


def rendered_line(prompt, spec):
    if spec['family'] == 'inkling':
        match = re.search(r'Thinking effort level: ([0-9.]+)', prompt)
        return match.group(0) if match else None
    match = re.search(r'Reasoning effort is set to (\w+)', prompt)
    return match.group(0) if match else ('(no reasoning instruction)' if '<think>' in prompt else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--backend', required=True)
    parser.add_argument('--solve', action='store_true')
    parser.add_argument('--profiles', default='swift,inkling-balanced')
    args = parser.parse_args()
    out = {}
    catalogue = {p['key']: p for p in status(args.base)['profiles']}
    for key in args.profiles.split(','):
        code, seconds, body = apply(args.base, key)
        assert code == 200, body
        profile = catalogue[key]
        spec = profile['reasoning']
        defaults = {'enable_thinking': True, 'preserve_thinking': True} if profile['family'] == 'swift' else {}
        rows = {}
        for level in spec['levels'] + (['max', 0.5] if profile['family'] == 'inkling' else []):
            kwargs = dict(defaults)
            if level == 'none' and profile['family'] == 'swift':kwargs['enable_thinking'] = False
            else:kwargs['reasoning_effort'] = level
            code2, rendered = native(args.backend, '/apply-template', dict(messages=[dict(role='user', content='hi')], chat_template_kwargs=kwargs))
            prompt = rendered.get('prompt', '') if isinstance(rendered, dict) else str(rendered)
            row = dict(template_status=code2, rendered=rendered_line(prompt, profile), tail=prompt[-60:].replace('\n', '\\n'))
            if code2 != 200:row['error'] = prompt[:160]
            rows[str(level)] = row
        # the default (no level sent) must render the profile default
        code3, rendered = native(args.backend, '/apply-template', dict(messages=[dict(role='user', content='hi')], chat_template_kwargs=defaults))
        rows['(template default)'] = dict(template_status=code3, rendered=rendered_line(rendered.get('prompt', '') if isinstance(rendered, dict) else '', profile))
        out[key] = dict(template=rows)
        if args.solve:
            out[key]['solve'] = {}
            for level in ('none', 'low', 'high' if profile['family'] == 'inkling' else 'medium', 'xhigh'):
                if level not in spec['levels']:continue
                r = chat(args.base, profile['model_id'], level, prompt=PUZZLE, max_tokens=3500)
                out[key]['solve'][level] = {k: r.get(k) for k in ('status', 'effort_header', 'completion_tokens', 'reasoning_chars', 'finish', 'seconds', 'error')}
        print(key, json.dumps(out[key], indent=1)[:3800], flush=True)
    apply(args.base, 'swift')
    Path('/tmp/eah-reasoning.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':main()
