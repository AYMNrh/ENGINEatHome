"""Drive the REAL OpenCode CLI against a gateway profile and verify it can complete a small agentic coding task.

    python validate_opencode_client.py --base http://100.75.206.4:18095/v1 --model qwen3.8-27b-obliterated [--variant none] --out opencode-result.json

Builds a scratch project containing a planted bug and a test script, writes an isolated OpenCode config (the API key is taken from the
user's existing ~/.config/opencode/opencode.json and is never printed or stored in the result), runs `opencode run --auto` inside the
scratch directory only, and then re-runs the project's test to check the fix really works. Counts tool calls from the JSON event stream.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

PROJECT = {
    'calc.py': '''def average(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    total = 0
    for v in values:
        total += v
    return total / (len(values) + 1)


def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x
''',
    'test_calc.py': '''from calc import average, clamp

assert average([2, 4, 6]) == 4, average([2, 4, 6])
assert average([10]) == 10
assert clamp(5, 0, 3) == 3 and clamp(-1, 0, 3) == 0 and clamp(2, 0, 3) == 2
print("ALL TESTS PASSED")
''',
    'README.md': 'Tiny calculator module. Run `python test_calc.py`.\n',
}


def user_provider():
    """The user's own gateway key, from OpenCode's credential store (auth.json) or the provider options; never printed or stored in results."""
    options = json.loads((Path.home() / '.config/opencode/opencode.json').read_text())['provider']['engineathome'].get('options', {})
    key = options.get('apiKey')
    if key and not key.startswith('{env:'):return key
    auth = Path.home() / '.local/share/opencode/auth.json'
    return json.loads(auth.read_text())['engineathome']['key']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--variant', default=None)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    key = user_provider()
    with tempfile.TemporaryDirectory(prefix='opencode-qual-') as folder:
        root = Path(folder)
        project = root / 'project'
        project.mkdir()
        for name, text in PROJECT.items():(project / name).write_text(text)
        before = subprocess.run(['python', 'test_calc.py'], cwd=project, capture_output=True, text=True)
        config = dict(provider=dict(engineathome=dict(npm='@ai-sdk/openai-compatible', name='Homelab', options=dict(baseURL=args.base, apiKey=key, timeout=900000),
                                                      models={args.model: dict(name=args.model, reasoning=True, tool_call=True, limit=dict(context=65536, output=8192))})),
                      model='engineathome/' + args.model, autoupdate=False, share='disabled')
        (root / 'opencode.json').write_text(json.dumps(config))
        env = dict(os.environ, OPENCODE_CONFIG=str(root / 'opencode.json'), OPENCODE_CONFIG_DIR=str(root / 'cfg'),
                   XDG_DATA_HOME=str(root / 'data'), XDG_CACHE_HOME=str(root / 'cache'), XDG_STATE_HOME=str(root / 'state'))
        binary = Path(os.environ.get('APPDATA', '')) / 'npm/node_modules/opencode-ai/bin/opencode.exe'
        command = [str(binary) if binary.exists() else 'opencode', 'run', '-m', 'engineathome/' + args.model, '--format', 'json', '--auto', '--dir', str(project), '--pure']
        if args.variant:command += ['--variant', args.variant]
        command.append('The function average() in calc.py gives wrong results and test_calc.py fails. Find the bug, fix it, and run python test_calc.py to confirm. Keep changes minimal.')
        began = time.time()
        try:
            run = subprocess.run(command, cwd=project, env=env, capture_output=True, text=True, timeout=args.timeout)
            timed_out = False
        except subprocess.TimeoutExpired as error:
            run, timed_out = error, True
        seconds = round(time.time() - began, 1)
        stdout = (run.stdout.decode() if isinstance(run.stdout, bytes) else run.stdout) if getattr(run, 'stdout', None) else ''
        events, tools = [], []
        for line in stdout.splitlines():
            try:event = json.loads(line)
            except ValueError:continue
            events.append(event)
            kind = json.dumps(event)[:4000]
            if event.get('type') in ('tool_use', 'tool') or '"tool"' in kind and event.get('type', '').startswith('tool'):
                tools.append(event.get('part', {}).get('tool') or event.get('tool'))
        after = subprocess.run(['python', 'test_calc.py'], cwd=project, capture_output=True, text=True)
        fixed = (project / 'calc.py').read_text()
        result = dict(model=args.model, variant=args.variant, seconds=seconds, timed_out=timed_out, return_code=getattr(run, 'returncode', None),
                      events=len(events), tool_calls=[t for t in tools if t], tests_failed_before=before.returncode != 0, tests_pass_after=after.returncode == 0,
                      test_output_after=(after.stdout + after.stderr)[-200:], diff_len_chars=abs(len(fixed) - len(PROJECT['calc.py'])), stderr_tail=(getattr(run, 'stderr', '') or '')[-300:] if not timed_out else '')
        Path(args.out).write_text(json.dumps(result, indent=1))
        print(json.dumps(result, indent=1))


if __name__ == '__main__':main()
