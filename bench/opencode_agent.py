"""Isolated OpenCode runner for the frozen hard coding task and native Strata."""
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from agent_tasks import checked, command, create_container, grade
from campaign_report import atomic_json
from qah.telemetry import Monitor

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'qwen3.8-flash-next-iq3_s'
CLI = Path(os.environ['APPDATA']) / 'npm/node_modules/opencode-ai/bin/opencode.exe'
PROFILES = [(1024, 'thinking1k'), (4096, 'thinking4k'), (0, 'thinking-auto')]


def config(url, budget, model=MODEL, context=65536, delegation=False, output_tokens=8192, reasoning_effort='high',
           sampling=None):
    settings = dict(model='strata/' + model, small_model='strata/' + model,
        enabled_providers=['strata'], autoupdate=False, share='disabled', snapshot=False,
        lsp=False, formatter=False, plugin=[],
        provider=dict(strata=dict(npm='@ai-sdk/openai-compatible', name='Local Strata',
            options=dict(baseURL=url + '/v1', apiKey='local-benchmark', timeout=600000),
            models={model: dict(name=model + ' %dK' % (context // 1024), reasoning=True, tool_call=True, temperature=True,
                limit=dict(context=context, output=output_tokens),
                options=dict(reasoningEffort=reasoning_effort, reasoning_budget_tokens=budget,
                             seed=42, top_k=1, min_p=0, repeat_penalty=1,
                             chat_template_kwargs=dict(enable_thinking=True)))})),
        agent=dict(build=dict(steps=100, temperature=0, top_p=1,
            permission=dict(task='deny', webfetch='deny', websearch='deny',
                            external_directory='deny', question='deny'))),
        compaction=dict(auto=True, prune=True, reserved=output_tokens))
    options = settings['provider']['strata']['models'][model]['options']
    if budget is None:
        options.pop('reasoning_budget_tokens')
    if sampling is not None:
        options.update(sampling)
        settings['agent']['build'].update(temperature=sampling['temperature'], top_p=sampling['top_p'])
    if delegation:
        settings['agent']['build']['permission']['task'] = {'*': 'deny', 'explore': 'allow'}
        settings['agent']['explore'] = dict(model='strata/' + model,
            permission={'*': 'deny', 'read': 'allow', 'grep': 'allow', 'glob': 'allow',
                        'external_directory': 'deny'})
    return settings


def environment(case, container):
    env = {k: v for k, v in os.environ.items() if k.upper() in (
        'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'APPDATA', 'LOCALAPPDATA',
        'USERPROFILE', 'TEMP', 'TMP', 'PROGRAMFILES', 'PROGRAMFILES(X86)')}
    for key, folder in [('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'),
                        ('XDG_CACHE_HOME', 'cache'), ('XDG_STATE_HOME', 'state')]:
        env[key] = str((case / folder).resolve())
    env.update(OPENCODE_CONFIG_DIR=str((case / 'config/opencode').resolve()),
        OPENCODE_CONFIG=str((case / 'opencode.json').resolve()),
        OPENCODE_DISABLE_PROJECT_CONFIG='true', OPENCODE_DISABLE_MODELS_FETCH='true',
        OPENCODE_DISABLE_AUTOUPDATE='true', OPENCODE_PURE='true',
        QAH_CONTAINER=container, NO_PROXY='127.0.0.1,localhost', PYTHONUTF8='1')
    return env


def tools(case):
    directory = case / 'config/opencode/tools'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'bash.ts').write_text('''import { tool } from "@opencode-ai/plugin";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
const run = promisify(execFile);
export default tool({
  description: "Run an offline Linux command in the task container. Repository is /workspace. Use this for all shell commands and tests; edits through normal file tools share this workspace.",
  args: {command: tool.schema.string(), description: tool.schema.string().optional(), timeout_seconds: tool.schema.number().positive().max(300).optional().describe("Wall-clock seconds, e.g. 120 for two minutes; default 120, maximum 300")},
  async execute(args) {
    const container = process.env.QAH_CONTAINER;
    if (!container || !/^(qah64-[a-z0-9]+|[a-f0-9]{64})$/.test(container)) throw new Error("Invalid task container");
    try {
      const seconds = args.timeout_seconds ?? 120;
      const r = await run("docker.exe", ["exec", "-w", "/workspace", container, "timeout", "--signal=TERM", "--kill-after=5s", `${seconds}s`, "bash", "-lc", args.command], {timeout: (seconds+10)*1000, maxBuffer: 4*1024*1024});
      return JSON.stringify({exit_code:0, stdout:r.stdout, stderr:r.stderr});
    } catch(e) {
      return JSON.stringify({exit_code:e.code || 1, stdout:e.stdout || "", stderr:e.stderr || String(e), timed_out:!!e.killed || e.code===124 || e.code===137});
    }
  }
});
''', encoding='utf-8')


def run(directory, url, native_pid, engine_log, full_matrix=False, efficiency=False, full_experts=False,
        stability_workspace=None, model=MODEL, skip_stability=False, workflow_guidance=False,
        context=65536, trials=None, delegation=False, stability_smoke=False, reference=False,
        reference_profile=None, worker_policy=None, backend_label=None, output_tokens=8192,
        task_ids=None, track_patch_steps=False, supervisor_provider='codex', reasoning_effort='high',
        unrestricted_reasoning=False, sampling=None):
    from qah.model import sha256
    from run import provenance, source_snapshot
    directory = Path(directory)
    state = json.loads((directory / 'status.json').read_text(encoding='utf-8-sig'))
    manifest = ROOT / 'bench/results/campaign-64k/tasks/manifest.json'
    manifest_data = json.loads(manifest.read_text())
    if not manifest_data.get('frozen') or len(manifest_data['tasks']) != 6:
        raise ValueError('Six frozen preflight-eligible tasks are required')
    tasks = manifest_data['tasks']
    for task in tasks:
        if task['oracle_path'].startswith('/mnt/c/'):
            task['oracle_path'] = 'C:/' + task['oracle_path'][7:]
    hard_task = next(t for t in tasks if t['instance_id'] == 'sphinx-doc__sphinx-10323')
    selected_tasks = tasks
    if task_ids is not None:
        requested = set(task_ids)
        selected_tasks = [task for task in tasks if task['instance_id'] in requested]
        found = {task['instance_id'] for task in selected_tasks}
        if found != requested:
            raise ValueError('Unknown task IDs: ' + ', '.join(sorted(requested - found)))
    profile_name = (reference_profile or ('dense-iq3s-thinking1k' if reference else
                    'coder-thinking1k' if model != MODEL else
                    'thinking1k-full-experts' if full_experts else 'thinking1k-fixed-timeout'))
    jobs = ([(1024, profile_name, t) for t in selected_tasks] if task_ids is not None else
            [(1024, profile_name, t) for t in tasks
             if t['instance_id'] in ('pytest-dev__pytest-6202', 'sphinx-doc__sphinx-10323')] if efficiency else
            [(1024, reference_profile or 'thinking1k', t) for t in tasks] if full_matrix
            else [(budget, name, hard_task) for budget, name in PROFILES])
    tail_skip = None if reference else 0 if full_experts else 7
    base = dict(context=context, backend=backend_label or ('llama.cpp' if reference else 'Strata0.1.59'), agent='OpenCode1.18.16',
                model=model,
                workflow_guidance=workflow_guidance, delegation=delegation, worker_policy=worker_policy,
                supervisor_provider=supervisor_provider,
                task_manifest_sha256=sha256(manifest), provenance=provenance(),
                expert_tail_skip=tail_skip, tool_timeout_unit='seconds',
                backend_config_sha256=sha256(directory / 'config.json'))
    source_snapshot(directory / ('sources-' + base['provenance']['source_sha256'] + '.json'))

    def event(kind, **values):
        row = dict(base, kind=kind, time=time.time(), **values)
        with (directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row) + '\n')
        return row

    def status(phase, **values):
        state.update(phase=phase, updated_at=time.time(), **values)
        atomic_json(directory / 'status.json', state)

    version = checked([CLI, '--version']).strip()
    if version != '1.18.16':
        raise RuntimeError('OpenCode version changed after preflight')
    timing = re.compile(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in ([\d.]+) ms.*?, (\d+) generated in ([\d.]+) ms')
    reference_prompt = re.compile(r'task\s+(\d+) \| new prompt, .*?task\.n_tokens = (\d+)')
    reference_timing = re.compile(
        r'task\s+(\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens[^\n]*\n'
        r'[^\n]*?\|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens')
    reference_drafts = re.compile(
        r'task\s+(\d+) \| draft acceptance = [\d.]+\s+\(\s*(\d+) accepted /\s*(\d+) generated')
    log_offset = len(Path(engine_log).read_text(encoding='utf-8', errors='replace'))
    rows = []
    control_workspace = None

    def collect_log(profile):
        nonlocal log_offset
        text = Path(engine_log).read_text(encoding='utf-8', errors='replace')
        latest = text[log_offset:]
        if reference:
            prompts = dict(reference_prompt.findall(latest))
            drafts = {task_id: (int(accepted), int(proposed))
                      for task_id, accepted, proposed in reference_drafts.findall(latest)}
            for task_id, prefill, fresh, decode, output in reference_timing.findall(latest):
                total = int(prompts[task_id]) if task_id in prompts else None
                accepted, proposed = drafts.get(task_id, (None, None))
                event('request_result', profile=profile, prompt_tokens=total,
                      cached_input_tokens=total - int(fresh) if total is not None else None,
                      draft_n=proposed, draft_n_accepted=accepted,
                      timings=dict(prompt_n=int(fresh), prompt_ms=float(prefill),
                                   predicted_n=int(output), predicted_ms=float(decode)))
            log_offset = len(text)
            return
        for match in timing.finditer(latest):
            total, cached, fresh, prefill_ms, output, decode_ms = map(float, match.groups())
            event('request_result', profile=profile, prompt_tokens=int(total), cached_input_tokens=int(cached),
                  timings=dict(prompt_n=int(fresh), prompt_ms=prefill_ms,
                               predicted_n=int(output), predicted_ms=decode_ms))
        log_offset = len(text)

    with Monitor(SimpleNamespace(process=SimpleNamespace(pid=native_pid)),
                 output=directory / 'telemetry.jsonl') as monitor:
        if stability_workspace:
            if time.time() + (600 if stability_smoke else 4200) >= state['deadline']:
                raise TimeoutError('Insufficient time for corrected one-hour stability')
            from opencode_stability import run as stability_run
            stability_run(directory, url, hard_task, stability_workspace, event, status, monitor, model=model,
                          context=context, duration_seconds=180 if stability_smoke else 3600)
            collect_log('thinking1k-stability')
            status('finished', status='completed', promotion_passed=False)
            return
        for trial in range(trials if trials is not None else (3 if full_matrix or efficiency else 2)):
            for budget, name, task in jobs:
                if unrestricted_reasoning:
                    budget = None
                if efficiency and trial > 0 and task['instance_id'] != 'pytest-dev__pytest-6202':
                    continue
                if time.time() + 1380 >= state['deadline']:
                    status('deadline', status='failed', blockers=['Insufficient remaining time for complete attempt'])
                    return
                case = directory / (('%s-%s-t%d' % (name, task['instance_id'], trial))
                                    if full_matrix or efficiency else ('%s-t%d' % (name, trial)))
                case.mkdir()
                workspace = case / 'workspace'
                workspace.mkdir()
                container = None
                started = time.time()
                status('opencode_coding', status='running', active_profile=name,
                       active_task=task['instance_id'], trial=trial)
                event('task_started', profile=name, instance_id=task['instance_id'], trial=trial)
                try:
                    copied = create_container(task['image_id'], uuid.uuid4().hex[:12])
                    try:
                        checked(['docker', 'cp', copied + ':/workspace/.', workspace])
                    finally:
                        command(['docker', 'rm', '-f', copied])
                    if checked(['git', 'status', '--porcelain'], cwd=workspace).strip():
                        raise RuntimeError('Frozen workspace is dirty')
                    container = checked(['docker', 'run', '-d', '--name', 'qah64-' + uuid.uuid4().hex[:12],
                        '--label', 'qah.campaign=64k-opencode', '--network', 'none', '--cap-drop', 'ALL',
                        '--security-opt', 'no-new-privileges', '--pids-limit', '256', '--memory', '4g',
                        '--cpus', '2', '--mount', 'type=bind,source=%s,target=/workspace' % workspace.resolve(),
                        '-w', '/workspace', task['image_id'], 'sleep', 'infinity']).strip()
                    atomic_json(case / 'opencode.json', config(url, budget, model=model, context=context,
                                                              delegation=delegation, output_tokens=output_tokens,
                                                              reasoning_effort=reasoning_effort, sampling=sampling))
                    tools(case)
                    prompt = ('Fix the repository issue below. Use normal file tools to inspect/edit this workspace; '
                        'use bash for Linux commands and focused offline tests in /workspace. Do not use host commands. '
                        'Make the smallest correct implementation change and do not modify tests to hide failures. '
                        'Preserve the fix after checking baseline failures. Finish with a concise explanation once verified. '
                        'No network access.\n\n' + task['problem_statement'])
                    if workflow_guidance:
                        prompt += ('\n\nWorkflow guidance: Your total agent wall-clock budget is 1200 seconds, starting at '
                            'UNIX time %.0f. Check current time with date +%%s when planning a costly command. '
                            'Inspect and run only tests relevant to this issue first. Avoid the full repository suite '
                            'unless a focused failure requires it. Prefer a foreground focused command with an explicit '
                            'timeout_seconds within the remaining budget. Avoid long sleeps and background test loops; '
                            'if polling is necessary, use waits of at most 5 seconds. Reserve the final 120 seconds '
                            'to restore the intended fix after baseline checks, inspect git diff --check and git diff '
                            '--stat, and confirm the actual source fix remains in the final working tree. Do not '
                            'leave the intended fix only in a stash. Once focused verification succeeds, finish '
                            'with a concise explanation rather than expanding test scope.' % started)
                    if delegation:
                        prompt += ('\n\nAct as orchestrator. Use the Task tool to create separate Explore sessions for as '
                            'many independent lines of investigation as the issue supports. Start with distinct searches '
                            'for root cause, affected callers and focused regression tests. Ask each child for concise '
                            'findings with file references. Delegate independent patch review too when time permits. '
                            'Keep the children read-only, avoid duplicate work, and make all edits and run tests yourself. '
                            'Keep the existing time budget and final patch checks.')
                    atomic_json(case / 'definition.json', dict(reasoning_budget_tokens=budget,
                        workflow_guidance=workflow_guidance, delegation=delegation, context=context,
                        reasoning_effort=reasoning_effort, output_tokens=output_tokens, max_steps=100,
                        agent_timeout_seconds=1200, prompt=prompt, version=version,
                        tool_timeout_unit='seconds', expert_tail_skip=tail_skip, worker_policy=worker_policy,
                        supervisor_provider=supervisor_provider))
                    timeout = False
                    worker_summary = None
                    patch_snapshots = []
                    argv = [str(CLI), 'run', '--pure', '--format', 'json', '--model', 'strata/' + model, prompt]
                    if unrestricted_reasoning:
                        argv.insert(3, '--thinking')
                    env = environment(case, container)
                    env.pop('DEEPSEEK_API_KEY', None)
                    if worker_policy:
                        from opencode_guard import run_worker
                        records, finished, timeout, worker_summary = run_worker(
                            argv, workspace, env, case, worker_policy, task['problem_statement'],
                            container, lambda phase: status(phase), supervisor_provider)
                        atomic_json(case / 'worker-summary.json', worker_summary)
                    else:
                        stdout_path = case / 'opencode.stdout.jsonl'
                        with stdout_path.open('w', encoding='utf-8') as stdout, \
                             (case / 'opencode.stderr.log').open('w', encoding='utf-8') as stderr:
                            process = subprocess.Popen(
                                argv, cwd=workspace, env=env,
                                stdout=subprocess.PIPE if track_patch_steps else stdout,
                                stderr=stderr, text=track_patch_steps, encoding='utf-8' if track_patch_steps else None,
                                errors='replace' if track_patch_steps else None, bufsize=1 if track_patch_steps else -1)
                            reader = None
                            if track_patch_steps:
                                patch_directory = case / 'patch-steps'
                                patch_directory.mkdir()

                                def observe_steps():
                                    step = 0
                                    last_patch = None
                                    for line in process.stdout:
                                        stdout.write(line)
                                        stdout.flush()
                                        try:
                                            record = json.loads(line)
                                        except ValueError:
                                            continue
                                        if record.get('type') != 'step_finish':
                                            continue
                                        step += 1
                                        result = command(['git', 'diff', '--binary', 'HEAD'], timeout=30, cwd=workspace)
                                        if result['returncode']:
                                            patch_snapshots.append(dict(step=step, error=result['output'][-1000:]))
                                            continue
                                        patch = result['output']
                                        if patch == last_patch:
                                            continue
                                        last_patch = patch
                                        path = patch_directory / ('step-%03d.patch' % step)
                                        path.write_text(patch, encoding='utf-8', newline='\n')
                                        patch_snapshots.append(dict(step=step, path=str(path.relative_to(case)),
                                                                    bytes=len(patch.encode('utf-8'))))

                                reader = threading.Thread(target=observe_steps, name='patch-step-observer', daemon=True)
                                reader.start()
                            try:
                                process.wait(timeout=1200)
                            except subprocess.TimeoutExpired:
                                timeout = True
                                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                               capture_output=True, timeout=30)
                            if reader:
                                reader.join(timeout=30)
                                if reader.is_alive():
                                    raise RuntimeError('Patch-step observer did not drain OpenCode output')
                        records = []
                        for line in (case / 'opencode.stdout.jsonl').read_text(encoding='utf-8').splitlines():
                            try:
                                records.append(json.loads(line))
                            except ValueError:
                                continue  # Non-JSON startup diagnostics remain in the raw artifact.
                    steps = [r['part'] for r in records if r.get('type') == 'step_finish']
                    usage = [step.get('tokens') or {} for step in steps]
                    token_totals = {}
                    if unrestricted_reasoning:
                        for key in ('reasoning', 'output'):
                            counts = [tokens.get(key) for tokens in usage]
                            token_totals[f'provider_{key}_tokens'] = (
                                sum(counts) if counts and all(isinstance(n, int) for n in counts) else None)
                    errors = [r for r in records if r.get('type') == 'error']
                    tool_results = [r['part'].get('state', {}) for r in records if r.get('type') == 'tool_use']
                    delegated = [r['part'].get('state', {}) for r in records
                                 if r.get('type') == 'tool_use' and r['part'].get('tool') == 'task']
                    child_sessions = {sid for tool in delegated
                                      if (sid := tool.get('metadata', {}).get('sessionId'))}
                    shell_failures = 0
                    shell_timeouts = 0
                    for record in records:
                        if record.get('type') != 'tool_use' or record['part'].get('tool') != 'bash':
                            continue
                        output = record['part'].get('state', {}).get('output', '')
                        try:
                            shell_result = json.loads(output)
                        except (ValueError, TypeError):
                            continue  # Native tool errors have no structured command output.
                        shell_failures += bool(shell_result.get('exit_code'))
                        shell_timeouts += bool(shell_result.get('timed_out'))
                    if not worker_policy:
                        finished = bool(not timeout and process.returncode == 0 and not errors
                                        and steps and steps[-1].get('reason') == 'stop' and len(steps) < 100)
                    checked(['git', 'add', '-N', '.'], cwd=workspace)
                    verdict = grade(task, workspace, case)
                    first_correct_patch_step = None
                    if track_patch_steps:
                        for snapshot in patch_snapshots:
                            if 'path' not in snapshot or not snapshot.get('bytes'):
                                continue
                            patch = (case / snapshot['path']).read_text(encoding='utf-8')
                            snapshot_grade = grade(
                                task, workspace, case / 'patch-step-grades' / ('step-%03d' % snapshot['step']),
                                patch=patch)
                            snapshot['grade_passed'] = snapshot_grade['passed']
                            if snapshot_grade['passed']:
                                first_correct_patch_step = snapshot['step']
                                break
                        database = case / 'data/opencode/opencode.db'
                        summaries = compaction_parts = 0
                        if database.is_file():
                            with sqlite3.connect(database) as connection:
                                summaries = connection.execute(
                                    "SELECT count(*) FROM message WHERE json_extract(data, '$.summary') = 1"
                                ).fetchone()[0]
                                compaction_parts = connection.execute(
                                    "SELECT count(*) FROM part WHERE json_extract(data, '$.type') = 'compaction'"
                                ).fetchone()[0]
                        compactions = dict(occurred=bool(summaries or compaction_parts),
                                           summaries=summaries, parts=compaction_parts)
                        atomic_json(case / 'patch-timeline.json', dict(
                            snapshots=patch_snapshots, first_correct_patch_step=first_correct_patch_step,
                            compactions=compactions))
                    else:
                        compactions = None
                    row = event('task_result', profile=name, trial=trial, instance_id=task['instance_id'],
                        passed=verdict['passed'] and finished, agent_finished=finished, grade=verdict,
                        elapsed_seconds=time.time() - started, agent_timeout=timeout,
                        turn_exit_reason=steps[-1].get('reason') if steps else None,
                        api_calls=len(steps), tool_calls=len(tool_results),
                        delegation_calls=len(delegated), completed_delegations=sum(tool.get('status') == 'completed' for tool in delegated),
                        delegation_session_ids=sorted(child_sessions),
                        tool_call_errors=sum(t.get('status') == 'error' for t in tool_results),
                        shell_command_failures=shell_failures, shell_command_timeouts=shell_timeouts,
                        first_correct_patch_step=first_correct_patch_step,
                        patch_snapshots=len(patch_snapshots), compactions=compactions,
                        supervision=worker_summary, memory=monitor.report(), **token_totals)
                    rows.append(row)
                    state['completed_tasks'] = len(rows)
                    state['passed_tasks'] = sum(r['passed'] for r in rows)
                    collect_log(name)
                    if task['instance_id'] == 'sphinx-doc__sphinx-10323':
                        control_workspace = workspace
                    status('attempt_finished')
                except Exception as error:
                    event('infrastructure_failure', profile=name, trial=trial, error=repr(error))
                    status('failed', status='failed', blockers=[repr(error)])
                    raise
                finally:
                    if container:
                        command(['docker', 'rm', '-f', container], timeout=30)
        if efficiency and not full_experts and not skip_stability:
            if time.time() + 4200 >= state['deadline']:
                raise TimeoutError('Insufficient remaining time for the full one-hour stability test')
            from opencode_stability import run as stability_run
            stability_task = next(t for t in tasks if t['instance_id'] == 'sphinx-doc__sphinx-10323')
            stability_run(directory, url, stability_task, control_workspace, event, status, monitor)
            collect_log('thinking1k-stability')
        atomic_json(directory / 'comparison.json', dict(attempts=len(rows), passes=sum(r['passed'] for r in rows),
                    patch_passes=sum(r['grade']['passed'] for r in rows), resources=monitor.report()))
        status('finished', status='completed', promotion_passed=False)
