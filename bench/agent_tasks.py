"""Prepare and grade a small frozen public-repository task set in Docker."""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from campaign_report import atomic_json

DATASET_REVISION = 'c104f840cc67f8b6eec6f759ebc8b2693d585d4a'
REPOS = ('psf/requests', 'pytest-dev/pytest', 'sphinx-doc/sphinx')


def command(args, timeout=600, cwd=None):
    patch_output = str(args[0]) == 'git' and len(args) > 1 and args[1] == 'diff'
    if os.name == 'nt' and str(args[0]) == 'git':
        # Windows copies cannot preserve Linux execute bits. Keep the override
        # local to this command; never rewrite the frozen repository's config.
        args = [args[0], '-c', 'core.filemode=false', *args[1:]]
    result = subprocess.run([str(a) for a in args], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE if patch_output else subprocess.STDOUT, timeout=timeout)
    return dict(returncode=result.returncode, output=result.stdout + ((result.stderr or '') if patch_output and result.returncode else ''))


def checked(args, timeout=600, cwd=None):
    result = command(args, timeout, cwd)
    if result['returncode']:
        raise RuntimeError('%s failed: %s' % (args[0], result['output'][-3000:]))
    return result['output']


def select(parquet, output):
    import pyarrow.parquet as pq  # Preparation-only optional dependency.
    rows = pq.read_table(parquet).to_pylist()
    ranks = {'<15 min fix': 0, '15 min - 1 hour': 1, '1-4 hours': 2, '>4 hours': 3}
    selected = []
    for repo in REPOS:
        candidates = sorted((r for r in rows if r['repo'] == repo),
                            key=lambda r: (ranks.get(r['difficulty'], 4), r['instance_id']))
        if len(candidates) < 2:
            raise ValueError('not enough dataset tasks for ' + repo)
        selected.extend(candidates)
    atomic_json(output, dict(dataset='princeton-nlp/SWE-bench_Verified', revision=DATASET_REVISION,
                            parquet_sha256=hashlib.sha256(Path(parquet).read_bytes()).hexdigest(),
                            selection='difficulty then instance_id, two preflight-eligible per repository',
                            candidates=selected))
    print(json.dumps([(r['instance_id'], r['version'], r['difficulty']) for r in selected[:12]]))


def junit(path):
    if not Path(path).exists():
        return dict(passed=False, error='missing pytest JUnit output')
    tree = ET.parse(path)
    cases = list(tree.iter('testcase'))
    errors = sum(len(list(tree.iter(tag))) for tag in ('failure', 'error'))
    skipped = len(list(tree.iter('skipped')))
    return dict(passed=bool(cases) and errors == 0 and skipped == 0,
                tests=len(cases), failures=len(list(tree.iter('failure'))),
                errors=len(list(tree.iter('error'))), skipped=skipped)


def create_container(image, suffix):
    return checked(['docker', 'create', '--name', 'qah64-' + suffix,
                    '--label', 'qah.campaign=64k', '--network', 'none', '--cap-drop', 'ALL',
                    '--security-opt', 'no-new-privileges', '--pids-limit', '256',
                    '--memory', '4g', '--cpus', '2', image, 'sleep', 'infinity']).strip()


def patch_container(container, patch, directory, name, exclude=()):
    path = directory / (name + '.patch')
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(patch)
    checked(['docker', 'cp', path, container + ':/tmp/' + name + '.patch'])
    checked(['docker', 'exec', '-w', '/workspace', container, 'git', 'apply', '--whitespace=nowarn'] +
            ['--exclude=' + path for path in exclude] + ['/tmp/' + name + '.patch'])


def run_checks(container, nodes, directory, name, timeout=120):
    destination = '/tmp/' + name + '.xml'
    result = command(['docker', 'exec', '-w', '/workspace', '-e', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1',
                      container, 'python', '-m', 'pytest', '-q', '--tb=short',
                      '--junitxml=' + destination] + nodes, timeout=timeout)
    path = directory / (name + '.xml')
    copied = command(['docker', 'cp', container + ':' + destination, path], timeout=30)
    (directory / (name + '.log')).write_text(result['output'], encoding='utf-8')
    return dict(junit(path) if copied['returncode'] == 0 else dict(passed=False, error='missing JUnit'),
                returncode=result['returncode'])


def prepare(pool, directory):
    deadline = time.monotonic() + 2700
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(pool).read_text())
    manifest = dict(dataset=data['dataset'], revision=data['revision'],
                    parquet_sha256=data['parquet_sha256'], tasks=[], exclusions=[], frozen=False)
    for repo in REPOS:
        count = 0
        for row in (r for r in data['candidates'] if r['repo'] == repo):
            if time.monotonic() >= deadline:
                raise RuntimeError('45-minute task-preparation deadline reached; manifest remains unfrozen')
            if count == 2:
                break
            case = directory / row['instance_id']
            case.mkdir(exist_ok=True)
            tag = 'qah64-' + row['instance_id'].lower().replace('__', '-')
            regression = json.loads(row['FAIL_TO_PASS'])
            preservation = sorted(set(json.loads(row['PASS_TO_PASS'])))[:20]
            (case / 'oracle.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
            dockerfile = ('FROM python:3.9-slim\n'
                'RUN apt-get update && apt-get install -y --no-install-recommends git build-essential && rm -rf /var/lib/apt/lists/*\n'
                'WORKDIR /workspace\n'
                'ENV SETUPTOOLS_SCM_PRETEND_VERSION=' + row['version'] + '\n'
                'RUN git init && git remote add origin https://github.com/' + repo + '.git && git fetch --depth 1 origin ' + row['base_commit'] + ' && git checkout FETCH_HEAD\n'
                'RUN python -m pip install --no-cache-dir "pip<26" "setuptools<70" wheel "setuptools_scm<8" "pytest<8" "pytest-timeout<3" mock hypothesis\n'
                'RUN python -m pip install --no-cache-dir -e .\n'
                'CMD ["sleep", "infinity"]\n')
            (case / 'Dockerfile').write_text(dockerfile, encoding='utf-8')
            started = time.time()
            container = None
            try:
                build = command(['docker', 'build', '-t', tag, case], timeout=600)
                (case / 'build.log').write_text(build['output'], encoding='utf-8')
                if build['returncode']:
                    raise RuntimeError('container build failed: ' + build['output'][-700:])
                container = create_container(tag, uuid.uuid4().hex[:12])
                checked(['docker', 'start', container])
                patch_container(container, row['test_patch'], case, 'oracle-tests')
                before = run_checks(container, regression, case, 'before')
                if before.get('failures', 0) == 0 or before.get('errors', 0) or before.get('skipped', 0):
                    raise RuntimeError('base must fail actual assertions without collection errors: ' + str(before))
                run_checks(container, preservation, case, 'preservation-base')
                cases = list(ET.parse(case / 'preservation-base.xml').iter('testcase'))
                supported = []
                for node in preservation:
                    matches = [c for c in cases if c.get('name') == node.split('::')[-1]]
                    if len(matches) == 1 and len(matches[0]) == 0:
                        supported.append(node)
                excluded_preservation = [n for n in preservation if n not in supported]
                preservation = supported
                if not preservation:
                    raise RuntimeError('no baseline-passing offline preservation checks')
                patch_container(container, row['patch'], case, 'known-fix')
                after = run_checks(container, regression + preservation, case, 'known-fix')
                if not after['passed'] or after['returncode']:
                    raise RuntimeError('known fix did not pass: ' + str(after))
                image_id = checked(['docker', 'image', 'inspect', '--format', '{{.Id}}', tag]).strip()
                manifest['tasks'].append(dict(instance_id=row['instance_id'], repo=repo, base_commit=row['base_commit'],
                    problem_statement=row['problem_statement'], image=tag, image_id=image_id,
                    regression=regression, preservation=preservation,
                    excluded_preservation=excluded_preservation,
                    oracle_path=str(case / 'oracle.json'), preflight=dict(before=before, after=after),
                    preparation_seconds=time.time() - started))
                count += 1
                print('eligible ' + row['instance_id'], flush=True)
            except (OSError, RuntimeError, subprocess.TimeoutExpired, ET.ParseError) as error:
                manifest['exclusions'].append(dict(instance_id=row['instance_id'], error=str(error)))
                print('excluded ' + row['instance_id'] + ': ' + str(error), flush=True)
            finally:
                if container:
                    command(['docker', 'rm', '-f', container], timeout=30)
                if not any(t['image'] == tag for t in manifest['tasks']):
                    command(['docker', 'image', 'rm', tag], timeout=30)
                atomic_json(directory / 'manifest.json', manifest)
            if time.time() - started > 900:
                raise RuntimeError('task setup exceeded its preparation budget')
        if count != 2:
            raise RuntimeError('could not preflight two tasks for ' + repo)
    manifest['frozen'] = True
    manifest['frozen_at'] = time.time()
    atomic_json(directory / 'manifest.json', manifest)
    return manifest


def grade(task, workspace, directory, patch=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if patch is None:
        patch = checked(['git', 'diff', '--binary', 'HEAD'], cwd=workspace)
    with (directory / 'agent.patch').open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(patch)
    oracle = json.loads(Path(task['oracle_path']).read_text())
    heldout_paths = re.findall(r'^\+\+\+ b/(.+)$', oracle['test_patch'], re.MULTILINE)
    if not heldout_paths:
        raise ValueError('held-out patch contains no test paths')
    container = create_container(task['image_id'], uuid.uuid4().hex[:12])
    try:
        checked(['docker', 'start', container])
        if patch.strip():
            # Agent-authored tests must neither replace nor prevent the held-out tests.
            patch_container(container, patch, directory, 'agent', exclude=heldout_paths)
        patch_container(container, oracle['test_patch'], directory, 'heldout-tests')
        result = run_checks(container, task['regression'] + task['preservation'], directory, 'grade')
        return dict(result, passed=result['passed'] and result['returncode'] == 0)
    finally:
        command(['docker', 'rm', '-f', container], timeout=30)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    select_p = sub.add_parser('select')
    select_p.add_argument('parquet')
    select_p.add_argument('output')
    prep = sub.add_parser('prepare')
    prep.add_argument('pool')
    prep.add_argument('directory')
    args = p.parse_args()
    if args.action == 'select':
        select(args.parquet, args.output)
    else:
        prepare(args.pool, args.directory)
