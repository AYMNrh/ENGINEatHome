"""Add the validated Inkling 2 Mini profiles to an existing serving config.json (idempotent; backs up first; prints no secrets).

Pins the research runtime (bin-rp2, llama.cpp fork baseline 7d3e5b1) into an immutable directory owned by serving, verifies
file hashes against the source, hard-links the validated GGUF next to the Swift model and verifies its checksum.
Nothing is rebuilt and no inference setting of the Swift profile is changed.

usage: python add_profiles.py [--config config.json] [--dry-run]
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

import profiles as registry

HOME = Path.home() / 'llm'
SOURCE_RUNTIME = HOME / 'experiments/inkling-kernels/bin-rp2'
RUNTIME = HOME / 'runtimes/inkling-7d3e5b1-bin-rp2'
SOURCE_MODEL = HOME / 'experiments/inkling-kernels/model/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf'
MODEL = HOME / 'models/Inkling-2-Mini-UD-IQ3_XXS.mainline.gguf'
MODEL_SHA256 = 'fd9ccedae17c1bb89744612fbb46730b5275f2a342133d0419776bd1ae311657'   # model/mainline.sha256 in the research tree
RUNTIME_FILES = ('llama-server', 'libggml-base.so.1.0.0', 'libggml-cpu.so.1.0.0', 'libggml-cuda.so.1.0.0', 'libggml.so.1.0.0',
                 'libllama-common.so.1.0.0', 'libllama.so.1.0.0', 'libllama-server-impl.so', 'libmtmd.so.1.0.0')
# Loaded / peak VRAM from the 2026-11-22 long-context qualification (LONGCTX_READINESS.md); the floors are the model + KV buffers
# (10,863 MiB weights, KV 4,352+106 MiB at 262,144, proportionally less at 204,800) rounded down for the residency guard.
MEASURED = dict(
    balanced=dict(loaded_mib=20542, peak_mib=20570, process_floor_mib=14000,
                  qualification='Qualified at 127,671 and 185,099 actual input tokens (all checks passed, 0 scheduler re-plans). Cold prefill 168-286 s.',
                  notes='Q8_0 K/V. Faster cold prefill than Maximum (ubatch 512).'),
    maximum=dict(loaded_mib=19824, peak_mib=19846, process_floor_mib=15000,
                 qualification=('Qualified with 242,134 actual input tokens (246.7K occupied). 262,144 is configured capacity, not a demonstrated '
                                'maximum input: nothing above 246.7K was tested. Cold prefill of 242K takes about 9.3 minutes.'),
                 notes='Q8_0 K/V. Microbatch 512 does not allocate at this context (7.40 GiB compute buffer fails), so it is not offered.'))


def install_runtime():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in RUNTIME_FILES:
        source, target = SOURCE_RUNTIME / name, RUNTIME / name
        digest = registry.sha256_file(source)
        if not target.exists() or registry.sha256_file(target) != digest:
            shutil.copy2(source, target)
            if registry.sha256_file(target) != digest:raise RuntimeError('copy verification failed: ' + name)
        manifest[name] = digest
    for link, target in (('libggml-base.so.1', 'libggml-base.so.1.0.0'), ('libggml-base.so', 'libggml-base.so.1'),
                         ('libggml-cpu.so.1', 'libggml-cpu.so.1.0.0'), ('libggml-cpu.so', 'libggml-cpu.so.1'),
                         ('libggml-cuda.so.1', 'libggml-cuda.so.1.0.0'), ('libggml-cuda.so', 'libggml-cuda.so.1'),
                         ('libggml.so.1', 'libggml.so.1.0.0'), ('libggml.so', 'libggml.so.1'),
                         ('libllama-common.so.1', 'libllama-common.so.1.0.0'), ('libllama-common.so', 'libllama-common.so.1'),
                         ('libllama.so.1', 'libllama.so.1.0.0'), ('libllama.so', 'libllama.so.1'),
                         ('libmtmd.so.1', 'libmtmd.so.1.0.0'), ('libmtmd.so', 'libmtmd.so.1')):
        path = RUNTIME / link
        if path.is_symlink() or path.exists():path.unlink()
        path.symlink_to(target)
    (RUNTIME / 'MANIFEST.json').write_text(json.dumps(dict(
        source=str(SOURCE_RUNTIME), baseline='llama.cpp fork, branch relpos-gpu, 7d3e5b1 (per LONGCTX_READINESS.md)',
        note=('The binary reports commit e4a7ea6 in --version: it was built (16:43) before 7d3e5b1 was committed (16:47) from the same '
              'tree. Identity is pinned by the SHA256 values below and was validated behaviourally.'), sha256=manifest), indent=2) + '\n')
    return manifest


def install_model():
    if not MODEL.exists():
        try:os.link(SOURCE_MODEL, MODEL)
        except OSError:shutil.copy2(SOURCE_MODEL, MODEL)
    if MODEL.stat().st_size != SOURCE_MODEL.stat().st_size:raise RuntimeError('model size differs from the validated file')
    if registry.sha256_file(MODEL) != MODEL_SHA256:raise RuntimeError('Inkling checkpoint checksum mismatch')
    stat = MODEL.stat()
    return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def merge(config_path, dry_run=False):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    existing = dict(config.get('profiles') or {})
    if 'swift' in existing:
        profiles = {'swift': existing['swift']}   # an existing Swift profile (possibly tuned by hand) is never rebuilt
    else:
        profiles = dict(registry.normalize({k: v for k, v in config.items() if k not in ('profiles', 'default_profile')})['profiles'])
    manifest = install_runtime() if not dry_run else {}
    model_stat = install_model() if not dry_run else dict(bytes=0, mtime_ns=0)
    inkling = registry.inkling_profiles(RUNTIME / 'llama-server', MODEL, model_stat, MODEL_SHA256, config['backend_port'], MEASURED)
    profiles.update(inkling)
    swift = profiles['swift']
    swift.setdefault('vram_loaded_mib', 19425)
    swift.setdefault('qualification', 'Production configuration validated 2026-11-19 (262,144 context, Q4_0 KV, MTP3, 175 W).')
    config['profiles'] = profiles
    config.setdefault('default_profile', 'swift')
    registry.normalize(config)   # validates before anything is written
    if dry_run:
        print(json.dumps({k: dict(model_id=p['model_id'], context=p['context'], kv=p['kv'], microbatch=p['microbatch']) for k, p in profiles.items()}, indent=1))
        return config
    backup = config_path.with_name(f'config.json.before-profiles-{time.strftime("%Y%m%dT%H%M%S")}')
    shutil.copy2(config_path, backup);backup.chmod(0o600)
    config_path.write_text(json.dumps(config, indent=2) + '\n');config_path.chmod(0o600)
    print('profiles written:', ', '.join(profiles), '| runtime files verified:', len(manifest), '| backup:', backup.name)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(Path(__file__).resolve().parent / 'config.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    merge(args.config, args.dry_run)
