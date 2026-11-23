"""Build a staging gateway configuration for qualifying the Obliterated profile without touching production.

    python stage_qualification.py --production ~/llm/serving/config.json --kv f16 --out staging/config-f16.json [--swift-cache-ram 8192]

The staging config is a copy of production with: gateway port 18095 (the production backend port is reused because the production
gateway is stopped while the GPU is qualified), a separate usage database, the Obliterated profile (checkpoint hash verified, the
tool-capable chat template extracted and pinned), and `selftest-oom`, a copy of the profile whose context cannot fit in VRAM and is
used only to prove failed-load recovery. Production configuration files are never modified.
"""
import argparse
import copy
import json
from pathlib import Path

import add_profiles
import profiles as registry

PROVISIONAL = {   # research measurements (run w2/w3/w5 in research/obliterated-q4km-2080ti); replaced by the qualification numbers on deployment
    'f16': dict(loaded_mib=20850, peak_mib=20850, process_floor_mib=18500, qualification='Provisional (staging): research measurements only.', qualification_status='experimental'),
    'q8_0': dict(loaded_mib=19150, peak_mib=19150, process_floor_mib=16800, qualification='Provisional (staging): research measurements only.', qualification_status='experimental')}


def build(production, kv, swift_cache_ram=None, port=18095, root=None):
    config = json.loads(Path(production).read_text())
    config['port'] = port
    root = Path(root or Path(production).parent)
    config['database'] = str(root / 'staging-usage.sqlite3')
    profiles = dict(config['profiles'])
    template_sha = add_profiles.install_template()
    stat = add_profiles.verify_obliterated()
    obl = registry.obliterated_profile(add_profiles.STOCK_RUNTIME, add_profiles.OBLITERATED_MODEL, stat, add_profiles.OBLITERATED_SHA256, config['backend_port'],
                                       add_profiles.TEMPLATE_FILE, template_sha, kv, PROVISIONAL[kv])
    profiles['obliterated'] = obl
    oom = copy.deepcopy(obl)
    oom.update(display_name='Selftest OOM', model_id='selftest-oom', context=262144, minimum_load_free_bytes=2 ** 30, minimum_process_device_bytes=0,
               qualification='Failure-injection profile; never offered in production.')
    argv = list(oom['argv'])
    argv[argv.index('--ctx-size') + 1] = '262144'
    argv[argv.index('--cache-type-k') + 1] = 'f16'
    argv[argv.index('--cache-type-v') + 1] = 'f16'
    argv[argv.index('--alias') + 1] = 'selftest-oom'
    oom['argv'] = argv
    profiles['selftest-oom'] = oom
    config['profiles'] = profiles
    if swift_cache_ram is not None:add_profiles.apply_swift_cache(config, swift_cache_ram)
    registry.normalize(config)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--production', required=True)
    parser.add_argument('--kv', choices=('f16', 'q8_0'), required=True)
    parser.add_argument('--swift-cache-ram', type=int)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    config = build(args.production, args.kv, args.swift_cache_ram, root=Path(args.out).resolve().parent)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(config, indent=2) + '\n')
    Path(args.out).chmod(0o600)
    print('wrote', args.out, '|', ', '.join(config['profiles']))
