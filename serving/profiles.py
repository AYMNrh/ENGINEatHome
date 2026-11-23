"""Model-profile registry: validated launch configurations, reasoning controls and public descriptions.

A profile is one complete, separately validated way to run the single GPU. Exactly one profile occupies the
RTX 2080 Ti at a time. This module is pure (no I/O besides optional file hashing helpers) so it is easy to test.

Configuration shape (config.json):
    profiles: {key: profile}, default_profile: key
If `profiles` is absent a single "swift" profile is synthesized from the legacy top-level keys, so a
configuration written before multi-model support keeps working untouched.
"""
import copy
import hashlib
from pathlib import Path
import re

SHORT_FLAGS = {'m': 'model', 'c': 'ctx-size', 'ub': 'ubatch-size', 'b': 'batch-size', 'ctk': 'cache-type-k', 'ctv': 'cache-type-v',
               'np': 'parallel', 'ngl': 'n-gpu-layers', 'fa': 'flash-attn', 't': 'threads', 'lv': 'verbosity'}

# What each model's chat template genuinely accepts (read from the GGUF templates on the homelab, 2026-11-22):
#   Swift   (qwen35)        enable_thinking bool + reasoning_effort in {xhigh (default), medium, low}
#   Inkling (inkling2_loop) reasoning_effort in {none, minimal, low, medium, high, xhigh, max} or a number 0..0.99
#                           (max and xhigh both map to 0.99; default 0.9 = high)
SWIFT_REASONING = dict(parameter='reasoning_effort', levels=['none', 'low', 'medium', 'xhigh'], default='xhigh', aliases={},
                       numeric=None, none_disables_thinking=True, request_override=True,
                       note='none turns thinking off (enable_thinking=false). low, medium and xhigh are the template levels.')
# Qwen3.8-27B-OBLITERATED uses the Qwen3.5-style template taken from the Swift GGUF (the model's own embedded template is a
# minimal ChatML without tool support). That template accepts enable_thinking and the same three effort levels as Swift.
# The default is `none` because every speed measurement of this profile was taken with thinking off.
OBLITERATED_REASONING = dict(parameter='reasoning_effort', levels=['none', 'low', 'medium', 'xhigh'], default='none', aliases={},
                             numeric=None, none_disables_thinking=True, request_override=True,
                             note=('none turns thinking off (enable_thinking=false) and is the default: speed was measured with thinking off. '
                                   'low, medium and xhigh are the levels of the shared Qwen template; their effect on this model was not measured.'))
INKLING_REASONING = dict(parameter='reasoning_effort', levels=['none', 'minimal', 'low', 'medium', 'high', 'xhigh'], default='high',
                         aliases={'max': 'xhigh'}, numeric=[0.0, 0.99], none_disables_thinking=False, request_override=True,
                         note='Template values none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99. max is an alias of xhigh.')


class ReasoningError(ValueError):
    def __init__(self, code, supported):
        super().__init__(code)
        self.code = code
        self.supported = supported


class ProfileError(ValueError):pass


def parse_argv(argv):
    """Map a llama-server argument vector to {flag-name: value or 'on'}; understands short flags and negative values."""
    values = {}
    items = list(argv)[1:]
    i = 0
    while i < len(items):
        token = items[i]
        if token.startswith('-') and not re.fullmatch(r'-\d+(\.\d+)?', token):
            name = token.lstrip('-')
            name = SHORT_FLAGS.get(name, name) if not token.startswith('--') else name
            nxt = items[i + 1] if i + 1 < len(items) else None
            if nxt is not None and (not nxt.startswith('-') or re.fullmatch(r'-\d+(\.\d+)?', nxt)):
                values[name] = nxt
                i += 2
                continue
            values[name] = 'on'
        i += 1
    return values


def launch_facts(argv):
    """Real launch settings without filesystem paths: [(label, value)] for display and a dict for the API."""
    v = parse_argv(argv or [])
    if not v:return []
    kv_k, kv_v = v.get('cache-type-k'), v.get('cache-type-v')
    ctx = int(v['ctx-size']) if v.get('ctx-size', '').isdigit() else None
    rows = [('Model file', Path(v.get('model', '')).name or 'unknown'),
            ('Context', f'{ctx:,} tokens' if ctx else 'unknown'),
            ('Parallel sequences', v.get('parallel', '1 (default)')),
            ('GPU layers', v.get('n-gpu-layers', 'unknown')),
            ('Flash attention', v.get('flash-attn', 'unknown')),
            ('KV cache', (kv_k if kv_k == kv_v else f'{kv_k} / {kv_v}') if kv_k else 'unknown'),
            ('Batch / microbatch', f"{v.get('batch-size', 'default')} / {v.get('ubatch-size', 'default')}")]
    if v.get('spec-type'):
        threshold = f", acceptance threshold {v['spec-draft-p-min']}" if v.get('spec-draft-p-min') else ''
        rows.append(('Speculative decoding', f"{v['spec-type']}, up to {v.get('spec-draft-n-max', '?')} draft tokens{threshold}"))
    if v.get('reasoning'):rows.append(('Reasoning', f"{v['reasoning']}, effort {v.get('reasoning-effort', '?')}, budget {v.get('reasoning-budget', '?')}"))
    rows.append(('Prompt cache in RAM', cache_ram_label(v)))
    if 'ctx-checkpoints' in v:rows.append(('Context checkpoints', v['ctx-checkpoints']))
    rows.append(('Context shifting', 'off' if 'no-context-shift' in v else 'runtime default'))
    template = v.get('chat-template-file')
    rows.append(('Chat template', f'file {Path(template).name}' if template else 'model-embedded (jinja)' if 'jinja' in v else 'runtime default'))
    return rows


RUNTIME_DEFAULT_CACHE_RAM_MIB = 8192   # `--cache-ram` default of the stock build and of the Inkling fork (checked with --help, 2026-11-23)


def cache_ram_mib(argv):
    """RAM prompt-cache size in MiB from the launch arguments: 0 = off, None = the runtime default (8192 MiB in llama.cpp)."""
    value = parse_argv(argv or []).get('cache-ram')
    return int(value) if value is not None and re.fullmatch(r'-?\d+', value) else None


def cache_ram_label(values):
    value = values.get('cache-ram')
    flags = ' (idle slots not cached)' if 'no-cache-idle-slots' in values else ''
    if value is None:return 'runtime default (8,192 MiB)'
    if value == '0':return 'off' + flags
    return ('unlimited' if value == '-1' else f'{int(value):,} MiB') + flags if re.fullmatch(r'-?\d+', value) else value


def mtp_facts(argv):
    v = parse_argv(argv or [])
    enabled = v.get('spec-type') == 'draft-mtp'
    return dict(enabled=enabled, draft_n_max=int(v['spec-draft-n-max']) if enabled and v.get('spec-draft-n-max', '').isdigit() else None,
                draft_p_min=float(v['spec-draft-p-min']) if enabled and v.get('spec-draft-p-min') else None)


def quantization_of(model):
    """Quantization label from a GGUF file name (informational; profiles may state it explicitly)."""
    match = re.search(r'(UD-)?(IQ\d_[A-Z]+|Q\d_K_[A-Z]|Q\d_K|Q\d_\d|Q\d|F16|BF16)', Path(model or '').name)
    return match.group(0) if match else None


def _kv(argv):
    v = parse_argv(argv or [])
    k, val = v.get('cache-type-k'), v.get('cache-type-v')
    return k if k == val else f'{k}/{val}' if k else None


def normalize(config):
    """Return a deep copy of config with a validated `profiles` registry and `default_profile`."""
    config = copy.deepcopy(config)
    if not config.get('profiles'):
        argv = config.get('argv') or []
        v = parse_argv(argv)
        config['profiles'] = {'swift': dict(
            display_name='Swift Production', family='swift', model_id=config['model_id'],
            executable=argv[0] if argv else None, model=config.get('model'), model_stat=config.get('model_stat'),
            checkpoint_sha256=config.get('checkpoint_sha256'), argv=argv, context=config.get('context'),
            kv=_kv(argv), microbatch=int(v['ubatch-size']) if v.get('ubatch-size', '').isdigit() else None,
            minimum_load_free_bytes=config.get('minimum_load_free_bytes', 0),
            minimum_process_device_bytes=config.get('minimum_process_device_bytes', 0),
            load_timeout=config.get('load_timeout', 240), sampling=config.get('sampling', {}),
            request_defaults=dict(enable_thinking=True, preserve_thinking=True), reasoning=copy.deepcopy(SWIFT_REASONING),
            prefix_cache=False, description='Qwen3.8-27B Swift 1.5, the validated production configuration.')}
        config['default_profile'] = 'swift'
    default = config.setdefault('default_profile', next(iter(config['profiles'])))
    if default not in config['profiles']:raise ProfileError('default_profile is not a configured profile: ' + default)
    ids = {}
    for key, p in config['profiles'].items():
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', key):raise ProfileError('invalid profile key: ' + key)
        for field in ('display_name', 'model_id', 'family', 'reasoning'):
            if field not in p:raise ProfileError(f'profile {key} is missing {field}')
        if p['model_id'] in ids:raise ProfileError(f"duplicate model_id {p['model_id']} in {key} and {ids[p['model_id']]}")
        ids[p['model_id']] = key
        r = p['reasoning']
        if r['default'] not in r['levels']:raise ProfileError(f'profile {key}: default reasoning level is not an advertised level')
        p.setdefault('sampling', {})
        p.setdefault('request_defaults', {})
        p.setdefault('load_timeout', 240)
        p.setdefault('minimum_load_free_bytes', 0)
        p.setdefault('minimum_process_device_bytes', 0)
        p.setdefault('prefix_cache', True)
    return config


def model_index(config):
    """model_id -> profile key."""
    return {p['model_id']: key for key, p in config['profiles'].items()}


def public_profile(key, p):
    """Profile description for the API and dashboard. No executable, model path, argv or checksum paths."""
    r = p['reasoning']
    ram = cache_ram_mib(p.get('argv'))
    if ram is None:ram = RUNTIME_DEFAULT_CACHE_RAM_MIB   # no --cache-ram flag: llama.cpp enables its 8,192 MiB default
    return dict(key=key, display_name=p['display_name'], model_id=p['model_id'], family=p['family'], context=p.get('context'),
                kv_cache=p.get('kv') or _kv(p.get('argv')), microbatch=p.get('microbatch'), prefix_cache=bool(p.get('prefix_cache', True)),
                description=p.get('description', ''), notes=p.get('notes', ''), qualification=p.get('qualification', ''),
                qualification_status=p.get('qualification_status', 'qualified'), quantization=p.get('quantization') or quantization_of(p.get('model')),
                mtp=mtp_facts(p.get('argv')), ram_cache=dict(enabled=bool(ram), mib=ram if ram else 0, label=cache_ram_label(parse_argv(p.get('argv') or []))),
                expected_vram_mib=p.get('vram_loaded_mib'),
                reasoning=dict(parameter=r['parameter'], levels=list(r['levels']), default=r['default'], aliases=dict(r.get('aliases', {})),
                               numeric=r.get('numeric'), request_override=bool(r.get('request_override', True)), note=r.get('note', '')),
                launch=[dict(label=a, value=b) for a, b in launch_facts(p.get('argv'))])


def resolve_effort(spec, requested, dashboard_default=None):
    """Decide the thinking effort to forward for one request.

    requested: what the client supplied (string, number or None). Returns dict(label, kwargs, source).
    `label` is the effort that is *forwarded to the chat template*; it is a request, not a measurement of the model.
    Raises ReasoningError for anything the model's template does not accept.
    """
    supported = list(spec['levels'])
    source = 'request'
    value = requested
    if value is None:
        value = dashboard_default if dashboard_default in spec['levels'] else spec['default']
        source = 'dashboard' if dashboard_default in spec['levels'] else 'profile'
    if isinstance(value, bool):raise ReasoningError('unsupported_reasoning_effort', supported)
    if isinstance(value, str):
        level = spec.get('aliases', {}).get(value.strip().lower(), value.strip().lower())
        if level not in spec['levels']:raise ReasoningError('unsupported_reasoning_effort', supported)
        if level == 'none' and spec.get('none_disables_thinking'):
            return dict(label='none', kwargs=dict(enable_thinking=False), source=source)
        return dict(label=level, kwargs={spec['parameter']: level}, source=source)
    if isinstance(value, (int, float)) and spec.get('numeric'):
        low, high = spec['numeric']
        if low <= float(value) <= high:return dict(label=str(float(value)), kwargs={spec['parameter']: float(value)}, source=source)
    raise ReasoningError('unsupported_reasoning_effort', supported)


# ------------------------------------------------------------------------------------------------
# Building the Inkling profiles from validated homelab artifacts (used by add_profiles.py)
# ------------------------------------------------------------------------------------------------
def sha256_file(path, block=8 * 2 ** 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):digest.update(chunk)
    return digest.hexdigest()


def inkling_argv(executable, model, context, microbatch, port, alias):
    """The qualified command lines (v06 long-context report) plus only what the gateway needs: host, port, alias, quiet."""
    return [str(executable), '-m', str(model), '-ngl', '99', '-fa', 'on', '-ctk', 'q8_0', '-ctv', 'q8_0', '-c', str(context),
            '-ub', str(microbatch), '-np', '1', '--jinja', '--no-warmup', '--host', '127.0.0.1', '--port', str(port),
            '--alias', alias, '--no-webui', '--log-disable']


def inkling_profiles(executable, model, model_stat, sha256, port, measured):
    """measured: {'balanced': {...}, 'maximum': {...}} loaded-VRAM figures from the qualification runs (MiB)."""
    common = dict(family='inkling', executable=str(executable), model=str(model), model_stat=model_stat, checkpoint_sha256=sha256,
                  kv='q8_0', sampling={}, request_defaults={}, reasoning=copy.deepcopy(INKLING_REASONING), prefix_cache=True,
                  load_timeout=420, runtime='llama.cpp fork relpos-gpu (7d3e5b1), validated build bin-rp2')
    out = {}
    for key, name, model_id, ctx, ub, text in (
            ('inkling-balanced', 'Inkling Balanced', 'inkling-2-mini-balanced', 204800, 512,
             'Inkling 2 Mini UD-IQ3_XXS, 204,800 context, microbatch 512. Faster cold prefill; intended for 128K-185K inputs.'),
            ('inkling-maximum', 'Inkling Maximum', 'inkling-2-mini-maximum', 262144, 256,
             'Inkling 2 Mini UD-IQ3_XXS, 262,144 configured context, microbatch 256 (microbatch 512 does not allocate at this size).')):
        m = measured[key.split('-')[1]]
        p = dict(common, display_name=name, model_id=model_id, context=ctx, microbatch=ub, description=text,
                 argv=inkling_argv(executable, model, ctx, ub, port, model_id), vram_loaded_mib=m['loaded_mib'],
                 minimum_load_free_bytes=(m['peak_mib'] + 512) * 2 ** 20, minimum_process_device_bytes=int(m['process_floor_mib']) * 2 ** 20,
                 qualification=m['qualification'], notes=m.get('notes', ''))
        out[key] = p
    return out


def obliterated_argv(executable, model, context, kv, port, alias, template_file, cache_ram=8192):
    """The measured launch line (research/obliterated-q4km-2080ti, run w2/w3) plus what the gateway needs: loopback host/port, alias,
    quiet, and the tool-capable chat template. Swift's flags are untouched; nothing else is guessed."""
    return [str(executable), '--model', str(model), '--ctx-size', str(context), '--host', '127.0.0.1', '--port', str(port),
            '--alias', alias, '--parallel', '1', '--n-gpu-layers', '99', '--flash-attn', 'on', '--cache-type-k', kv, '--cache-type-v', kv,
            '--batch-size', '2048', '--ubatch-size', '512', '--threads', '8', '--fit', 'off', '--no-context-shift',
            '--cache-ram', str(cache_ram), '--ctx-checkpoints', '4', '--no-webui', '--jinja', '--chat-template-file', str(template_file),
            '--no-warmup', '--spec-type', 'draft-mtp', '--spec-draft-n-max', '3', '--spec-draft-p-min', '0.6',
            '--spec-draft-type-k', 'q4_0', '--spec-draft-type-v', 'q4_0', '--spec-draft-n-ubatch', '64', '--log-disable'] + (
                ['--no-cache-idle-slots'] if cache_ram > 0 else [])   # llama.cpp #27148 hardening, see docs/obliterated-profile-and-ram-cache.md


def obliterated_profile(executable, model, model_stat, sha256, port, template_file, template_sha256, kv, measured, cache_ram=8192, context=65536):
    """measured: loaded_mib, peak_mib, process_floor_mib, qualification, qualification_status, notes (from the qualification run)."""
    return dict(
        display_name='Qwen3.8-27B-OBLITERATED Optimized', model_id='qwen3.8-27b-obliterated', family='obliterated', executable=str(executable),
        model=str(model), model_stat=model_stat, checkpoint_sha256=sha256, quantization='Q4_K_M', kv=kv, context=context, microbatch=512,
        argv=obliterated_argv(executable, model, context, kv, port, 'qwen3.8-27b-obliterated', template_file, cache_ram),
        chat_template_file=str(template_file), chat_template_sha256=template_sha256,
        sampling={}, request_defaults={}, reasoning=copy.deepcopy(OBLITERATED_REASONING), prefix_cache=cache_ram > 0, load_timeout=300,
        vram_loaded_mib=measured['loaded_mib'], minimum_load_free_bytes=(measured['peak_mib'] + 768) * 2 ** 20,
        minimum_process_device_bytes=int(measured['process_floor_mib']) * 2 ** 20,
        qualification=measured['qualification'], qualification_status=measured.get('qualification_status', 'experimental'), notes=measured.get('notes', ''),
        description=('Qwen3.8-27B-OBLITERATED Q4_K_M (Pliny, OBLITERATUS) with its embedded MTP head: native MTP draft depth 3, acceptance threshold 0.6, '
                     '65,536 context, 175 W.'))
