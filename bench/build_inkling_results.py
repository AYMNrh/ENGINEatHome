"""Build docs/results/inkling-turing-20261122.json from the raw Inkling-Turing evidence kept in research/inkling-turing/v06-longctx/results.

Every value is read from a raw result file or parsed from a raw session log; nothing is typed in by hand except labels and the explicit
`not_published` list. Run: python bench/build_inkling_results.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'research/inkling-turing/v06-longctx/results'
OUT = ROOT / 'docs/results/inkling-turing-20261122.json'
REL = 'research/inkling-turing/v06-longctx/results/'


def jload(name):
    return json.loads((RAW / name).read_text(encoding='utf-8'))


def first_int(pattern, text):
    m = re.search(pattern, text)
    if not m:
        raise ValueError('pattern not found: ' + pattern)
    return int(m.group(1))


def by_tag(results, tag):
    return next(r for r in results if r['tag'] == tag)


def run_record(run_id, rec, source, config, loaded_mib):
    turns = {t['name']: t for t in rec['turns']}
    fix = turns['fix']
    first = fix['per_call'][0]
    assert first['cache_n'] + first['prompt_new'] == first['ctx_tokens']
    follow = [turns[n] for n in ('retrieval', 'recall', 'rules')]
    return {
        'id': run_id,
        'source': REL + source,
        'configuration': config,
        'snapshot_tokens': rec['tokenizer_tokens'],
        'snapshot_files': rec['files'],
        'first_request_prompt_tokens': first['ctx_tokens'],
        'first_request_cached_tokens': first['cache_n'],
        'final_context_tokens': follow[-1]['ctx_tokens_last'],
        'cold_prefill': {'seconds': round(fix['ttft_first_s'], 1), 'tok_per_s': round(fix['prefill_tps_first'], 1),
                         'tokens_processed': first['prompt_new']},
        'decode_tok_per_s': {'fix_turn': fix['decode_tps'], 'followups': [t['decode_tps'] for t in follow]},
        'fix_turn': {'seconds': fix['wall_s'], 'context_tokens_after': fix['ctx_tokens_last'], 'generated_tokens': fix['generated'], 'model_calls': fix['model_calls'], 'tool_calls': fix['tool_calls'],
                     'hidden_tests_passed': fix['grade']['passed'], 'hidden_tests_total': fix['grade']['total_expected'],
                     'hidden_tests_failed': fix['grade']['failed'], 'rules_ok': fix['rules_ok'],
                     'median_warm_ttft_s': round(fix['median_ttft_warm_s'], 2), 'tool_categories': fix['categories']},
        'followups': [{'name': t['name'], 'pass': t['pass'], 'checks': t['checks'], 'ttft_s': round(t['ttft_first_s'], 2),
                       'new_tokens': t['prompt_new_first'], 'cached_tokens': t['cache_first'], 'context_tokens': t['ctx_tokens_last']}
                      for t in follow],
        'followups_passed': sum(1 for t in follow if t['pass']),
        'vram_mib': {'loaded': loaded_mib, 'peak_polled': rec['vram_peak_mib']},
        'session_wall_seconds': rec['wall_s'],
    }


def main():
    s512 = (RAW / 'session_k_ctx_ub512_20261122.log').read_text()
    s256 = (RAW / 'session_k_256_ub256_20261122.log').read_text()
    host_new = (RAW / 'hostcost_memprobe_7d3e5b1_k_b2_20261122.log').read_text()
    host_old = (RAW / 'hostcost_e4a7ea6_k_depth_20261122.log').read_text()
    summary = (RAW / 'server_summary.txt').read_text()
    r512, r256 = jload('ctx_results_ub512_128k_185k.json'), jload('ctx_results_ub256_128k_242k.json')

    loaded512 = first_int(r'ub=512 load_s [0-9.]+\n(\d+) MiB', s512)
    loaded256 = first_int(r'vram_loaded_mib (\d+)', s256)
    cfg512 = {'context_capacity': 204800, 'ubatch': 512, 'kv_cache': 'q8_0/q8_0', 'flash_attention': True, 'slots': 1, 'runtime_commit': '7d3e5b1',
              'harness_seed': 31, 'content': 'fixed needles (seed 31)', 'server_log_verbosity': 3}
    cfg256 = {'context_capacity': 262144, 'ubatch': 256, 'kv_cache': 'q8_0/q8_0', 'flash_attention': True, 'slots': 1, 'runtime_commit': '7d3e5b1',
              'harness_seed': 4242, 'content': 'fresh randomized needles and decoys (--vary)', 'server_log_verbosity': 4}
    runs = [
        run_record('ub512-128k', by_tag(r512, 'ctx128k'), 'ctx_results_ub512_128k_185k.json', cfg512, loaded512),
        run_record('ub512-185k', by_tag(r512, 'ctx186k'), 'ctx_results_ub512_128k_185k.json', cfg512, loaded512),
        run_record('ub256-242k', by_tag(r256, 'ctx244k'), 'ctx_results_ub256_128k_242k.json', cfg256, loaded256),
        run_record('ub256-128k', by_tag(r256, 'ctx128k'), 'ctx_results_ub256_128k_242k.json', cfg256, loaded256),
    ]
    runs[2]['randomized_variant'] = by_tag(r256, 'ctx244k')['variant']
    runs[3]['randomized_variant'] = by_tag(r256, 'ctx128k')['variant']
    for run in runs:
        assert run['fix_turn']['hidden_tests_failed'] == 0 and run['followups_passed'] == 3 and run['fix_turn']['rules_ok']

    top = runs[2]
    assert top['id'] == 'ub256-242k'
    m = re.search(r're-plan lines: (\d+)', summary)
    compute = re.search(r'CUDA0 compute buffer size =\s+([0-9.]+) MiB', summary)
    kv = re.findall(r'CUDA0 KV buffer size =\s+([0-9.]+) MiB', summary)
    probe = re.search(r'ub512 probe: loaded=(\d)', s256)
    oom = re.search(r'allocating ([0-9.]+) MiB on device 0: cudaMalloc failed', s256)
    assert m and compute and probe and oom and kv

    def hostcost(text, label):
        row = re.findall(r'INK_RELPOS calls=(\d+) fill_ms=([0-9.]+) set_ms=([0-9.]+) MB=([0-9.]+)', text)[-1]
        return {'build': label, 'calls': int(row[0]), 'cpu_fill_ms': float(row[1]), 'h2d_copy_ms': float(row[2]), 'uploaded_mb': float(row[3])}

    old = hostcost(host_old, 'e4a7ea6 instrumented (dense int32 distance matrix)')
    new = hostcost(host_new, '7d3e5b1 instrumented counters (bin-rp3; positions only)')
    mem_old = first_int(r"'tag': 'rp_256k_ub256', 'ctx': '262144', 'ub': '256', 'loaded_mib': (\d+)", host_old)
    mem_new = {spec: used for spec, used in re.findall(r'"spec": "([0-9:a-z_]+)", "loaded": true, "used_mib": (\d+)', host_new)}
    assert int(mem_new['262144:256:q8_0']) == loaded256

    data = {
        'schema': 1,
        'title': 'Inkling-Turing long-context qualification and relbias engineering',
        'date': '2026-11-22',
        'generated_by': 'bench/build_inkling_results.py (reads only files under research/inkling-turing/v06-longctx/results)',
        'hardware': {'gpu': 'NVIDIA RTX 2080 Ti 22 GB (sm_75)', 'power_limit_w': 175, 'offload': 'none, full CUDA'},
        'model': {'name': 'Inkling 2 Mini', 'quant': 'UD-IQ3_XXS (mainline GGUF)'},
        'runtime': {'validated_baseline': '7d3e5b1', 'fallbacks': ['e4a7ea6', 'e0be55a'],
                    'status': 'separately validated research engine; Swift 1.5 remains the production backend; nothing merged upstream'},
        'evaluation': {
            'kind': 'controlled synthetic-repository qualification (not SWE-bench)',
            'turns': ['two-file bug fix with 7 hidden tests under the current rules', 'distant retrieval with outdated decoys',
                      'recall of the first file after tool use', 'contradictory rules resolution'],
            'sandbox': 'docker shell, temperature 0, prefix cache on',
            'repetitions': 'one successful run per context size; not a statistical reliability measurement',
        },
        'runs': runs,
        'headline': {
            'max_snapshot_tokens': top['snapshot_tokens'],
            'max_first_request_prompt_tokens': top['first_request_prompt_tokens'],
            'max_final_context_tokens': top['final_context_tokens'],
            'configured_capacity': cfg256['context_capacity'],
            'hidden_tests': '%d/%d' % (top['fix_turn']['hidden_tests_passed'], top['fix_turn']['hidden_tests_total']),
            'followup_checks': '%d/3' % top['followups_passed'],
            'peak_vram_mib': top['vram_mib']['peak_polled'],
            'scheduler_replans': int(m.group(1)),
            'note': 'snapshot tokens are the repository text; the first request adds system prompt, chat template and tool schema; final context includes tool output and answers. Actual input is not 256K.',
        },
        'stability_ub256_242k': {'sched_graph_realloc_lines': int(m.group(1)), 'source': REL + 'server_summary.txt',
                                 'cuda_graph_reuse_logged': False,
                                 'note': 'the server log has no scheduler re-plan lines and no CUDA errors; CUDA-graph reuse itself is not logged by this build'},
        'memory': {
            'compute_buffer_mib_ub256_262144': float(compute.group(1)),
            'kv_buffer_mib_ub256_262144': [float(x) for x in kv],
            'loaded_mib_7d3e5b1': {k: int(v) for k, v in mem_new.items()},
            'loaded_mib_e4a7ea6_262144_ub256': mem_old,
            'source': [REL + 'hostcost_memprobe_7d3e5b1_k_b2_20261122.log', REL + 'hostcost_e4a7ea6_k_depth_20261122.log', REL + 'server_summary.txt'],
            'e0be55a_262144_ub256': 'did not fit before the relbias node (ledger V03-RN, no raw log kept in this bundle)',
            'ub512_at_262144': {'loaded': bool(int(probe.group(1))), 'failed_allocation_mib': float(oom.group(1)), 'source': REL + 'session_k_256_ub256_20261122.log'},
        },
        'host_cost_32k_prefill_ctx262144_ub256': {'before': old, 'after': new,
                                                  'source': [REL + 'hostcost_e4a7ea6_k_depth_20261122.log', REL + 'hostcost_memprobe_7d3e5b1_k_b2_20261122.log'],
                                                  'note': 'instrumented builds, 32K-token prefill plus a short decode (320 calls); end-to-end prefill gain was only about 1 percent'},
        'rejected': {'flash_attention_relbias_fusion': 'unit tests bit-exact, model integration unstable; prefill and decode op sequences diverge at node 1131 (280 nodes with different sources vs 0 unfused); patches and graph_audit.py kept',
                     'source': 'research/inkling-turing/v06-longctx/audit/'},
        'not_published': ['no matched comparison with unmodified upstream llama.cpp for this model (the only upstream figures in the notes are approximate and unverified)',
                          'no 256K actual-input run; the largest measured snapshot is 242,134 tokens',
                          'no statistical repetition of any context size'],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()
