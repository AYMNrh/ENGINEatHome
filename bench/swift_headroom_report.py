"""Render the 256K headroom and tool-continuation evidence without inference."""
import argparse
import json
from pathlib import Path


def write(folder):
    stage = folder / 'headroom'
    data = json.loads((stage / 'summary.json').read_text())
    state = json.loads((folder / 'status.json').read_text())
    restored = json.loads((folder / 'restoration.json').read_text()) if (folder / 'restoration.json').is_file() else None
    lines = ['# Swift 1.5 / 256K capacity with 128K input and large output headroom', '',
             'RTX 2080 Ti 22 GB, Q4_0 target and draft KV, MTP3, Flash Attention, one sequence. '
             'All model weights including token embeddings and both caches are on CUDA. '
             'Batch 512, microbatch 128, draft microbatch 64. Thinking uses the checkpoint template with '
             'xhigh and reasoning-budget -1. Sampling is temperature 1, top-p 0.95, top-k 20, '
             'min-p 0, presence penalty 0, repeat penalty 1 and seed 42.', '',
             f'The initial rendered prompt has exactly {data["target_input"]:,} token IDs. '
             f'The total output allowance is {data["total_output_allowance"]:,}, with 256 tokens of '
             'additional allocation headroom. This output allowance covers both reasoning and final content. '
             'EOS is respected. The fixture asks for a complete Python log-processing CLI with validation, '
             'aggregation, tests and usage examples, following a synthetic archive.', '',
             '| Request | Actual input | Reused / fresh | Output allowance | Generated | '
             'Prefill tok/s | Decode tok/s | TTFT s | End-to-end s | Natural stop |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for name in ('cold', 'cached'):
        if name not in data:
            continue
        request = data[name]
        t = request['timings']
        lines.append(f'| {name} | {request["actual_input_tokens"]:,} | {t["cache_n"]:,} / {t["prompt_n"]:,} '
                     f'| {request["output_limit"]:,} | {request["output_tokens"]:,} '
                     f'| {t["prompt_per_second"]:.2f} | {t["predicted_per_second"]:.2f} '
                     f'| {request["ttft_seconds"]:.3f} | {request["wall_seconds"]:.2f} '
                     f'| {request["natural_stop"]} |')
    buffers = data['placement']['buffers']
    kv = buffers['_logged_allocation_events_bytes']['CUDA0_KV_bytes']
    lines += ['', '## KV allocation and memory observations', '',
              f'Main KV reserves {kv[0]/2**20:,.2f} MiB at initialization.'
              + (f' Draft KV reserves {sum(kv[1:])/2**20:,.2f} MiB.' if len(kv) > 1 else ''),
              f'Total logged KV is {buffers["CUDA0_KV_bytes"]/2**20:,.2f} MiB. '
              'The runtime sizes cache tensors for the configured capacity and allocates their buffers '
              'during context initialization. Filled positions increase as input and output are processed.', '',
              '| Stage | Process CUDA MiB | Device used MiB | Device free MiB |',
              '| --- | ---: | ---: | ---: |']
    for name, memory in data['memory'].items():
        gpu = memory['device']
        lines.append(f'| {name} | {memory["process_device_bytes"]/2**20:,.0f} '
                     f'| {gpu["used"]/2**20:,.0f} | {gpu["free"]/2**20:,.0f} |')
    lines += ['', 'Device memory is sampled and includes other GPU users. Process allocations are from '
              'nvidia-smi. Linux lacks a directly comparable WDDM shared-GPU-memory counter. '
              'Variation in workspace and graphs can change observed total VRAM despite a fixed KV reservation.', '',
              '## Tool continuations', '',
              'The model calls archive_lookup for B. The harness returns a deterministic record and digest. '
              'The model then calls verify_digest, receives a verification result, and must return all three '
              'original archive keys plus verified=true. These are local test tools. Every assistant message, '
              'including reasoning_content and tool_calls, is sent back with its corresponding tool response. '
              'Before each continuation, the rendered prompt is checked for exact prior reasoning text. '
              'A preserve_thinking=false control is also rendered to show whether the template switch matters.', '',
              '| Turn | Actual input | max_tokens | Nonempty thinking | Previous traces retained | Finish |',
              '| ---: | ---: | ---: | --- | --- | --- |']
    for turn in data.get('tools', []):
        retained = turn['retained_reasoning']
        lines.append(f'| {turn["turn"]} | {turn["actual_input_tokens"]} | {turn["max_tokens"]:,} '
                     f'| {turn["nonempty_reasoning"]} | {sum(retained)}/{len(retained)} '
                     f'| {turn["finish_reason"]} |')
    lines += ['', f'Fresh reasoning across tool turns: {data.get("reasoning_across_tools")}. '
              f'Final archive/tool-result recall correct: {data.get("tool_final_correct")}.', '',
              'Preservation means the prior reasoning trace is present in the model input. '
              'The final recall check separately measures whether the model uses the archived facts '
              'and tool results correctly. This is one controlled conversation. '
              'The preserve_thinking=false control did not change these prompts. The embedded template '
              'keeps assistant reasoning after the last user query even with that option disabled, '
              'which covers this uninterrupted tool sequence. This result does not establish retention '
              'across a new user query or compaction.', '',
              f'Run state: {state["status"]}. Error: {state.get("error")}.',
              f'Original server restoration: {restored["health"] if restored else "pending"}.', '',
              '## Reproduce', '',
              'Temporarily unload the idle original server using its verified PID, then run from the benchmark workspace.',
              '',
              '    PYTHONPATH=src python3 bench/swift_homelab.py --replace-pid <8080-pid> '
              '--live-audit live-audit --output headroom-fresh --headroom',
              '    python3 bench/swift_headroom_report.py headroom-fresh', '',
              'The controller restores the captured original command and verifies port 8080. Full prompt IDs, '
              'messages, model outputs, native timings, telemetry and tool prompt checks are retained in headroom/.', '']
    result = dict(summary=data, state=state, restoration=restored)
    proof = stage / 'server-accepted-output-allowance.json'
    if proof.is_file():
        result['server_accepted_limits'] = json.loads(proof.read_text())
    chat_proof = stage / 'server-accepted-chat-max-tokens.json'
    if chat_proof.is_file():
        result['server_accepted_chat_limits'] = json.loads(chat_proof.read_text())
    (folder / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    (folder / 'report.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    write(parser.parse_args().folder)
