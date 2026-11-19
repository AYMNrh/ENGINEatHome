"""Render the selected Swift serving configuration and final measured verdict; no inference."""
from bench.render_cards import Card,load,GREEN,CYAN,AMBER,MUTED,WHITE,BORDER


def render():
    data=load('swift-serving-20261119.json');routing=data['routing_screen'];serving=data['serving']
    c=Card('SWIFT 1.5 / SELECTED WINNER','Qwen3.8-27B GSQ-RCO IQ3_S MTP / stock llama.cpp / RTX 2080 Ti 22GB','01','19 NOV 2026')
    c.metric(0,'PRODUCTION CAPACITY','256K','262,144 total tokens',CYAN)
    c.metric(1,'KV / DRAFTING','Q4 / MTP3','Weights + KV on GPU')
    c.metric(2,'GPU POWER LIMIT','175 W','No routing patch',CYAN)
    c.metric(3,'THINKING','xhigh','No reasoning budget',GREEN)
    c.text(70,375,'FILLED INPUT / MTP3',25,WHITE,True)
    for x,title in ((560,'COLD PREFILL / tok/s'),(1060,'SUSTAINED DECODE / tok/s')):c.text(x,375,title,23,MUTED)
    for i,row in enumerate(data['filled_context_mtp3']):
        y=453+i*126
        c.text(70,y-8,f'{row["context"]//1024}K capacity',29,WHITE,True)
        c.text(70,y+32,f'{row["input"]:,} actual input',22,MUTED)
        c.bar(560,y+32,390,row['prefill'],350,CYAN,f'{row["prefill"]:.1f}')
        c.bar(1060,y+32,390,row['sustained'],30,GREEN,f'{row["sustained"]:.2f}')
    c.text(70,851,'Selected for production: long context + GPU residency + validated stock runtime.',24,GREEN)
    c.footer(['One historical cold/replay sample per profile; 512-token sustained decode window.',
              'Swift coding: 4/4 on 5070 Ti, 3/4 on 2080 Ti. Strata retains the broader 18/18 result.',
              'Recommendation is deployment-specific; MTP coding quality remains unmeasured.'])
    c.save('swift-selected-winner.png')

    c=Card('STOCK RETAINED / ROUTING CLOSED','Same Swift model, settings, prompt IDs and 175 W / three paired short screens','02','19 NOV 2026')
    c.metric(0,'PATCH DECODE GAIN',f'+{routing["median_decode_gain_percent"]:.2f}%','Below 5% stop/go rule',AMBER)
    c.metric(1,'ACTUAL INPUT','4,096','128K capacity',CYAN)
    c.metric(2,'GENERATED','256 tokens','Each timed sample',CYAN)
    c.metric(3,'STOCK FIXTURE','5 / 5','Two passing observations')
    for i,r in enumerate(routing['profiles']):
        y=420+i*180
        c.text(70,y,'STOCK' if r['profile']=='stock' else 'IQ4 ROUTING PATCH',29,WHITE,True)
        c.bar(70,y+93,760,r['decode_tps'],40,GREEN if i==0 else CYAN,f'Median decode: {r["decode_tps"]:.2f} tok/s')
        c.text(950,y+12,f'{r["request_seconds"]:.3f} s request',29,WHITE)
        c.text(950,y+61,f'{r["cached_ttft_seconds"]:.3f} s cached TTFT',24,MUTED)
        c.text(950,y+101,f'{r["vram_peak_gib"]:.3f} GiB peak',24,MUTED)
    c.text(70,825,'Decision: stop patch investigation; serve the validated stock engine.',27,GREEN,True)
    c.footer(['Short screen, not filled-128K throughput. Timing precedes numerical instrumentation.',
              'Paired gains: +1.04%, +0.87%, +0.83%. No expensive patched quality phase launched.',
              'No new kernel optimization or benchmark sweep in the production-serving phase.'])
    c.save('swift-stock-routing-verdict.png')

    c=Card('SWIFT / PRIVATE PRODUCTION API','Stock Swift 1.5 Qwen3.8-27B IQ3_S MTP / RTX 2080 Ti 22GB / systemd','03','19 NOV 2026')
    c.metric(0,'ACCESS','Tailscale','Scoped bearer keys',CYAN)
    c.metric(1,'LIFECYCLE','On demand','60-minute idle unload')
    c.metric(2,'LOADED DEVICE',f'{serving["loaded_sample_vram_gib"]:.2f} GiB','Short functional check',CYAN)
    c.metric(3,'FREE DEVICE',f'{serving["loaded_sample_free_gib"]:.2f} GiB','Sample, not exact peak',CYAN)
    features=[('API + AGENT PROTOCOL','Autoload, streaming reasoning, tool calls and continuations: passed.'),
              ('SAFE MODEL LIFECYCLE','Active/queued requests + persisted leases block unload and restart.'),
              ('CRASH RECOVERY','Owned backend recovery + gateway systemd restart: passed.'),
              ('USAGE ANALYTICS','SQLite metadata, client/model filters, cache subsets counted once.'),
              ('HOME ASSISTANT','25 entities registered; REST polling confirmed; native dashboard.')]
    for i,(title,description) in enumerate(features):
        y=365+i*105;c.rect((52,y,1548,y+90),outline=BORDER)
        c.text(70,y+10,title,23,GREEN,True);c.text(70,y+45,description,23,WHITE)
    c.footer(['No prompts, completions, reasoning text or API keys stored in the usage ledger.',
              'Three async regressions + short live checks. No new long-context benchmark or reboot.',
              'HA UI buttons and browser interactions were not manually exercised. Keys excluded.'])
    c.save('swift-production-serving.png')


if __name__=='__main__':render()
