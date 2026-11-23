"""Render the Obliterated profile and RAM-cache cards from docs/results/obliterated-20261123.json; no inference."""
from bench.render_cards import Card,load,GREEN,CYAN,AMBER,MUTED,WHITE,BORDER


def render():
    d=load('obliterated-20261123.json');kv=d['kv_comparison_63591_tokens'];f,q=kv['f16'],kv['q8_0'];cs=d['cache_seconds']
    c=Card('OBLITERATED Q4_K_M / 64K PROFILE','Qwen3.8-27B-OBLITERATED / native MTP3 / stock llama.cpp / RTX 2080 Ti 22GB / 175 W','01','23 NOV 2026')
    c.metric(0,'CONTEXT CAPACITY','64K','65,536 tokens, nothing larger',CYAN)
    c.metric(1,'KV PRECISION','F16','Q8_0 validated fallback')
    c.metric(2,'DECODE AT 63.6K',f'{f["decode_tps"]:.1f} tok/s','MTP3 / F16 KV',GREEN)
    c.metric(3,'MTP ACCEPTANCE','90%','47/52 drafts at 63.6K',CYAN)
    for x,title in ((70,'F16 VS Q8_0 KV / 63,591 INPUT TOKENS'),):c.text(x,375,title,25,WHITE,True)
    rows=[('DECODE / tok/s',f['decode_tps'],q['decode_tps'],45,'{:.1f}'),('PREFILL / tok/s',f['prefill_tps'],q['prefill_tps'],500,'{:.1f}'),('PEAK VRAM / MiB',f['peak_vram_mib'],q['peak_vram_mib'],22528,'{:,}')]
    for i,(title,a,b,limit,fmt) in enumerate(rows):
        y=470+i*140
        c.text(70,y-52,title,23,MUTED)
        c.bar(70,y,640,a,limit,GREEN,'F16  '+fmt.format(a))
        c.bar(800,y,640,b,limit,CYAN,'Q8_0 '+fmt.format(b))
    c.text(70,885,f'F16 kept 1,680 MiB free with VRAM flat (+26 MiB) over {f["soak_requests"]} requests; Q8_0 kept 3,378 MiB free.',22,GREEN)
    c.footer(['Selected F16: faster decode and all checks passed; Q8_0 passed the same checks as the fallback.',
              'Thinking off by default; Apply is explicit in Home Assistant and Swift stays the default.',
              'Soaks were 12 min (F16) and 8 min (Q8_0). Contexts above 65,536 are untested.'])
    c.save('obliterated-profile.png')

    c=Card('RAM PROMPT CACHE / RESTORE TIME','--cache-ram 8192 --no-cache-idle-slots / cold prompt versus cached follow-up','02','23 NOV 2026')
    c.metric(0,'OBLITERATED 63.6K','0.51 s','vs 158 s cold',GREEN)
    c.metric(1,'AFTER UNRELATED PROMPT','0.2 s','Restored from RAM pool',GREEN)
    c.metric(2,'SWIFT RESUME','0.2-0.5 s','vs 25-41 s cold',CYAN)
    c.metric(3,'SWIFT RAM COST','+2.1 GB','About 52K cached tokens',CYAN)
    o=cs['obliterated_63591'];t=cs['obliterated_12600_first_turn'];s=cs['swift_13k_to_20k']
    cases=[('OBLITERATED / 63,591 TOKENS',o['cold'],o['cached_followup'],'{:.2f} s'),
           ('OBLITERATED / 12.6K FIRST TURN',t['cold'],t['cached_range'][1],'{:.1f} s'),
           ('SWIFT / 13-20K (best cold, worst cached)',s['cold_range'][0],s['resume_range'][1],'{:.1f} s')]
    for i,(title,cold,warm,fmt) in enumerate(cases):
        y=465+i*135
        c.text(70,y-62,title,23,WHITE,True)
        c.bar(70,y,640,cold,160,AMBER,'Cold  '+fmt.format(cold))
        c.bar(800,y,640,warm,160,GREEN,'Cached '+fmt.format(warm))
    c.text(70,880,'Common scale 0 to 160 s. Caches die with the process: nothing is ever restored across models.',22,MUTED)
    c.footer(['Host available RAM stayed above 20 GB of 31 GB. No extra VRAM is used by the cache.',
              'A prompt sharing 25%+ of a small entry consumes it: slower re-read, never a wrong answer.',
              'Single-run timings per case, not a latency distribution.'])
    c.save('obliterated-ram-cache.png')

    i=d['isolation'];oc=d['opencode']
    c=Card('QUALIFICATION / ISOLATION / OPENCODE','Two clients, unique markers, sequential, interleaved and concurrent / real OpenCode 1.18.16','03','23 NOV 2026')
    c.metric(0,'CONTAMINATED',f'{i["contaminated"]} / {i["requests"]}','Responses across clients',GREEN)
    c.metric(1,'OPENCODE TASK','Passed','read, read, edit, bash',GREEN)
    c.metric(2,'PRODUCTION RUN',f'{oc["production_default_s"]:.1f} s','Default thinking, real key',CYAN)
    c.metric(3,'SERVING TESTS',f'{d["tests_passing"]} pass','Homelab suite',CYAN)
    items=[('CLIENT ISOLATION','Two clients, markers, interleaved + concurrent + issue repro: 0 mix-ups.'),
           ('UPSTREAM BUG STILL OPEN','llama.cpp #27148; fix #27624 unmerged. Hardening flag --no-cache-idle-slots.'),
           ('FAILED-LOAD RECOVERY','Forced OOM: 503, previous profile restored, one process, inference works.'),
           ('MODEL SWITCHING','Cache never crosses models; wrong-model request gets 409 model_not_active.'),
           ('HOME ASSISTANT','Select never switches; Apply explicit; Swift default; Obliterated never auto-starts.')]
    for n,(title,description) in enumerate(items):
        y=365+n*105;c.rect((52,y,1548,y+90),outline=BORDER)
        c.text(70,y+10,title,23,GREEN,True);c.text(70,y+45,description,23,WHITE)
    c.footer(['0 of 400 is evidence, not proof: about 2.5% upper bound (95%) on 120 cached 12K-scale requests.',
              'Inkling profiles are unchanged and still use the default cache settings.',
              'Production Home Assistant UI was validated via an isolated instance, not clicked by hand.'])
    c.save('obliterated-qualification.png')


if __name__=='__main__':render()
