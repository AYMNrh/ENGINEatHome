"""Short serving checks against localhost. Never persist text or credentials."""
import asyncio
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from aiohttp import ClientSession,ClientTimeout


async def main(recovery=False):
    root=Path(__file__).resolve().parent
    keys=json.loads((root/'credentials.json').read_text())
    config=json.loads((root/'config.json').read_text());base='http://127.0.0.1:8080'
    results={'started':time.time(),'checks':{},'requests':[]}
    async with ClientSession(timeout=ClientTimeout(total=300)) as http:
        async def call(method,path,body=None,who='admin',expected=200):
            headers={'Authorization':'Bearer '+keys[who]} if who else {}
            async with http.request(method,base+path,json=body,headers=headers) as response:
                value=await response.json()
                if response.status!=expected:raise RuntimeError(f'{path}: HTTP {response.status}, code='+str(value.get('error',{}).get('code')))
                return value
        def body(text,**kw):
            return dict(model=config['model_id'],messages=[dict(role='user',content=text)],
                        max_tokens=32,temperature=0,**kw)
        async def small():
            value=await call('POST','/v1/chat/completions',body('Reply with READY.',chat_template_kwargs={'enable_thinking':False}),who='python')
            assert value['choices'][0]['message'].get('content')
            results['requests'].append({'kind':'nonstream','usage':value.get('usage'),'timings':value.get('timings')})
            return value
        await call('GET','/v1/models',who=None,expected=401)
        await call('POST','/admin/load',{},who='python',expected=403)
        await call('GET','/v1/models',who='python')
        results['checks']['authentication_and_scopes']=True
        await small()
        status=await call('GET','/admin/status');assert status['model']['state']=='ready'
        results['loaded_status']=status
        results['checks']['first_request_autoload_and_generation']=True
        if not recovery:
            repeated=await small()
            assert repeated.get('usage',{}).get('prompt_tokens_details',{}).get('cached_tokens',0)>0
            results['checks']['reported_prefix_cache_reuse']=True
        if recovery:
            unit='engineathome-api.service'
            def main_pid():return int(subprocess.check_output(['systemctl','show',unit,'-p','MainPID','--value']))
            parent=main_pid()
            children=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
            owned=[int(pid) for pid in children if Path(f'/proc/{pid}/exe').resolve().name=='llama-server']
            assert len(owned)==1
            os.kill(owned[0],signal.SIGKILL)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                status=await call('GET','/admin/status')
                if status['model']['state']=='failed':break
                await asyncio.sleep(.2)
            assert status['model']['state']=='failed'
            await small()
            results['checks']['owned_backend_crash_autoload_recovery']=True
            before=await call('GET','/analytics/totals',who='analytics')
            subprocess.run(['sudo','-n','systemctl','kill','--kill-whom=main','--signal=SIGKILL',unit],check=True)
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                try:
                    async with http.get(base+'/healthz') as response:
                        if response.status==200 and main_pid()!=parent:break
                except OSError:pass
                await asyncio.sleep(.2)
            assert main_pid()!=parent
            status=await call('GET','/admin/status');assert status['model']['state']=='unloaded'
            after=await call('GET','/analytics/totals',who='analytics')
            assert before['periods']==after['periods']
            await small();await call('POST','/admin/unload',{})
            results['checks']['gateway_systemd_crash_recovery_and_ledger_persistence']=True
            results['finished']=time.time();results['final_status']=await call('GET','/admin/status')
            (root/'validation-recovery.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps({'checks':results['checks'],'seconds':results['finished']-results['started']}))
            return
        async with http.post(base+'/v1/chat/completions',json=body('Explain briefly why a mutex prevents concurrent writes.',stream=True,stream_options={'include_usage':True}),headers={'Authorization':'Bearer '+keys['python']}) as response:
            assert response.status==200
            await call('POST','/admin/unload',{},expected=409)
            raw=await response.text();events=[]
            for line in raw.splitlines():
                if line.startswith('data: ') and line[6:]!='[DONE]':events.append(json.loads(line[6:]))
            assert 'data: [DONE]' in raw
            assert any(c.get('delta',{}).get('reasoning_content') for e in events for c in e.get('choices',[]))
            assert any(e.get('usage') for e in events)
            results['requests'].append({'kind':'stream','usage':next(e['usage'] for e in reversed(events) if e.get('usage'))})
        results['checks']['reasoning_stream_usage_and_active_unload_guard']=True
        cancelled=body('Explain in detail how databases implement transactional isolation.',stream=True)
        cancelled['max_tokens']=256
        async with http.post(base+'/v1/chat/completions',json=cancelled,headers={'Authorization':'Bearer '+keys['python']}) as response:
            assert response.status==200
            await response.content.readany()
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            status=await call('GET','/admin/status')
            if status['model']['active_requests']==0 and status['last_inference']['status']!='complete':break
            await asyncio.sleep(.1)
        assert status['model']['active_requests']==0
        assert status['last_inference']['status'] in ('interrupted','client_disconnected')
        results['checks']['client_disconnect_is_incomplete']=True
        tool=dict(type='function',function=dict(name='get_status',description='Return device state',parameters=dict(type='object',properties={'device':{'type':'string'}},required=['device'])))
        value=body('Call get_status for device lamp.',tools=[tool],tool_choice={'type':'function','function':{'name':'get_status'}},chat_template_kwargs={'enable_thinking':False})
        value['max_tokens']=256
        answer=await call('POST','/v1/chat/completions',value,who='python')
        message=answer['choices'][0]['message'];calls=message.get('tool_calls',[])
        assert calls and calls[0]['function']['name']=='get_status'
        assert json.loads(calls[0]['function']['arguments'])['device']=='lamp'
        continuation=dict(value);continuation.pop('tool_choice');continuation['max_tokens']=32
        continuation['messages']=value['messages']+[message,dict(role='tool',tool_call_id=calls[0]['id'],content='{"state":"on"}'),dict(role='user',content='Reply with the state, without another tool call.')]
        final=await call('POST','/v1/chat/completions',continuation,who='python')
        assert final['choices'][0]['message'].get('content')
        results['checks']['tool_call_and_continuation']=True
        lease=await call('POST','/v1/leases',{'ttl_seconds':60},who='python')
        await call('POST','/admin/unload',{},expected=409)
        await call('PUT','/v1/leases/'+lease['lease_id'],{'ttl_seconds':120},who='python')
        await call('DELETE','/v1/leases/'+lease['lease_id'],who='python')
        results['checks']['lease_create_renew_release_and_unload_guard']=True
        await call('POST','/admin/restart',{})
        await small()
        results['checks']['restart_and_generation']=True
        await call('PATCH','/admin/settings',{'auto_unload':False})
        assert not (await call('GET','/admin/status'))['auto_unload']
        await call('PATCH','/admin/settings',{'auto_unload':True,'idle_seconds':3600})
        for period in ('hour','day','week','month'):
            value=await call('GET','/analytics/summary?bucket='+period+'&client=python&model='+config['model_id'],who='analytics')
            assert value['rows']
        results['checks']['settings_and_analytics_filters']=True
        await call('POST','/admin/unload',{})
        assert (await call('GET','/admin/status'))['model']['state']=='unloaded'
        results['checks']['unload_keeps_gateway_available']=True
        results['final_status']=await call('GET','/admin/status')
    results['finished']=time.time()
    (root/'validation-live.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'checks':results['checks'],'seconds':results['finished']-results['started']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recovery',action='store_true')
    asyncio.run(main(parser.parse_args().recovery))
