import asyncio
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from aiohttp import web,ClientSession,ClientPayloadError
from gateway import create_app
from lifecycle import Lifecycle,Unavailable
import profiles as registry

SWIFT_SPEC=registry.SWIFT_REASONING
INKLING_SPEC=registry.INKLING_REASONING


def make_profiles(port):
    base=dict(model_stat={},executable='/bin/true',model='/nonexistent',checkpoint_sha256='x',minimum_load_free_bytes=0,minimum_process_device_bytes=0)
    return {
        'swift':dict(base,display_name='Swift Production',family='swift',model_id='fixture-swift',context=262144,kv='q4_0',microbatch=128,
                     sampling=dict(temperature=1.0,top_p=.95),request_defaults=dict(enable_thinking=True,preserve_thinking=True),
                     reasoning=dict(SWIFT_SPEC),argv=['x','-c','262144','-ub','128']),
        'inkling-balanced':dict(base,display_name='Inkling Balanced',family='inkling',model_id='fixture-balanced',context=204800,kv='q8_0',microbatch=512,
                                reasoning=dict(INKLING_SPEC),argv=['x','-c','204800','-ub','512','-ctk','q8_0','-ctv','q8_0']),
        'inkling-maximum':dict(base,display_name='Inkling Maximum',family='inkling',model_id='fixture-maximum',context=262144,kv='q8_0',microbatch=256,
                               reasoning=dict(INKLING_SPEC),argv=['x','-c','262144','-ub','256','-ctk','q8_0','-ctv','q8_0'])}


class Base(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.held=asyncio.Event();self.release=asyncio.Event();self.entered=0;self.seen=[]
        self.usage=dict(prompt_tokens=100,completion_tokens=20,total_tokens=120,
                        prompt_tokens_details=dict(cached_tokens=80),completion_tokens_details=dict(reasoning_tokens=5))
        async def backend(request):
            body=await request.json();self.seen.append(body);text=body.get('messages',[{}])[0].get('content','')
            if text=='hold':
                self.entered+=1
                if self.entered==2:self.held.set()
                await self.release.wait()
            if not body.get('stream'):
                return web.json_response(dict(choices=[dict(message=dict(content='PRIVATE_OUTPUT'))],usage=self.usage))
            response=web.StreamResponse(headers={'Content-Type':'text/event-stream'});await response.prepare(request)
            first=dict(choices=[dict(delta=dict(reasoning_content='PRIVATE_REASONING'),finish_reason=None)])
            await response.write(('data: '+json.dumps(first)+'\n\n').encode())
            if text=='break':
                request.transport.close();return response
            final=dict(choices=[],usage=self.usage)
            for _ in range(2):await response.write(('data: '+json.dumps(final)+'\n\n').encode())
            await response.write(b'data: [DONE]\n\n');await response.write_eof();return response
        server=web.Application();server.router.add_post('/v1/chat/completions',backend)
        server.router.add_get('/slots',lambda request:web.json_response([dict(is_processing=False)]))
        self.backend=web.AppRunner(server,access_log=None);await self.backend.setup()
        site=web.TCPSite(self.backend,'127.0.0.1',0);await site.start();port=site._server.sockets[0].getsockname()[1]
        test=self
        class Fake(Lifecycle):
            def __init__(self,*args):
                super().__init__(*args);self.loads=0;self.starts=[];self.terminated=0;self.fail=set();self.running=0;self.overlap=False
            async def _start(self,key,probe=False):
                self.running+=1
                if self.running>1:self.overlap=True
                try:
                    await asyncio.sleep(.02);self.starts.append((key,probe))
                    self.state='switching' if self.switch else 'loading'
                    if key in self.fail:
                        await self._terminate();self.state='failed';self.last_error='fixture_load_failure';self.active_key=None;self.next_retry[key]=time.monotonic()+15   # as the real _start does
                        raise Unavailable('fixture_load_failure')
                    self.loads+=1;self.state='ready';self.active_key=key;self.process=SimpleNamespace(returncode=None,pid=123)
                    self.loaded_at=time.monotonic();self.load_seconds=.01;self.activity=time.monotonic()
                finally:self.running-=1
            async def _terminate(self):self.terminated+=1;self.process=None;self.loaded_at=None
            def _static_preflight(self,key):pass
            async def sample(self):self.gpu=dict(used=1,free=2**30,power_limit_watts=175);self.gpu_at=time.time()
        self.Fake=Fake
        keys={hashlib.sha256(key.encode()).hexdigest():dict(client=client,scopes=scopes) for key,client,scopes in
              [('infer-secret','python',['inference']),('other-secret','opencode',['inference']),('control-secret','admin',['control','analytics'])]}
        self.config=dict(keys=keys,database=str(Path(self.tmp.name)/'usage.db'),api_base_url='http://private/v1',backend_port=port,
                    backend_url=f'http://127.0.0.1:{port}',timezone='UTC',request_timeout=10,max_active_requests=8,
                    profiles=make_profiles(port),default_profile='swift')
        self.app=create_app(self.config,manager_factory=Fake);self.runner=web.AppRunner(self.app,access_log=None,handler_cancellation=True)
        await self.runner.setup();site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]);self.client=ClientSession()
        self.infer={'Authorization':'Bearer infer-secret'};self.other={'Authorization':'Bearer other-secret'};self.admin={'Authorization':'Bearer control-secret'}
        self.manager=self.app['manager']

    async def asyncTearDown(self):
        self.release.set();await self.client.close();await self.runner.cleanup();await self.backend.cleanup();self.tmp.cleanup()

    async def post(self,path,headers=None,**body):
        async with self.client.post(self.url+path,headers=headers or self.admin,json=body) as response:
            return response.status,await response.json(content_type=None)

    async def chat(self,model=None,headers=None,**extra):
        body=dict(messages=[dict(role='user',content='hi')],**extra)
        if model:body['model']=model
        async with self.client.post(self.url+'/v1/chat/completions',headers=headers or self.infer,json=body) as response:
            return response.status,dict(response.headers),await response.json(content_type=None)

    async def status(self):
        async with self.client.get(self.url+'/admin/status',headers=self.admin) as response:return await response.json()

    async def patch(self,**body):
        async with self.client.patch(self.url+'/admin/settings',headers=self.admin,json=body) as response:return response.status,await response.json()


class ProfileRegistryTests(unittest.TestCase):
    def test_legacy_single_model_config_becomes_one_swift_profile(self):
        legacy=dict(model_id='swift-1.5',argv=['/bin/server','-c','262144','-ub','128','-ctk','q4_0','-ctv','q4_0'],context=262144,sampling=dict(seed=42))
        config=registry.normalize(legacy)
        self.assertEqual(list(config['profiles']),['swift'])
        self.assertEqual(config['default_profile'],'swift')
        self.assertEqual((config['profiles']['swift']['kv'],config['profiles']['swift']['microbatch']),('q4_0',128))
        self.assertEqual(config['profiles']['swift']['request_defaults'],dict(enable_thinking=True,preserve_thinking=True))

    def test_invalid_registries_are_rejected(self):
        good=make_profiles(1)
        with self.assertRaises(registry.ProfileError):registry.normalize(dict(profiles=good,default_profile='missing'))
        duplicate=json.loads(json.dumps(good));duplicate['inkling-maximum']['model_id']='fixture-swift'
        with self.assertRaises(registry.ProfileError):registry.normalize(dict(profiles=duplicate,default_profile='swift'))
        bad_default=json.loads(json.dumps(good));bad_default['swift']['reasoning']['default']='ultra'
        with self.assertRaises(registry.ProfileError):registry.normalize(dict(profiles=bad_default,default_profile='swift'))
        no_reasoning=json.loads(json.dumps(good));del no_reasoning['swift']['reasoning']
        with self.assertRaises(registry.ProfileError):registry.normalize(dict(profiles=no_reasoning,default_profile='swift'))
        with self.assertRaises(registry.ProfileError):registry.normalize(dict(profiles={'Bad Key':good['swift']},default_profile='Bad Key'))

    def test_add_profiles_keeps_a_tuned_swift_profile_and_refreshes_inkling(self):
        import add_profiles
        tuned = make_profiles(1)
        tuned['swift']['load_timeout'] = 777
        tuned['swift']['minimum_load_free_bytes'] = 123
        base = dict(profiles=tuned, default_profile='swift', backend_port=18081)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(base))
            merged = add_profiles.merge(path, dry_run=True)
        self.assertEqual((merged['profiles']['swift']['load_timeout'], merged['profiles']['swift']['minimum_load_free_bytes']), (777, 123))
        self.assertEqual(sorted(merged['profiles']), ['inkling-balanced', 'inkling-maximum', 'swift'])
        argv = merged['profiles']['inkling-maximum']['argv']
        self.assertEqual((argv[argv.index('-c') + 1], argv[argv.index('-ub') + 1], argv[argv.index('-ctk') + 1]), ('262144', '256', 'q8_0'))
        self.assertNotIn('512', [argv[argv.index('-ub') + 1]])   # ub 512 does not allocate at 262,144
        balanced = merged['profiles']['inkling-balanced']['argv']
        self.assertEqual((balanced[balanced.index('-c') + 1], balanced[balanced.index('-ub') + 1]), ('204800', '512'))
        self.assertEqual(merged['default_profile'], 'swift')

    def test_public_description_has_no_paths_or_checksums(self):
        public=registry.public_profile('inkling-maximum',registry.normalize(dict(profiles=make_profiles(1),default_profile='swift'))['profiles']['inkling-maximum'])
        text=json.dumps(public)
        for private in ('/nonexistent','/bin/true','checkpoint'):self.assertNotIn(private,text)
        self.assertEqual((public['context'],public['microbatch'],public['kv_cache']),(262144,256,'q8_0'))

    def test_effort_resolution_matches_each_template(self):
        swift,inkling=SWIFT_SPEC,INKLING_SPEC
        self.assertEqual(registry.resolve_effort(swift,None)['kwargs'],{'reasoning_effort':'xhigh'})
        self.assertEqual(registry.resolve_effort(swift,'none')['kwargs'],{'enable_thinking':False})
        self.assertEqual(registry.resolve_effort(inkling,None)['kwargs'],{'reasoning_effort':'high'})
        self.assertEqual(registry.resolve_effort(inkling,'max')['label'],'xhigh')
        self.assertEqual(registry.resolve_effort(inkling,0.5)['kwargs'],{'reasoning_effort':0.5})
        self.assertEqual(registry.resolve_effort(inkling,None,'low')['source'],'dashboard')
        for spec,value in ((swift,'high'),(swift,'minimal'),(swift,0.5),(inkling,'ultra'),(inkling,1.5),(inkling,True)):
            with self.assertRaises(registry.ReasoningError):registry.resolve_effort(spec,value)

    def test_launch_facts_parse_short_flags_and_negative_values(self):
        argv=['x','-m','/p/model.gguf','-c','204800','-ub','512','-ctk','q8_0','-ctv','q8_0','-fa','on','--reasoning-budget','-1','--jinja']
        facts=dict(registry.launch_facts(argv))
        self.assertEqual((facts['Context'],facts['KV cache'],facts['Batch / microbatch']),('204,800 tokens','q8_0','default / 512'))
        self.assertEqual(facts['Model file'],'model.gguf')


class ServingTests(Base):
    async def test_lifecycle_auth_concurrent_load_active_guard_and_leases(self):
        async with self.client.get(self.url+'/v1/models') as response:self.assertEqual(response.status,401)
        async with self.client.post(self.url+'/admin/load',headers=self.infer,json={}) as response:self.assertEqual(response.status,403)
        async def run():
            async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'messages':[{'role':'user','content':'hold'}]}) as response:return response.status
        tasks=[asyncio.create_task(run()) for _ in range(2)];await asyncio.wait_for(self.held.wait(),2)
        self.assertEqual(self.manager.loads,1)
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,409)
        self.release.set();self.assertEqual(await asyncio.gather(*tasks),[200,200])
        async with self.client.post(self.url+'/v1/leases',headers=self.infer,json={'ttl_seconds':60}) as response:lease=(await response.json())['lease_id']
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,409)
        async with self.client.delete(self.url+'/v1/leases/'+lease,headers=self.infer) as response:self.assertEqual(response.status,200)
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,200)
        status,_,_=await self.chat();self.assertEqual(status,200)
        self.assertEqual(self.manager.loads,2)

    async def test_automatic_unload_and_on_demand_reload_of_the_applied_profile(self):
        manager=self.manager
        status,_=await self.post('/admin/profile/apply',profile='inkling-balanced');self.assertEqual(status,200)
        manager.idle_seconds=.01;manager.activity=time.monotonic()-1
        deadline=time.monotonic()+7
        while manager.state!='unloaded' and time.monotonic()<deadline:await asyncio.sleep(.05)
        self.assertEqual((manager.state,manager.active_key,manager.applied),('unloaded',None,'inkling-balanced'))
        manager.idle_seconds=3600
        status,_,_=await self.chat('fixture-balanced');self.assertEqual(status,200)
        self.assertEqual((manager.active_key,self.manager.starts[-1][0]),('inkling-balanced','inkling-balanced'))  # reloaded the applied profile, not the default

    async def test_stream_usage_is_counted_once_and_database_contains_no_payloads_or_keys(self):
        async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'stream':True,'messages':[{'role':'user','content':'PRIVATE_PROMPT'}]}) as response:
            text=await response.text();self.assertIn('PRIVATE_REASONING',text);self.assertNotIn('prompt_tokens',text)
        async with self.client.get(self.url+'/analytics/summary?timezone=UTC',headers=self.admin) as response:
            row=(await response.json())['rows'][0]
        self.assertEqual((row['input_tokens'],row['cached_tokens'],row['fresh_input_tokens'],row['output_tokens'],row['total_tokens']),(100,80,20,20,120))
        with self.app['store'].connect() as db:
            dump='\n'.join(db.iterdump());raw=dict(db.execute('SELECT * FROM requests').fetchone())
        self.assertEqual(raw['reasoning_tokens'],5);self.assertEqual(raw['status'],'complete')
        for secret in ('PRIVATE_PROMPT','PRIVATE_REASONING','PRIVATE_OUTPUT','infer-secret','control-secret'):self.assertNotIn(secret,dump)
        for unit in ('hour','day','week','month'):
            rows=self.app['store'].query(0,time.time()+1,unit,'UTC','python','fixture-swift')
            self.assertEqual(rows[0]['requests'],1)

    async def test_broken_stream_and_restart_recovery_do_not_invent_usage(self):
        async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'stream':True,'messages':[{'role':'user','content':'break'}]}) as response:
            try:await response.read()
            except ClientPayloadError:pass
        latest=self.app['store'].latest()
        self.assertNotEqual(latest['status'],'complete');self.assertIsNone(latest['input_tokens']);self.assertIsNone(latest['output_tokens'])
        self.assertEqual(self.manager.active,0)
        self.app['store'].start('interrupted-id','python','fixture-swift',True)
        from store import Store
        recovered=Store(self.app['store'].path)
        with recovered.connect() as db:row=db.execute('SELECT status,error FROM requests WHERE id=?',('interrupted-id',)).fetchone()
        self.assertEqual(tuple(row),('interrupted','gateway_restart'))


class ProfileSwitchingTests(Base):
    async def test_selecting_a_profile_never_switches_the_runtime(self):
        await self.chat()
        loads,starts=self.manager.loads,list(self.manager.starts)
        status,body=await self.post('/admin/profile/select',profile='inkling-maximum')
        self.assertEqual(status,200)
        self.assertEqual((body['profile']['selected'],body['profile']['active'],body['profile']['applied']),('inkling-maximum','swift','swift'))
        self.assertEqual((self.manager.loads,self.manager.starts,self.manager.state,self.manager.terminated),(loads,starts,'ready',0))
        status,_,_=await self.chat();self.assertEqual(status,200)  # still served by Swift
        self.assertEqual(self.seen[-1]['model'],'fixture-swift')
        self.assertEqual(self.app['store'].setting('selected_profile'),'inkling-maximum')   # persisted

    async def test_apply_switches_verifies_and_updates_routing_models_and_status(self):
        await self.chat()
        await self.post('/admin/profile/select',profile='inkling-balanced')
        status,body=await self.post('/admin/profile/apply')
        self.assertEqual(status,200);self.assertTrue(body['apply']['changed'])
        self.assertEqual((body['profile']['active'],body['profile']['applied'],body['model']['state']),('inkling-balanced','inkling-balanced','ready'))
        self.assertEqual(self.manager.starts[-1],('inkling-balanced',True))   # started with the inference probe
        self.assertEqual(self.manager.terminated,1)
        self.assertEqual((body['model']['model_id'],body['model']['context_capacity']),('fixture-balanced',204800))
        async with self.client.get(self.url+'/v1/models',headers=self.infer) as response:models=await response.json()
        self.assertEqual([(m['id'],m['loaded'],m['context']) for m in models['data']],[('fixture-balanced',True,204800)])
        status,_,data=await self.chat('fixture-balanced');self.assertEqual(status,200)
        self.assertEqual(self.seen[-1]['model'],'fixture-balanced')
        self.assertNotIn('temperature',self.seen[-1])   # Swift sampling defaults are not applied to Inkling
        status,_,_=await self.chat('fixture-swift')
        self.assertEqual(status,409)
        status,_=await self.post('/admin/profile/apply',profile='inkling-balanced')   # already active: no restart
        self.assertEqual(status,200);self.assertEqual(self.manager.terminated,1)

    async def test_inactive_unknown_and_missing_model_ids(self):
        status,_,data=await self.chat('fixture-balanced')
        self.assertEqual((status,data['error']['code']),(409,'model_not_active'))
        self.assertIn('fixture-swift',data['error']['message']);self.assertEqual(data['error']['served_model'],'fixture-swift')
        self.assertEqual(self.manager.active_key,None)   # a request for another model never loads or switches anything
        self.assertEqual(self.manager.loads,0)
        status,_,data=await self.chat('nonexistent');self.assertEqual((status,data['error']['code']),(400,'unknown_model'))
        status,_,_=await self.chat();self.assertEqual(status,200)   # no model: the applied profile serves it
        self.assertEqual(self.seen[-1]['model'],'fixture-swift')

    async def test_invalid_profiles_are_rejected(self):
        for path in ('/admin/profile/select','/admin/profile/apply'):
            status,body=await self.post(path,profile='nope')
            self.assertEqual((status,body['error']['code']),(400,'unknown_profile'))
        status,body=await self.post('/admin/profile/select');self.assertEqual(status,400)
        status,body=await self.post('/admin/profile/apply',headers=self.infer,profile='swift');self.assertEqual(status,403)
        self.assertEqual(self.manager.loads,0)

    async def test_active_requests_block_a_switch_and_nothing_is_interrupted(self):
        task=asyncio.create_task(self.hold())
        for _ in range(60):
            if self.manager.active:break
            await asyncio.sleep(.05)
        self.assertEqual(self.manager.active,1)
        status,body=await self.post('/admin/profile/apply',profile='inkling-balanced')
        self.assertEqual((status,body['error']['code']),(409,'inference_requests_active'))
        self.assertIn('Nothing was interrupted',body['error']['message'])
        self.assertEqual((self.manager.active_key,self.manager.terminated),('swift',0))
        self.release.set();self.assertEqual(await task,200)
        status,_=await self.post('/admin/profile/apply',profile='inkling-balanced');self.assertEqual(status,200)

    async def hold(self):
        async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'messages':[{'role':'user','content':'hold'}]}) as response:return response.status

    async def test_session_leases_block_a_switch(self):
        await self.chat()
        status,data=await self.post('/v1/leases',headers=self.infer,ttl_seconds=60);lease=data['lease_id']
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')
        self.assertEqual((status,body['error']['code']),(409,'session_leases_active'))
        self.assertEqual((self.manager.active_key,self.manager.terminated),('swift',0))
        async with self.client.delete(self.url+'/v1/leases/'+lease,headers=self.infer) as response:self.assertEqual(response.status,200)
        status,_=await self.post('/admin/profile/apply',profile='inkling-maximum');self.assertEqual(status,200)

    async def test_concurrent_lifecycle_operations_are_serialized(self):
        await self.chat()
        results=await asyncio.gather(self.post('/admin/profile/apply',profile='inkling-balanced'),self.post('/admin/profile/apply',profile='inkling-maximum'),
                                     self.post('/admin/profile/apply',profile='swift'),self.post('/admin/restart'))
        self.assertTrue(all(status in (200,409) for status,_ in results),results)
        self.assertFalse(self.manager.overlap,'two loads ran at the same time')
        self.assertEqual(self.manager.state,'ready');self.assertEqual(self.manager.active_key,self.manager.applied)

    async def test_failed_switch_cleans_up_and_restores_the_previous_profile(self):
        await self.chat()
        self.manager.fail.add('inkling-maximum')
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')
        self.assertEqual((status,body['error']['code'],body['error']['reason'],body['error']['restored']),(503,'switch_failed','fixture_load_failure','swift'))
        self.assertIn('Restored Swift Production',body['error']['message'])
        self.assertEqual((self.manager.state,self.manager.active_key,self.manager.applied),('ready','swift','swift'))
        self.assertIsNone(self.manager.switch)
        self.assertGreaterEqual(self.manager.terminated,1)
        status,_,_=await self.chat();self.assertEqual(status,200)
        st=await self.status();self.assertFalse(st['profile']['last_switch']['ok']);self.assertEqual(st['profile']['last_switch']['restored'],'swift')

    async def test_switch_to_a_cooling_down_profile_does_not_unload_the_running_model(self):
        await self.chat()
        self.manager.fail.add('inkling-maximum')
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')
        self.assertEqual((status,body['error']['restored']),(503,'swift'))
        stops,loads=self.manager.terminated,self.manager.loads
        self.manager.fail.discard('inkling-maximum')
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')   # inside the failed-load cooldown
        self.assertEqual((status,body['error']['code']),(503,'load_cooldown'))
        self.assertEqual((self.manager.terminated,self.manager.loads,self.manager.state,self.manager.active_key),(stops,loads,'ready','swift'))
        status,_,_=await self.chat();self.assertEqual(status,200)

    async def test_failed_switch_prioritizes_swift_when_the_previous_profile_also_fails(self):
        await self.post('/admin/profile/apply',profile='inkling-balanced')
        self.manager.fail.update({'inkling-maximum','inkling-balanced'})
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')
        self.assertEqual((status,body['error']['restored']),(503,'swift'))
        self.assertEqual((self.manager.active_key,self.manager.applied),('swift','swift'))
        self.manager.next_retry.clear();self.manager.fail.add('swift')   # cooldowns expired; now even Swift cannot start
        status,body=await self.post('/admin/profile/apply',profile='inkling-maximum')   # nothing can be restored: reported, state failed
        self.assertEqual(status,503);self.assertEqual(body['error']['restored'],None)
        self.assertEqual((self.manager.state,self.manager.active_key),('failed',None))

    async def test_usage_is_attributed_to_the_model_that_served_it_across_switches(self):
        await self.chat(reasoning_effort='low')
        await self.post('/admin/profile/apply',profile='inkling-balanced');await self.chat('fixture-balanced',reasoning_effort='medium')
        await self.post('/admin/profile/apply',profile='inkling-maximum');await self.chat('fixture-maximum')
        await self.chat('fixture-swift')   # rejected: attributed to what was asked, never to a profile
        with self.app['store'].connect() as db:
            rows=[tuple(r) for r in db.execute('SELECT model,profile,effort,status FROM requests ORDER BY started')]
        self.assertEqual(rows,[('fixture-swift','swift','low','complete'),('fixture-balanced','inkling-balanced','medium','complete'),
                               ('fixture-maximum','inkling-maximum','high','complete'),('fixture-swift',None,None,'failed')])
        async with self.client.get(self.url+'/analytics/requests?profile=inkling-balanced',headers=self.admin) as response:data=await response.json()
        self.assertEqual([(r['model'],r['profile']) for r in data['rows']],[('fixture-balanced','inkling-balanced')])
        self.assertEqual(sorted(data['models']),['fixture-balanced','fixture-maximum','fixture-swift'])

    async def test_legacy_rows_are_attributed_to_swift_once_and_rejected_rows_never_are(self):
        import sqlite3
        from store import Store
        path = Path(self.tmp.name) / 'legacy.db'
        db = sqlite3.connect(path)   # the schema as it was before profiles existed
        db.executescript('''CREATE TABLE requests(id TEXT PRIMARY KEY,client TEXT NOT NULL,model TEXT NOT NULL,started REAL NOT NULL,finished REAL,
            status TEXT NOT NULL,http_status INTEGER,error TEXT,stream INTEGER,input_tokens INTEGER,output_tokens INTEGER,total_tokens INTEGER,cached_tokens INTEGER,
            fresh_input_tokens INTEGER,reasoning_tokens INTEGER,prefill_tps REAL,decode_tps REAL,ttft_s REAL,upstream_ttft_s REAL,latency_s REAL,upstream_s REAL,
            usage_reported INTEGER DEFAULT 0);
            INSERT INTO requests(id,client,model,started,status) VALUES('old','python','fixture-swift',1,'complete');''')
        db.commit();db.close()
        index = registry.model_index(self.app['config'])
        migrated = Store(path, index)
        with migrated.connect() as db:self.assertEqual(db.execute('SELECT profile FROM requests').fetchone()[0], 'swift')
        migrated.start('rejected', 'python', 'fixture-maximum', False)   # a request refused because that model was not served
        Store(path, index)   # restart: the rejected row must not be credited to Inkling Maximum
        with migrated.connect() as db:self.assertIsNone(db.execute("SELECT profile FROM requests WHERE id='rejected'").fetchone()[0])
        self.assertEqual(migrated.models(), ['fixture-swift'])   # only models that actually served requests

    async def test_client_supplied_model_names_are_validated_and_bounded(self):
        for bad in (['x'], 7, {'a': 1}):
            status, _, data = await self.raw_chat(dict(model=bad, messages=[dict(role='user', content='hi')]))
            self.assertEqual((status, data['error']['code']), (400, 'invalid_model'))
        status, _, data = await self.chat('n' * 5000);self.assertEqual((status, data['error']['code']), (400, 'unknown_model'))
        with self.app['store'].connect() as db:
            self.assertEqual([len(r[0]) for r in db.execute("SELECT model FROM requests WHERE status='failed'")], [128])
        async with self.client.get(self.url + '/analytics/summary?timezone=UTC', headers=self.admin) as response:self.assertEqual((await response.json())['models'], [])

    async def raw_chat(self, body):
        async with self.client.post(self.url + '/v1/chat/completions', headers=self.infer, json=body) as response:
            return response.status, dict(response.headers), await response.json(content_type=None)

    async def test_settings_are_atomic_and_non_object_bodies_are_rejected(self):
        before = await self.status()
        status, body = await self.patch(auto_unload=False, default_profile='nope');self.assertEqual(status, 400)
        status, body = await self.patch(auto_unload=False, reasoning_effort='ultra');self.assertEqual(status, 400)
        status, body = await self.patch(idle_seconds=100, reasoning_effort='ultra');self.assertEqual(status, 400)
        after = await self.status()
        self.assertEqual((after['auto_unload'], after['idle_seconds']), (before['auto_unload'], before['idle_seconds']))   # nothing was applied
        status, body = await self.patch(auto_unload=False, idle_seconds=120, reasoning_effort='low');self.assertEqual(status, 200)
        self.assertEqual((body['auto_unload'], body['idle_seconds'], body['reasoning']['configured_default']), (False, 120, 'low'))
        for path in ('/admin/profile/select', '/admin/profile/apply'):
            for payload in ([1, 2], 'text', 3):
                async with self.client.post(self.url + path, headers=self.admin, json=payload) as response:self.assertEqual(response.status, 400, (path, payload))
        async with self.client.patch(self.url + '/admin/settings', headers=self.admin, json=['x']) as response:self.assertEqual(response.status, 400)
        self.assertEqual(self.manager.loads, 0)

    async def test_status_is_consistent_through_selection_switching_and_unload(self):
        st=await self.status()
        self.assertEqual((st['model']['state'],st['profile']['active'],st['profile']['applied'],st['profile']['selected'],st['profile']['default']),
                         ('unloaded',None,'swift','swift','swift'))
        self.assertEqual([p['key'] for p in st['profiles']],['swift','inkling-balanced','inkling-maximum'])
        self.assertEqual(st['model']['active_model_id'],None)
        await self.post('/admin/profile/select',profile='inkling-maximum');st=await self.status()
        self.assertEqual((st['profile']['selected'],st['profile']['active'],st['model']['model_id']),('inkling-maximum',None,'fixture-swift'))
        await self.post('/admin/profile/apply');st=await self.status()
        self.assertEqual((st['model']['state'],st['profile']['active'],st['model']['active_model_id'],st['reasoning']['profile']),('ready','inkling-maximum','fixture-maximum','inkling-maximum'))
        self.assertEqual(st['reasoning']['levels'],['none','minimal','low','medium','high','xhigh'])
        await self.post('/admin/unload');st=await self.status()
        self.assertEqual((st['model']['state'],st['profile']['active'],st['profile']['applied']),('unloaded',None,'inkling-maximum'))
        async with self.client.get(self.url+'/admin/profile',headers=self.admin) as response:p=await response.json()
        self.assertEqual((p['selected'],p['active'],p['applied']),('inkling-maximum',None,'inkling-maximum'))
        self.assertNotIn('/bin/true',json.dumps(st))

    async def test_default_profile_setting_is_persisted_but_does_not_switch(self):
        status,body=await self.patch(default_profile='inkling-balanced')
        self.assertEqual((status,body['profile']['default'],body['profile']['applied']),(200,'inkling-balanced','swift'))
        status,body=await self.patch(default_profile='nope');self.assertEqual(status,400)


class ReasoningTests(Base):
    async def test_per_profile_defaults_and_dashboard_default_without_reload(self):
        status,headers,_=await self.chat();self.assertEqual((status,headers['X-Reasoning-Effort'],headers['X-Model-Profile']),(200,'xhigh','swift'))
        self.assertEqual(self.seen[-1]['chat_template_kwargs'],dict(enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh'))
        loads=self.manager.loads
        status,body=await self.patch(reasoning_effort='medium');self.assertEqual((status,body['reasoning']['configured_default']),(200,'medium'))
        status,headers,_=await self.chat();self.assertEqual(headers['X-Reasoning-Effort'],'medium')
        self.assertEqual(self.seen[-1]['chat_template_kwargs']['reasoning_effort'],'medium')
        self.assertEqual(self.manager.loads,loads)   # no reload to change the default
        await self.post('/admin/profile/apply',profile='inkling-balanced')
        status,headers,_=await self.chat('fixture-balanced');self.assertEqual(headers['X-Reasoning-Effort'],'high')   # Inkling default is high
        self.assertEqual(self.seen[-1]['chat_template_kwargs'],{'reasoning_effort':'high'})
        status,body=await self.patch(reasoning_effort='low');self.assertEqual(body['reasoning']['configured_default'],'low')
        status,headers,_=await self.chat('fixture-balanced');self.assertEqual(headers['X-Reasoning-Effort'],'low')
        await self.post('/admin/profile/apply',profile='swift')
        status,headers,_=await self.chat();self.assertEqual(headers['X-Reasoning-Effort'],'medium')   # each profile keeps its own default

    async def test_request_overrides_precedence_and_isolation_between_clients(self):
        await self.patch(reasoning_effort='low')
        status,headers,_=await self.chat(reasoning_effort='xhigh');self.assertEqual(headers['X-Reasoning-Effort'],'xhigh')   # top-level OpenAI field
        status,headers,_=await self.chat(reasoning_effort='xhigh',chat_template_kwargs=dict(reasoning_effort='medium'),headers=self.other)
        self.assertEqual(headers['X-Reasoning-Effort'],'medium')   # template kwargs win, as in the original gateway
        status,headers,_=await self.chat(headers=self.infer);self.assertEqual(headers['X-Reasoning-Effort'],'low')   # other clients' overrides changed nothing
        self.assertEqual((await self.status())['reasoning']['configured_default'],'low')
        status,headers,_=await self.chat(reasoning_effort='none');self.assertEqual((headers['X-Reasoning-Effort'],self.seen[-1]['chat_template_kwargs']['enable_thinking']),('none',False))
        self.assertNotIn('reasoning_effort',self.seen[-1]['chat_template_kwargs'])
        status,headers,_=await self.chat(chat_template_kwargs=dict(enable_thinking=False));self.assertEqual(headers['X-Reasoning-Effort'],'none')

    async def test_inkling_levels_numeric_and_alias_are_forwarded_to_the_template(self):
        await self.post('/admin/profile/apply',profile='inkling-maximum')
        for sent,label,forwarded in (('minimal','minimal','minimal'),('none','none','none'),('max','xhigh','xhigh'),(0.3,'0.3',0.3)):
            status,headers,_=await self.chat('fixture-maximum',reasoning_effort=sent)
            self.assertEqual((status,headers['X-Reasoning-Effort'],self.seen[-1]['chat_template_kwargs']['reasoning_effort']),(200,label,forwarded))
        self.assertNotIn('enable_thinking',self.seen[-1]['chat_template_kwargs'])   # Swift-only parameter is never sent to Inkling

    async def test_unsupported_and_missing_settings_are_rejected_not_forwarded(self):
        seen=len(self.seen)
        for value in ('high','minimal','ultra',0.5,True):
            status,_,data=await self.chat(reasoning_effort=value)
            self.assertEqual((status,data['error']['code']),(400,'unsupported_reasoning_effort'),value)
            self.assertEqual(data['error']['supported'],['none','low','medium','xhigh'])
        self.assertEqual(len(self.seen),seen)   # nothing reached the model
        await self.post('/admin/profile/apply',profile='inkling-balanced')
        status,_,data=await self.chat('fixture-balanced',reasoning_effort='ultra');self.assertEqual((status,data['error']['code']),(400,'unsupported_reasoning_effort'))
        status,_,_=await self.chat('fixture-balanced',reasoning_effort=1.2);self.assertEqual(status,400)
        status,body=await self.patch(reasoning_effort='minimal');self.assertEqual(status,200)   # supported by Inkling
        await self.post('/admin/profile/apply',profile='swift')
        status,body=await self.patch(reasoning_effort='minimal');self.assertEqual((status,body['error']['code']),(400,'unsupported_reasoning_effort'))
        status,body=await self.patch(reasoning_effort='high',profile='nope');self.assertEqual(status,400)
        await self.post('/admin/profile/apply',profile='inkling-balanced')
        status, body = await self.patch(reasoning_effort='max');self.assertEqual((status, body['reasoning']['configured_default']), (200, 'xhigh'))   # alias stored canonically

    async def test_reasoning_and_tool_call_content_are_passed_through_untouched(self):
        tools=[dict(type='function',function=dict(name='f',parameters={}))]
        status,_,_=await self.chat(tools=tools,tool_choice='auto')
        self.assertEqual(self.seen[-1]['tools'],tools);self.assertEqual(self.seen[-1]['tool_choice'],'auto')


if __name__=='__main__':unittest.main()
