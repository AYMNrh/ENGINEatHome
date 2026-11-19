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
from aiohttp import web,ClientSession,ClientPayloadError
from gateway import create_app
from lifecycle import Lifecycle


class ServingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.held=asyncio.Event();self.release=asyncio.Event();self.entered=0
        self.usage=dict(prompt_tokens=100,completion_tokens=20,total_tokens=120,
                        prompt_tokens_details=dict(cached_tokens=80),completion_tokens_details=dict(reasoning_tokens=5))
        async def backend(request):
            body=await request.json();text=body.get('messages',[{}])[0].get('content','')
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
        class Fake(Lifecycle):
            loads=0
            async def _start(self):
                self.loads+=1;self.state='ready';self.process=SimpleNamespace(returncode=None,pid=123)
                self.loaded_at=time.monotonic();self.load_seconds=.01
            async def _terminate(self):self.process=None;self.loaded_at=None
            async def sample(self):self.gpu=dict(used=1,free=2**30,power_limit_watts=175);self.gpu_at=time.time()
        keys={hashlib.sha256(key.encode()).hexdigest():dict(client=client,scopes=scopes) for key,client,scopes in
              [('infer-secret','python',['inference']),('control-secret','admin',['control','analytics'])]}
        config=dict(keys=keys,database=str(Path(self.tmp.name)/'usage.db'),model_id='fixture-model',api_base_url='http://private/v1',
                    backend_url=f'http://127.0.0.1:{port}',context=262144,timezone='UTC',sampling={},request_timeout=10,max_active_requests=8)
        self.app=create_app(config,manager_factory=Fake);self.runner=web.AppRunner(self.app,access_log=None,handler_cancellation=True)
        await self.runner.setup();site=web.TCPSite(self.runner,'127.0.0.1',0);await site.start()
        self.url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]);self.client=ClientSession()
        self.infer={'Authorization':'Bearer infer-secret'};self.admin={'Authorization':'Bearer control-secret'}

    async def asyncTearDown(self):
        self.release.set();await self.client.close();await self.runner.cleanup();await self.backend.cleanup();self.tmp.cleanup()

    async def test_lifecycle_auth_concurrent_load_active_guard_and_leases(self):
        async with self.client.get(self.url+'/v1/models') as response:self.assertEqual(response.status,401)
        async with self.client.post(self.url+'/admin/load',headers=self.infer,json={}) as response:self.assertEqual(response.status,403)
        async def run():
            async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'messages':[{'role':'user','content':'hold'}]}) as response:return response.status
        tasks=[asyncio.create_task(run()) for _ in range(2)];await asyncio.wait_for(self.held.wait(),2)
        self.assertEqual(self.app['manager'].loads,1)
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,409)
        self.release.set();self.assertEqual(await asyncio.gather(*tasks),[200,200])
        async with self.client.post(self.url+'/v1/leases',headers=self.infer,json={'ttl_seconds':60}) as response:lease=(await response.json())['lease_id']
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,409)
        async with self.client.delete(self.url+'/v1/leases/'+lease,headers=self.infer) as response:self.assertEqual(response.status,200)
        async with self.client.post(self.url+'/admin/unload',headers=self.admin,json={}) as response:self.assertEqual(response.status,200)
        async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={}) as response:self.assertEqual(response.status,200)
        self.assertEqual(self.app['manager'].loads,2)
        manager=self.app['manager'];manager.idle_seconds=.01;manager.activity=time.monotonic()-1
        deadline=time.monotonic()+7
        while manager.state!='unloaded' and time.monotonic()<deadline:await asyncio.sleep(.05)
        self.assertEqual(manager.state,'unloaded')

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
            rows=self.app['store'].query(0,time.time()+1,unit,'UTC','python','fixture-model')
            self.assertEqual(rows[0]['requests'],1)

    async def test_broken_stream_and_restart_recovery_do_not_invent_usage(self):
        async with self.client.post(self.url+'/v1/chat/completions',headers=self.infer,json={'stream':True,'messages':[{'role':'user','content':'break'}]}) as response:
            try:await response.read()
            except ClientPayloadError:pass
        # The finished response means finalization has completed before middleware returns.
        latest=self.app['store'].latest()
        self.assertNotEqual(latest['status'],'complete');self.assertIsNone(latest['input_tokens']);self.assertIsNone(latest['output_tokens'])
        self.assertEqual(self.app['manager'].active,0)
        self.app['store'].start('interrupted-id','python','fixture-model',True)
        from store import Store
        recovered=Store(self.app['store'].path)
        with recovered.connect() as db:row=db.execute('SELECT status,error FROM requests WHERE id=?',('interrupted-id',)).fetchone()
        self.assertEqual(tuple(row),('interrupted','gateway_restart'))


if __name__=='__main__':unittest.main()
