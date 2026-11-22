"""Private authenticated OpenAI gateway. Payloads exist only in memory, never in the ledger."""
import argparse
import asyncio
import datetime as dt
import hashlib
import json
import logging
from pathlib import Path
import re
import signal
import time
import uuid

from aiohttp import web,ClientSession,ClientTimeout,ClientError
from lifecycle import Lifecycle,Busy,Unavailable,ModelNotActive,SwitchFailed
import profiles as registry
from store import Store,Counters

HINTS=dict(inference_requests_active='Requests are running or queued; wait for them to finish. Nothing was interrupted.',
           session_leases_active='A session lease is held; release it or let it expire. Leases are never bypassed.',
           model_loading='The runtime is loading; try again when it is ready.',
           backend_still_processing='The native server is still processing a request.',
           backend_activity_could_not_be_verified='Native server activity could not be verified, so nothing was stopped.',
           request_queue_full='The request queue is full.')


def error(status,code,message=None,**extra):
    return web.json_response(dict(error=dict(message=message or HINTS.get(code,code),type='gateway_error',code=code,**extra)),status=status)


@web.middleware
async def protection(request,handler):
    public=request.path in ('/healthz','/analytics','/')
    if not public:
        header=request.headers.get('Authorization','')
        token=header[7:] if header.startswith('Bearer ') else ''
        identity=request.app['config']['keys'].get(hashlib.sha256(token.encode()).hexdigest()) if token else None
        if not identity:return error(401,'authentication_required')
        request['identity']=identity
        needed='inference' if request.path.startswith('/v1/') or request.path=='/completion' else 'analytics' if request.path.startswith('/analytics/') else 'control'
        if needed not in identity['scopes']:return error(403,'insufficient_scope')
        if request.path=='/admin/status' and identity['client']=='home-assistant':request.app['ha_last_poll']=time.time()
    try:return await handler(request)
    except web.HTTPException:raise
    except Busy as exc:return error(409,str(exc))
    except Unavailable as exc:return error(503,str(exc))
    except (ValueError,KeyError,TypeError):return error(400,'invalid_request')
    except asyncio.CancelledError:raise
    except Exception as exc:
        logging.error('handler_failed type=%s',type(exc).__name__)
        return error(500,'internal_gateway_error')


async def inference(request):
    app=request.app;config=app['config'];manager=app['manager'];store=app['store']
    id=str(uuid.uuid4());began=time.perf_counter();values={};status='failed';http_status=500;failure=None
    counted=False;upstream_start=None;counters=Counters();response=None;key=None;label=None
    try:
        body=await request.json()
        if not isinstance(body,dict):raise ValueError('object required')
    except (ValueError,TypeError):return error(400,'invalid_json_object')
    stream=body.get('stream',False)
    if not isinstance(stream,bool):return error(400,'stream_must_be_boolean')
    requested=body.get('model')
    # A request names exactly one model. Unknown ids are rejected; known but inactive ids are rejected, never served by another model.
    key=manager.applied if requested is None else manager.key_for_model(requested)
    await asyncio.to_thread(store.start,id,request['identity']['client'],requested or config['profiles'][manager.applied]['model_id'],stream)
    try:
        if key is None:
            http_status=400;failure='unknown_model';return error(http_status,failure)
        profile=config['profiles'][key]
        body['model']=profile['model_id']
        for name,value in profile.get('sampling',{}).items():body.setdefault(name,value)
        kwargs=body.setdefault('chat_template_kwargs',{})
        if not isinstance(kwargs,dict):raise ValueError('invalid template parameters')
        for name,value in profile.get('request_defaults',{}).items():kwargs.setdefault(name,value)
        spec=profile['reasoning']
        # Precedence is unchanged from the original gateway: template kwargs, then the top-level OpenAI field, then the default.
        chosen=kwargs.get(spec['parameter'],body.get(spec['parameter']))
        try:resolved=registry.resolve_effort(spec,chosen,await asyncio.to_thread(manager.reasoning_default,key))
        except registry.ReasoningError as exc:
            http_status=400;failure=exc.code
            return error(400,exc.code,f"{profile['display_name']} supports reasoning_effort: "+', '.join(exc.supported),supported=exc.supported)
        label=resolved['label']
        if spec.get('none_disables_thinking') and kwargs.get('enable_thinking') is False and not (chosen is not None and label=='none'):label='none'
        else:kwargs.update(resolved['kwargs'])
        wanted_usage=bool((body.get('stream_options') or {}).get('include_usage'))
        if stream and request.path.startswith('/v1/'):
            body.setdefault('stream_options',{})['include_usage']=True
        async with asyncio.timeout(config.get('request_timeout',14400)):
            await manager.enter(key);counted=True;upstream_start=time.perf_counter()
            # Attribution is recorded only once the request is admitted to the model that will serve it.
            await asyncio.to_thread(store.tag,id,profile['model_id'],key,label)
            async with app['http'].post(config['backend_url']+request.path,json=body) as upstream:
                http_status=upstream.status
                headers={'X-Request-ID':id,'Cache-Control':'no-store','X-Model-Profile':key,'X-Reasoning-Effort':label}
                if upstream.status>=400:
                    failure='backend_http_'+str(upstream.status)
                    data=await upstream.read()
                    return web.Response(body=data,status=upstream.status,headers=headers,content_type='application/json')
                if not stream:
                    data=await upstream.read()
                    try:counters.observe(json.loads(data))
                    except (ValueError,TypeError):failure='invalid_backend_json';http_status=502;return error(502,failure)
                    status='complete'
                    return web.Response(body=data,status=upstream.status,headers=headers,content_type='application/json')
                response=web.StreamResponse(status=upstream.status,headers=dict(headers,**{'Content-Type':'text/event-stream','X-Accel-Buffering':'no'}))
                await response.prepare(request)
                buffer=b''
                async for chunk in upstream.content.iter_any():
                    buffer+=chunk
                    if len(buffer)>4*2**20:raise ValueError('oversized SSE event')
                    while True:
                        match=re.search(rb'\r?\n\r?\n',buffer)
                        if not match:break
                        packet,buffer=buffer[:match.end()],buffer[match.end():]
                        lines=[line[5:].lstrip() for line in packet.splitlines() if line.startswith(b'data:')]
                        data=b'\n'.join(lines);hide=False
                        if data==b'[DONE]':counters.done=True
                        elif data:
                            event=json.loads(data)
                            if event.get('error'):failure='backend_stream_error'
                            counters.observe(event)
                            if counters.first_token and 'ttft_s' not in values:
                                now=time.perf_counter();values.update(ttft_s=now-began,upstream_ttft_s=now-upstream_start)
                            hide=not wanted_usage and event.get('choices')==[] and bool(event.get('usage'))
                        if not hide:await response.write(packet)
                if buffer.strip():raise ValueError('incomplete SSE packet')
                status='failed' if failure else 'complete' if counters.done else 'incomplete'
                if status=='incomplete':failure='upstream_stream_incomplete'
                await response.write_eof()
                return response
    except ModelNotActive as exc:
        http_status=409;failure='model_not_active'
        active=config['profiles'][exc.active]['display_name'] if exc.active else None
        served=config['profiles'][exc.applied]['model_id']
        return error(409,failure,f"{exc.requested} is not the model being served. Serving: {served}"+(' (loaded)' if exc.active else ' (not loaded yet)')+
                     '. Apply the other profile from the dashboard or POST /admin/profile/apply; requests are never served by a different model.',
                     requested_model=exc.requested,served_model=served,active_profile=exc.active,hint='POST /admin/profile/apply')
    except Busy as exc:http_status=409;failure=str(exc);return error(http_status,failure)
    except Unavailable as exc:http_status=503;failure=str(exc);return error(http_status,failure)
    except (asyncio.TimeoutError,TimeoutError):
        http_status=504;status='timeout';failure='request_timeout'
        if response:return response
        return error(504,failure)
    except asyncio.CancelledError:
        status='interrupted';failure='handler_cancelled';http_status=499;raise
    except (ConnectionResetError,BrokenPipeError):
        status='client_disconnected';failure='client_disconnected';http_status=499
        return response if response is not None else error(499,failure)
    except ClientError:
        status='failed';failure='backend_connection_lost';http_status=502
        return response if response is not None else error(502,failure)
    except (ValueError,TypeError):
        failure='invalid_request_or_backend_event';http_status=400 if response is None else 502
        return response if response is not None else error(http_status,failure)
    finally:
        values.update(counters.finish());values['latency_s']=time.perf_counter()-began
        values['upstream_s']=time.perf_counter()-upstream_start if upstream_start is not None else None
        try:await asyncio.shield(asyncio.to_thread(store.finish,id,status,http_status,failure,values))
        finally:
            if counted:await asyncio.shield(manager.leave())


async def status(request):
    result=await request.app['manager'].status();result['home_assistant_last_poll']=request.app['ha_last_poll']
    result['last_inference']=await asyncio.to_thread(request.app['store'].latest)
    return web.json_response(result)


async def action(request):
    await getattr(request.app['manager'],request.match_info['action'])()
    return await status(request)


async def settings(request):
    manager=request.app['manager'];body=await request.json()
    if set(body)-{'auto_unload','idle_seconds','default_profile','reasoning_effort','profile'}:raise ValueError('unknown setting')
    if 'auto_unload' in body:
        if not isinstance(body['auto_unload'],bool):raise ValueError('boolean required')
        manager.auto_unload=body['auto_unload'];await asyncio.to_thread(manager.store.set_setting,'auto_unload',manager.auto_unload)
    if 'idle_seconds' in body:
        value=body['idle_seconds']
        if not isinstance(value,int) or isinstance(value,bool) or not 60<=value<=86400:raise ValueError('idle range')
        manager.idle_seconds=value;await asyncio.to_thread(manager.store.set_setting,'idle_seconds',value)
    if 'default_profile' in body:
        if body['default_profile'] not in manager.profiles:return error(400,'unknown_profile',supported=list(manager.profiles))
        await asyncio.to_thread(manager.set_default,body['default_profile'])
    if 'reasoning_effort' in body:
        key=body.get('profile') or manager.applied
        if key not in manager.profiles:return error(400,'unknown_profile',supported=list(manager.profiles))
        try:await asyncio.to_thread(manager.set_reasoning_default,key,body['reasoning_effort'])
        except registry.ReasoningError as exc:
            return error(400,exc.code,f"{manager.profiles[key]['display_name']} supports reasoning_effort: "+', '.join(exc.supported),supported=exc.supported)
    return await status(request)


async def profiles_list(request):
    manager=request.app['manager']
    return web.json_response(dict(profiles=list(manager.public.values()),default=manager.default))


async def profile_get(request):
    manager=request.app['manager']
    return web.json_response(dict(selected=manager.selected,active=manager.active_key,applied=manager.applied,default=manager.default,
                                  state=manager.state,switching=manager.switch,last_switch=manager.last_switch))


async def profile_select(request):
    """Choosing a target never touches the running model; it only records the choice."""
    manager=request.app['manager'];body=await request.json()
    key=body.get('profile')
    if key not in manager.profiles:return error(400,'unknown_profile',supported=list(manager.profiles))
    await asyncio.to_thread(manager.select,key)
    return await status(request)


async def profile_apply(request):
    manager=request.app['manager']
    try:body=await request.json()
    except ValueError:body={}
    key=body.get('profile') or manager.selected
    if key not in manager.profiles:return error(400,'unknown_profile',supported=list(manager.profiles))
    try:result=await manager.apply(key)
    except Busy as exc:return error(409,str(exc),profile=key)
    except SwitchFailed as exc:
        return error(503,'switch_failed',f'Could not switch to {manager.profiles[key]["display_name"]}: {exc.code}. '+
                     (f'Restored {manager.profiles[exc.restored]["display_name"]}.' if exc.restored else 'The previous runtime could not be restored automatically.'),
                     reason=exc.code,restored=exc.restored,profile=key)
    status_body=json.loads((await status(request)).body)
    status_body['apply']=result
    return web.json_response(status_body)


async def models(request):
    manager=request.app['manager'];key=manager.active_key or manager.applied;p=manager.profiles[key]
    return web.json_response(dict(object='list',data=[dict(id=p['model_id'],object='model',owned_by='local',loaded=manager.active_key is not None,
                                                           context=p.get('context'),profile=key)]))


async def lease(request):
    async with request.app['manager'].transition:
        return await lease_locked(request)


async def lease_locked(request):
    store=request.app['store'];identity=request['identity'];client=identity['client']
    if request.method=='GET':return web.json_response(dict(leases=[r for r in await asyncio.to_thread(store.leases) if r['client']==client or 'control' in identity['scopes']]))
    id=request.match_info.get('id') or str(uuid.uuid4())
    if request.match_info.get('id'):
        existing=next((r for r in await asyncio.to_thread(store.leases) if r['id']==id),None)
        if not existing:return error(404,'lease_not_found')
        if existing['client']!=client and 'control' not in identity['scopes']:return error(403,'lease_owner_required')
        client=existing['client']
    if request.method=='DELETE':
        await asyncio.to_thread(store.release,id);return web.json_response(dict(released=True))
    body=await request.json();ttl=body.get('ttl_seconds',3600)
    if not isinstance(ttl,int) or isinstance(ttl,bool) or not 60<=ttl<=86400:raise ValueError('invalid lease TTL')
    expires=time.time()+ttl;await asyncio.to_thread(store.lease,id,client,expires)
    return web.json_response(dict(lease_id=id,expires=expires,ttl_seconds=ttl))


def timestamp(value,default):
    if not value:return default
    try:return float(value)
    except ValueError:
        parsed=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
        if parsed.tzinfo is None:raise ValueError('timezone required')
        return parsed.timestamp()


async def analytics(request):
    config=request.app['config'];store=request.app['store'];q=request.query
    timezone=q.get('timezone',config['timezone'])
    if timezone not in ('UTC',config['timezone']):raise ValueError('timezone unsupported')
    if request.path.endswith('/totals'):
        return web.json_response(dict(timezone=timezone,periods=await asyncio.to_thread(store.totals,timezone)))
    start=timestamp(q.get('start'),time.time()-7*86400);end=timestamp(q.get('end'),time.time()+1)
    if not start<end or end-start>367*86400:raise ValueError('time range must be <=367 days')
    rows=await asyncio.to_thread(store.query,start,end,q.get('bucket','day'),timezone,q.get('client'),q.get('model'),request.path.endswith('/requests'),q.get('profile'))
    manager=request.app['manager']
    return web.json_response(dict(rows=rows,timezone=timezone,clients=await asyncio.to_thread(store.clients),
        model=config['profiles'][manager.applied]['model_id'],models=await asyncio.to_thread(store.models),bucket=q.get('bucket','day'),
        accounting='input includes cached; fresh=input-cached only when both reported; total=input+output; reasoning is an output subset; buckets use request start',
        unknown='null means unreported; failed/incomplete requests may lack final usage'))


async def page(request):return web.FileResponse(Path(__file__).with_name('analytics.html'),headers={'Cache-Control':'no-store'})


def create_app(config,store=None,manager_factory=Lifecycle):
    config=registry.normalize(config)
    app=web.Application(middlewares=[protection],client_max_size=32*2**20)
    app['config']=config;app['store']=store or Store(config['database'],registry.model_index(config));app['ha_last_poll']=None
    async def resources(app):
        async with ClientSession(timeout=ClientTimeout(total=None,connect=15)) as http:
            app['http']=http;app['manager']=manager_factory(config,app['store'],http)
            background=asyncio.create_task(app['manager'].background())
            yield
            background.cancel()
            try:await background
            except asyncio.CancelledError:pass
            await app['manager'].close()
    app.cleanup_ctx.append(resources)
    app.router.add_get('/healthz',lambda request:web.json_response(dict(gateway='ok')))
    app.router.add_get('/',page);app.router.add_get('/analytics',page)
    app.router.add_get('/admin/status',status)
    app.router.add_post('/admin/{action:load|unload|restart}',action)
    app.router.add_patch('/admin/settings',settings)
    app.router.add_get('/v1/models',models)
    app.router.add_get('/admin/profiles',profiles_list);app.router.add_get('/admin/profile',profile_get)
    app.router.add_post('/admin/profile/select',profile_select);app.router.add_post('/admin/profile/apply',profile_apply)
    for path in ('/v1/chat/completions','/v1/completions','/completion'):app.router.add_post(path,inference)
    for prefix in ('/v1/leases','/admin/leases'):
        app.router.add_get(prefix,lease);app.router.add_post(prefix,lease)
        app.router.add_put(prefix+'/{id}',lease);app.router.add_delete(prefix+'/{id}',lease)
    for name in ('summary','requests','totals'):app.router.add_get('/analytics/'+name,analytics)
    return app


async def main(config):
    app=create_app(config);runner=web.AppRunner(app,access_log=None,shutdown_timeout=config['request_timeout']+10,handler_cancellation=True)
    await runner.setup();sites=[];stop=asyncio.Event()
    loop=asyncio.get_running_loop()
    for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,stop.set)
    local=web.TCPSite(runner,'127.0.0.1',config['port']);await local.start();sites.append(local)
    async def bind_tail():
        while True:
            try:
                site=web.TCPSite(runner,config['tailscale_ip'],config['port']);await site.start();sites.append(site)
                logging.info('private_gateway_listening port=%s',config['port']);return
            except OSError:await asyncio.sleep(5)
    binding=asyncio.create_task(bind_tail())
    try:await stop.wait()
    finally:
        app['manager'].accepting=False
        binding.cancel()
        try:await binding
        except asyncio.CancelledError:pass
        for site in sites:await site.stop()
        while app['manager'].active:await asyncio.sleep(.1)
        await runner.cleanup()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    asyncio.run(main(json.loads(args.config.read_text())))
