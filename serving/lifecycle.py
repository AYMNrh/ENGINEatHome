"""Own one stock llama-server, serialize transitions, and protect admitted requests."""
import asyncio
import logging
import os
from pathlib import Path
import time

from qah.telemetry import gpu_sample


class Busy(RuntimeError):pass
class Unavailable(RuntimeError):pass


class Lifecycle:
    def __init__(self,config,store,http):
        self.config=config;self.store=store;self.http=http;self.transition=asyncio.Lock()
        self.process=None;self.load_task=None;self.state='unloaded';self.active=0;self.accepting=True
        self.started=time.monotonic();self.loaded_at=None;self.load_seconds=None
        self.last_request=(store.latest() or {}).get('started')
        self.activity=time.monotonic();self.gpu=None;self.gpu_at=None;self.last_error=None;self.next_retry=0
        self.auto_unload=store.setting('auto_unload',True)
        self.idle_seconds=store.setting('idle_seconds',config.get('idle_seconds',3600))

    async def sample(self):
        try:self.gpu=await asyncio.to_thread(gpu_sample);self.gpu_at=time.time()
        except Exception:self.gpu=None

    async def _start(self):
        if time.monotonic()<self.next_retry:raise Unavailable('load_cooldown')
        began=time.monotonic();self.state='loading';self.last_error=None
        try:
            await self.sample()
            if not self.gpu or self.gpu.get('power_limit_watts')!=175:raise Unavailable('power_limit_must_be_175W')
            if self.gpu['free']<self.config['minimum_load_free_bytes']:raise Unavailable('insufficient_free_vram')
            probe=await asyncio.create_subprocess_exec('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
            output,_=await asyncio.wait_for(probe.communicate(),10)
            if probe.returncode or output.strip():raise Unavailable('another_cuda_workload_is_active')
            model=Path(self.config['model']);stat=model.stat()
            if dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)!=self.config['model_stat']:raise Unavailable('model_file_changed_reverification_required')
            # Do not attach to, or stop, an independently owned process on the backend port.
            try:
                reader,writer=await asyncio.wait_for(asyncio.open_connection('127.0.0.1',self.config['backend_port']),1)
                writer.close();await writer.wait_closed()
            except (OSError,asyncio.TimeoutError):pass
            else:raise Unavailable('backend_port_already_in_use')
            env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_ARG_','CUDA_VISIBLE_DEVICES'))}
            env['LD_LIBRARY_PATH']=str(Path(self.config['argv'][0]).parent)+':'+env.get('LD_LIBRARY_PATH','')
            self.process=await asyncio.create_subprocess_exec(*self.config['argv'],env=env,
                stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL)
            deadline=time.monotonic()+self.config.get('load_timeout',240)
            while time.monotonic()<deadline:
                if self.process.returncode is not None:raise Unavailable('backend_exited_'+str(self.process.returncode))
                try:
                    async with self.http.get(self.config['backend_url']+'/health',timeout=2) as response:
                        if response.status==200:
                            pid=self.process.pid
                            probe=await asyncio.create_subprocess_exec('nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL)
                            raw,_=await asyncio.wait_for(probe.communicate(),10)
                            memory=[int(line.split(',')[1].strip())*2**20 for line in raw.decode().splitlines() if line.split(',')[0].strip()==str(pid)]
                            if not memory or sum(memory)<self.config['minimum_process_device_bytes']:raise Unavailable('gpu_residency_guard_failed')
                            await self.sample()
                            if not self.gpu or self.gpu['free']<512*2**20:raise Unavailable('vram_reserve_guard_failed')
                            self.state='ready';self.loaded_at=time.monotonic();self.activity=time.monotonic()
                            self.load_seconds=time.monotonic()-began
                            logging.info('model_ready pid=%s load_seconds=%.3f',pid,self.load_seconds)
                            return
                except Unavailable:raise
                except (OSError,asyncio.TimeoutError):pass
                except Exception as error:
                    # HTTP connection failures during startup are expected; never log bodies/keys.
                    if type(error).__module__.startswith('aiohttp'):pass
                    else:raise
                await asyncio.sleep(.2)
            raise Unavailable('backend_load_timeout')
        except BaseException as error:
            await self._terminate()
            self.state='failed';self.last_error=str(error) if isinstance(error,Unavailable) else 'backend_start_failed'
            self.next_retry=time.monotonic()+15
            logging.warning('model_load_failed code=%s',self.last_error)
            raise Unavailable(self.last_error) from None

    async def _load_locked(self):
        if self.process and self.process.returncode is not None:
            self.process=None;self.state='failed';self.last_error='backend_exited'
        if self.state=='ready':return
        if not self.load_task or self.load_task.done():
            self.load_task=asyncio.create_task(self._start())
            self.load_task.add_done_callback(lambda task:None if task.cancelled() else task.exception())
        await asyncio.shield(self.load_task)

    async def load(self):
        if not self.accepting:raise Unavailable('gateway_draining')
        async with self.transition:await self._load_locked()

    async def enter(self):
        async with self.transition:
            if not self.accepting:raise Unavailable('gateway_draining')
            if self.active>=self.config.get('max_active_requests',8):raise Busy('request_queue_full')
            await self._load_locked()
            self.active+=1;self.activity=time.monotonic();self.last_request=time.time()

    async def leave(self):
        self.active-=1;self.activity=time.monotonic()

    async def _terminate(self):
        process=self.process
        if process and process.returncode is None:
            process.terminate()
            try:await asyncio.wait_for(process.wait(),20)
            except asyncio.TimeoutError:process.kill();await process.wait()
        self.process=None;self.loaded_at=None

    async def _unload_locked(self,ignore_leases=False):
        if self.active:raise Busy('inference_requests_active')
        if self.load_task and not self.load_task.done():raise Busy('model_loading')
        if not ignore_leases and await asyncio.to_thread(self.store.leases):raise Busy('session_leases_active')
        if self.process and self.process.returncode is None:
            try:
                async with self.http.get(self.config['backend_url']+'/slots',timeout=3) as response:
                    slots=await response.json()
                    if response.status!=200 or any(slot.get('is_processing') for slot in slots):raise Busy('backend_still_processing')
            except Busy:raise
            except Exception:raise Busy('backend_activity_could_not_be_verified') from None
        self.state='unloading';await self._terminate();self.state='unloaded';self.activity=time.monotonic()
        logging.info('model_unloaded')

    async def unload(self):
        async with self.transition:await self._unload_locked()

    async def restart(self):
        async with self.transition:
            await self._unload_locked();await self._load_locked()

    async def status(self):
        return dict(model=dict(state=self.state,model_id=self.config['model_id'],api_base_url=self.config['api_base_url'],
            context_capacity=self.config['context'],active_requests=self.active,
            load_seconds=self.load_seconds,uptime_seconds=time.monotonic()-self.loaded_at if self.loaded_at else None,
            last_request=self.last_request,last_error=self.last_error,
            session_leases=len(await asyncio.to_thread(self.store.leases))),
            gateway=dict(uptime_seconds=time.monotonic()-self.started,accepting=self.accepting),
            gpu=self.gpu,gpu_sample_at=self.gpu_at,auto_unload=self.auto_unload,idle_seconds=self.idle_seconds,
            timezone=self.config.get('timezone','Europe/Paris'))

    async def background(self):
        while True:
            await self.sample()
            if self.state=='ready' and self.process and self.process.returncode is not None:
                self.last_error='backend_exited_'+str(self.process.returncode);self.state='failed';self.process=None
                logging.warning('backend_crash code=%s',self.last_error)
            if self.auto_unload and self.state=='ready' and not self.active and time.monotonic()-self.activity>=self.idle_seconds:
                try:await self.unload()
                except Busy:pass
            await asyncio.sleep(5)

    async def close(self):
        self.accepting=False
        while self.active:await asyncio.sleep(.1)
        if self.load_task and not self.load_task.done():
            try:await self.load_task
            except Unavailable:pass
        async with self.transition:
            # Check the native slot too: cancellation may finish just after the HTTP handler.
            while self.process and self.process.returncode is None:
                try:await self._unload_locked(ignore_leases=True);break
                except Busy:await asyncio.sleep(.2)
