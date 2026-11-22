"""Own one llama-server at a time (one GPU, several validated profiles), serialize transitions, protect admitted requests.

Vocabulary used by the gateway and the dashboard:
  selected  the profile a user picked in the UI; choosing it never touches the runtime
  applied   the profile the gateway serves: it is what an on-demand load starts and what requests must name
  active    the profile whose process currently occupies the GPU (None while unloaded)
  default   the profile applied after a clean gateway start
"""
import asyncio
import logging
import os
from pathlib import Path
import time

from qah.telemetry import gpu_sample
import profiles as registry


class Busy(RuntimeError):pass
class Unavailable(RuntimeError):pass


class ModelNotActive(RuntimeError):
    def __init__(self, requested, active, applied):
        super().__init__('model_not_active')
        self.requested, self.active, self.applied = requested, active, applied


class SwitchFailed(Unavailable):
    def __init__(self, code, restored=None):
        super().__init__(code)
        self.code, self.restored = code, restored


class Lifecycle:
    def __init__(self, config, store, http):
        if 'profiles' not in config:config = registry.normalize(config)
        self.config = config
        self.profiles = config['profiles']
        self.store, self.http = store, http
        self.transition = asyncio.Lock()
        self.process = None
        self.load_task = None
        self.apply_task = None
        self.state = 'unloaded'
        self.active = 0
        self.accepting = True
        self.started = time.monotonic()
        self.loaded_at = None
        self.load_seconds = None
        self.last_request = (store.latest() or {}).get('started')
        self.activity = time.monotonic()
        self.gpu = None
        self.gpu_at = None
        self.last_error = None
        self.next_retry = {}
        self.active_key = None
        self.switch = None
        self.last_switch = None
        self.auto_unload = store.setting('auto_unload', True)
        self.idle_seconds = store.setting('idle_seconds', config.get('idle_seconds', 3600))
        default = store.setting('default_profile', config['default_profile'])
        self.default = default if default in self.profiles else config['default_profile']
        self.applied = self.default
        selected = store.setting('selected_profile', self.default)
        self.selected = selected if selected in self.profiles else self.default
        self.public = {key: registry.public_profile(key, p) for key, p in self.profiles.items()}

    # ------------------------------------------------------------------ helpers
    def profile(self, key=None):
        return self.profiles[key or self.active_key or self.applied]

    def key_for_model(self, model_id):
        return registry.model_index(self.config).get(model_id)

    def reasoning_default(self, key):
        return self.store.setting('reasoning_default.' + key, None)

    def set_reasoning_default(self, key, level):
        spec = self.profiles[key]['reasoning']
        if not isinstance(level, str) or level not in spec['levels']:raise registry.ReasoningError('unsupported_reasoning_effort', list(spec['levels']))
        self.store.set_setting('reasoning_default.' + key, level)

    def select(self, key):
        if key not in self.profiles:raise KeyError(key)
        self.selected = key
        self.store.set_setting('selected_profile', key)

    def set_default(self, key):
        if key not in self.profiles:raise KeyError(key)
        self.default = key
        self.store.set_setting('default_profile', key)

    async def sample(self):
        try:self.gpu = await asyncio.to_thread(gpu_sample);self.gpu_at = time.time()
        except Exception:self.gpu = None

    def _running(self):
        return self.process is not None and self.process.returncode is None

    def _static_preflight(self, key):
        """Checks that need no GPU, run before anything is unloaded for a switch."""
        p = self.profiles[key]
        try:stat = Path(p['model']).stat()
        except OSError:raise Unavailable('model_file_missing') from None
        if dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns) != p['model_stat']:raise Unavailable('model_file_changed_reverification_required')
        if not p.get('executable') or not os.access(p['executable'], os.X_OK):raise Unavailable('runtime_executable_missing')

    async def _compute_apps(self, fields):
        probe = await asyncio.create_subprocess_exec('nvidia-smi', '--query-compute-apps=' + fields, '--format=csv,noheader,nounits',
                                                     stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        output, _ = await asyncio.wait_for(probe.communicate(), 10)
        return probe.returncode, output.decode()

    async def _probe(self, key):
        """A switch is only successful once the new runtime answers a real (tiny) inference request."""
        p = self.profiles[key]
        kwargs = dict(p.get('request_defaults') or {})
        kwargs.update(registry.resolve_effort(p['reasoning'], 'none')['kwargs'])
        body = dict(messages=[dict(role='user', content='Reply with the single word: ok')], max_tokens=24, temperature=0, stream=False,
                    chat_template_kwargs=kwargs)
        try:
            async with self.http.post(self.config['backend_url'] + '/v1/chat/completions', json=body, timeout=120) as response:
                data = await response.json()
                if response.status != 200 or not data.get('choices'):raise Unavailable('inference_probe_failed')
        except Unavailable:raise
        except Exception:raise Unavailable('inference_probe_failed') from None

    # ------------------------------------------------------------------ load / unload
    async def _start(self, key, probe=False):
        p = self.profiles[key]
        if time.monotonic() < self.next_retry.get(key, 0):raise Unavailable('load_cooldown')
        began = time.monotonic()
        self.state = 'switching' if self.switch else 'loading'
        self.last_error = None
        try:
            await self.sample()
            if not self.gpu or self.gpu.get('power_limit_watts') != 175:raise Unavailable('power_limit_must_be_175W')
            if self.gpu['free'] < p['minimum_load_free_bytes']:raise Unavailable('insufficient_free_vram')
            code, apps = await self._compute_apps('pid')
            if code or apps.strip():raise Unavailable('another_cuda_workload_is_active')
            self._static_preflight(key)
            # Do not attach to, or stop, an independently owned process on the backend port.
            try:
                reader, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', self.config['backend_port']), 1)
                writer.close();await writer.wait_closed()
            except (OSError, asyncio.TimeoutError):pass
            else:raise Unavailable('backend_port_already_in_use')
            env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_ARG_', 'CUDA_VISIBLE_DEVICES'))}
            env['LD_LIBRARY_PATH'] = str(Path(p['executable']).parent) + ':' + env.get('LD_LIBRARY_PATH', '')
            self.process = await asyncio.create_subprocess_exec(*p['argv'], env=env, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            deadline = time.monotonic() + p.get('load_timeout', 240)
            while time.monotonic() < deadline:
                if self.process.returncode is not None:raise Unavailable('backend_exited_' + str(self.process.returncode))
                try:
                    async with self.http.get(self.config['backend_url'] + '/health', timeout=2) as response:
                        if response.status == 200:
                            pid = self.process.pid
                            code, raw = await self._compute_apps('pid,used_gpu_memory')
                            memory = [int(line.split(',')[1].strip()) * 2 ** 20 for line in raw.splitlines() if line.split(',')[0].strip() == str(pid)]
                            if not memory or sum(memory) < p['minimum_process_device_bytes']:raise Unavailable('gpu_residency_guard_failed')
                            await self.sample()
                            if not self.gpu or self.gpu['free'] < 512 * 2 ** 20:raise Unavailable('vram_reserve_guard_failed')
                            if probe:
                                if self.switch:self.switch['phase'] = 'verifying_inference'
                                await self._probe(key)
                            self.state = 'ready'
                            self.active_key = key
                            self.loaded_at = time.monotonic()
                            self.activity = time.monotonic()
                            self.load_seconds = time.monotonic() - began
                            logging.info('model_ready profile=%s pid=%s load_seconds=%.3f', key, pid, self.load_seconds)
                            return
                except Unavailable:raise
                except (OSError, asyncio.TimeoutError):pass
                except Exception as error:
                    # HTTP connection failures during startup are expected; never log bodies/keys.
                    if type(error).__module__.startswith('aiohttp'):pass
                    else:raise
                await asyncio.sleep(.2)
            raise Unavailable('backend_load_timeout')
        except BaseException as error:
            await self._terminate()
            self.active_key = None
            self.state = 'failed'
            self.last_error = str(error) if isinstance(error, Unavailable) else 'backend_start_failed'
            self.next_retry[key] = time.monotonic() + 15
            logging.warning('model_load_failed profile=%s code=%s', key, self.last_error)
            raise Unavailable(self.last_error) from None

    async def _load_locked(self):
        if self.process and self.process.returncode is not None:
            self.process = None
            self.active_key = None
            self.state = 'failed'
            self.last_error = 'backend_exited'
        if self.state == 'ready':return
        if not self.load_task or self.load_task.done():
            self.load_task = asyncio.create_task(self._start(self.applied))
            self.load_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        await asyncio.shield(self.load_task)

    async def load(self):
        if not self.accepting:raise Unavailable('gateway_draining')
        async with self.transition:await self._load_locked()

    async def enter(self, key):
        async with self.transition:
            if not self.accepting:raise Unavailable('gateway_draining')
            if key != self.applied:raise ModelNotActive(self.profiles[key]['model_id'], self.active_key, self.applied)
            if self.active >= self.config.get('max_active_requests', 8):raise Busy('request_queue_full')
            await self._load_locked()
            self.active += 1
            self.activity = time.monotonic()
            self.last_request = time.time()

    async def leave(self):
        self.active -= 1
        self.activity = time.monotonic()

    async def _terminate(self):
        process = self.process
        if process and process.returncode is None:
            process.terminate()
            try:await asyncio.wait_for(process.wait(), 20)
            except asyncio.TimeoutError:process.kill();await process.wait()
        self.process = None
        self.loaded_at = None

    async def _slots_idle(self):
        if self._running():
            try:
                async with self.http.get(self.config['backend_url'] + '/slots', timeout=3) as response:
                    slots = await response.json()
                    if response.status != 200 or any(slot.get('is_processing') for slot in slots):raise Busy('backend_still_processing')
            except Busy:raise
            except Exception:raise Busy('backend_activity_could_not_be_verified') from None

    async def _unload_locked(self, ignore_leases=False):
        if self.active:raise Busy('inference_requests_active')
        if self.load_task and not self.load_task.done():raise Busy('model_loading')
        if not ignore_leases and await asyncio.to_thread(self.store.leases):raise Busy('session_leases_active')
        await self._slots_idle()
        self.state = 'unloading'
        await self._terminate()
        self.active_key = None
        self.state = 'unloaded'
        self.activity = time.monotonic()
        logging.info('model_unloaded')

    async def unload(self):
        async with self.transition:await self._unload_locked()

    async def restart(self):
        async with self.transition:
            await self._unload_locked()
            await self._load_locked()

    # ------------------------------------------------------------------ profile switching
    async def _wait_released(self, key):
        """After stopping the old process, wait until the GPU really has the room the new profile needs."""
        need = self.profiles[key]['minimum_load_free_bytes']
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            await self.sample()
            if self.gpu and self.gpu['free'] >= need:return
            await asyncio.sleep(.5)
        raise Unavailable('gpu_memory_not_released')

    async def _restore(self, previous, failed):
        """After a failed switch bring back what worked: the previous profile, then the configured default (Swift)."""
        if previous is None:return None
        candidates = [previous] if previous != failed else []
        fallback = self.config['default_profile']
        if fallback not in candidates and fallback != failed:candidates.append(fallback)
        original = self.last_error
        for key in candidates:
            try:
                if self.switch:self.switch.update(phase='restoring', restoring=key)
                await self._start(key, probe=True)
                self.applied = key
                self.last_error = original
                return key
            except Unavailable:
                continue
        self.last_error = original
        return None

    async def _apply(self, key):
        async with self.transition:
            if not self.accepting:raise Unavailable('gateway_draining')
            running = self._running()
            if running and self.active_key == key and self.state == 'ready':
                self.applied = key
                return dict(changed=False, active=key, previous=key)
            if self.active:raise Busy('inference_requests_active')
            if self.load_task and not self.load_task.done():raise Busy('model_loading')
            if await asyncio.to_thread(self.store.leases):raise Busy('session_leases_active')
            await self._slots_idle()
            self._static_preflight(key)
            previous = self.active_key if running else None
            self.switch = {'from': previous, 'to': key, 'phase': 'unloading' if running else 'loading', 'started': time.time()}
            began = time.time()
            try:
                if running:
                    self.state = 'switching'
                    await self._terminate()
                    self.active_key = None
                    self.switch['phase'] = 'releasing_gpu'
                    await self._wait_released(key)
                self.switch['phase'] = 'loading'
                await self._start(key, probe=True)
                self.applied = key
                self.last_switch = dict(ok=True, at=time.time(), seconds=round(time.time() - began, 1), to=key, previous=previous, error=None, restored=None)
                return dict(changed=True, active=key, previous=previous)
            except BaseException as error:
                code = str(error) if isinstance(error, Unavailable) else 'switch_failed'
                self.last_error = code
                restored = await self._restore(previous, key)
                self.last_switch = dict(ok=False, at=time.time(), seconds=round(time.time() - began, 1), to=key, previous=previous, error=code, restored=restored)
                raise SwitchFailed(code, restored) from None
            finally:
                self.switch = None

    async def apply(self, key):
        """Make `key` the served profile. Safe only when idle; transactional; verified by a real inference."""
        if key not in self.profiles:raise KeyError(key)
        # Run detached from the HTTP handler so a client disconnect cannot leave a half-finished switch.
        self.apply_task = asyncio.create_task(self._apply(key))
        self.apply_task.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        return await asyncio.shield(self.apply_task)

    # ------------------------------------------------------------------ status
    async def status(self):
        key = self.active_key or self.applied
        p = self.profiles[key]
        return dict(
            model=dict(state=self.state, model_id=p['model_id'], active_model_id=self.profiles[self.active_key]['model_id'] if self.active_key else None,
                       api_base_url=self.config['api_base_url'], context_capacity=p.get('context'), active_requests=self.active,
                       load_seconds=self.load_seconds, uptime_seconds=time.monotonic() - self.loaded_at if self.loaded_at else None,
                       last_request=self.last_request, last_error=self.last_error,
                       session_leases=len(await asyncio.to_thread(self.store.leases))),
            profile=dict(selected=self.selected, active=self.active_key, applied=self.applied, default=self.default,
                         switching=self.switch, last_switch=self.last_switch),
            profiles=list(self.public.values()),
            reasoning=dict(profile=self.applied, **{k: v for k, v in self.public[self.applied]['reasoning'].items()},
                           configured_default=self.reasoning_default(self.applied) or self.public[self.applied]['reasoning']['default'],
                           default_source='dashboard' if self.reasoning_default(self.applied) else 'profile'),
            gateway=dict(uptime_seconds=time.monotonic() - self.started, accepting=self.accepting),
            gpu=self.gpu, gpu_sample_at=self.gpu_at, auto_unload=self.auto_unload, idle_seconds=self.idle_seconds,
            timezone=self.config.get('timezone', 'Europe/Paris'))

    async def background(self):
        while True:
            await self.sample()
            if self.state == 'ready' and self.process and self.process.returncode is not None:
                self.last_error = 'backend_exited_' + str(self.process.returncode)
                self.state = 'failed'
                self.process = None
                self.active_key = None
                logging.warning('backend_crash code=%s', self.last_error)
            if self.auto_unload and self.state == 'ready' and not self.active and time.monotonic() - self.activity >= self.idle_seconds:
                try:await self.unload()
                except Busy:pass
            await asyncio.sleep(5)

    async def close(self):
        self.accepting = False
        while self.active:await asyncio.sleep(.1)
        for task in (self.apply_task, self.load_task):
            if task and not task.done():
                try:await task
                except (Unavailable, Busy):pass
        async with self.transition:
            # Check the native slot too: cancellation may finish just after the HTTP handler.
            while self.process and self.process.returncode is None:
                try:await self._unload_locked(ignore_leases=True);break
                except Busy:await asyncio.sleep(.2)
