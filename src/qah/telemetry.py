"""Sample device-wide VRAM and process RAM; missing counters remain null."""
import ctypes
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def host_inventory():
    if os.name != 'nt':
        try:
            fields = {}
            for line in Path('/proc/meminfo').read_text().splitlines():
                name, value = line.split(':', 1)
                if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
                    fields[name] = int(value.split()[0]) * 1024
            return dict(platform=os.name, physical_total_bytes=fields['MemTotal'],
                        physical_available_bytes=fields['MemAvailable'],
                        swap_used_bytes=fields['SwapTotal'] - fields['SwapFree'],
                        scope='current Linux kernel/WSL guest, not Windows host')
        except (OSError, KeyError, ValueError) as error:
            return dict(platform=os.name, memory=None, error=str(error))
    from ctypes import wintypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', wintypes.DWORD), ('load', wintypes.DWORD)] + [
            (key, ctypes.c_ulonglong) for key in ('physical_total', 'physical_available',
            'pagefile_total', 'pagefile_available', 'virtual_total', 'virtual_available', 'extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return dict(platform=os.name, memory=None, error=ctypes.get_last_error())
    return dict(platform=os.name, physical_total_bytes=status.physical_total,
                physical_available_bytes=status.physical_available,
                commit_limit_bytes=status.pagefile_total, commit_available_bytes=status.pagefile_available)


def gpu_sample():
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=5, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if p.returncode:
            return None
        fields = p.stdout.splitlines()[0].split(',')
        def number(value):
            try:
                return float(value.strip())
            except ValueError:
                return None
        used, free = [int(x.strip()) * 1024**2 for x in fields[:2]]
        return dict(used=used, free=free, utilization_percent=number(fields[2]),
                    temperature_c=number(fields[3]), power_watts=number(fields[4]),
                    power_limit_watts=number(fields[5]))
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def process_memory(pid):
    if os.name != 'nt':
        try:
            values = {}
            for line in Path('/proc/%d/status' % pid).read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    name, value = line.split(':', 1)
                    values[name] = int(value.split()[0]) * 1024
            return dict(rss=values.get('VmRSS'), peak_rss=values.get('VmHWM'), pagefile=None)
        except (OSError, ValueError):
            return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (key, ctypes.c_size_t) for key in ('peak_rss', 'rss', 'peak_paged', 'paged',
                                              'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    kernel, psapi = ctypes.WinDLL('kernel32', use_last_error=True), ctypes.WinDLL('psapi')
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    handle = kernel.OpenProcess(0x1000 | 0x10, False, pid)
    if not handle:
        return None
    try:
        counter = Counters()
        counter.cb = ctypes.sizeof(counter)
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counter), counter.cb):
            return dict(rss=counter.rss, peak_rss=counter.peak_rss, pagefile=counter.pagefile)
    finally:
        kernel.CloseHandle(handle)


def resource_summary(samples):
    """Integrate only covered power intervals; do not bridge gaps over five seconds."""
    gpu = [s['gpu'] for s in samples if s.get('gpu')]
    power = [g['power_watts'] for g in gpu if g.get('power_watts') is not None]
    joules = covered = 0.0
    for left, right in zip(samples, samples[1:]):
        dt = right['monotonic'] - left['monotonic']
        a, b = left.get('gpu') or {}, right.get('gpu') or {}
        if 0 < dt <= 5 and a.get('power_watts') is not None and b.get('power_watts') is not None:
            joules += dt * (a['power_watts'] + b['power_watts']) / 2
            covered += dt
    duration = samples[-1]['monotonic'] - samples[0]['monotonic'] if len(samples) > 1 else 0
    rss = [s['ram']['peak_rss'] for s in samples if s.get('ram') and s['ram'].get('peak_rss') is not None]
    available = [s['host']['physical_available_bytes'] for s in samples
                 if s.get('host', {}).get('physical_available_bytes') is not None]
    return dict(device_vram_peak_sampled_bytes=max((g['used'] for g in gpu), default=None),
                device_vram_min_free_bytes=min((g['free'] for g in gpu), default=None),
                process_peak_rss_bytes=max(rss, default=None),
                host_min_available_bytes=min(available, default=None),
                gpu_power_mean_watts=joules / covered if covered else None,
                gpu_power_peak_watts=max(power, default=None),
                gpu_energy_wh=joules / 3600 if covered else None,
                power_covered_seconds=covered, observed_seconds=duration,
                power_coverage_fraction=covered / duration if duration else None,
                gpu_temperature_peak_c=max((g['temperature_c'] for g in gpu if g.get('temperature_c') is not None), default=None),
                gpu_utilization_mean_percent=(sum(g['utilization_percent'] for g in gpu if g.get('utilization_percent') is not None)
                    / sum(g.get('utilization_percent') is not None for g in gpu)
                    if any(g.get('utilization_percent') is not None for g in gpu) else None),
                power_scope='device-wide GPU including other applications; CPU and wall power unavailable',
                samples=len(samples), device_counter_includes_other_processes=True, exact_vram_peak=False)


class Monitor:
    def __init__(self, executor, interval=1.0, output=None):
        self.executor = executor
        self.interval = interval
        self.output = Path(output) if output else None
        self.samples = []
        self.stop = threading.Event()

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        while not self.stop.is_set():
            process = self.executor.process
            sample = dict(time=time.time(), monotonic=time.monotonic(), gpu=gpu_sample(),
                          ram=process_memory(process.pid) if process else None, host=host_inventory())
            self.samples.append(sample)
            if self.output:
                with self.output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(sample) + '\n')
            self.stop.wait(self.interval)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=6)

    def report(self):
        return dict(resource_summary(self.samples), target_sample_interval_seconds=self.interval,
                    scope='trial startup through current request; cumulative')
