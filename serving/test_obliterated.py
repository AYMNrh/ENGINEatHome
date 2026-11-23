"""Tests for the Qwen3.8-27B-OBLITERATED profile, RAM prompt-cache reporting and the Swift cache-ram change."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lifecycle import Lifecycle, Unavailable
import profiles as registry
from test_serving import Base, make_profiles

MEASURED = dict(loaded_mib=19150, peak_mib=19150, process_floor_mib=17000, qualification='fixture', qualification_status='experimental')


def obliterated(kv='q8_0', cache_ram=8192):
    return registry.obliterated_profile('/rt/llama-server', '/m/Qwen3.8-27B-OBLITERATED-Q4_K_M.gguf', dict(bytes=1, mtime_ns=1), 'a' * 64, 18081,
                                        '/t/qwen35-tools.jinja', 'b' * 64, kv, MEASURED, cache_ram)


class ObliteratedProfileTests(unittest.TestCase):
    def test_launch_line_is_the_measured_one_and_adds_only_gateway_plumbing(self):
        v = registry.parse_argv(obliterated('f16')['argv'])
        for flag, value in (('spec-type', 'draft-mtp'), ('spec-draft-n-max', '3'), ('spec-draft-p-min', '0.6'), ('batch-size', '2048'),
                            ('ubatch-size', '512'), ('flash-attn', 'on'), ('cache-ram', '8192'), ('ctx-size', '65536'), ('parallel', '1'),
                            ('cache-type-k', 'f16'), ('cache-type-v', 'f16'), ('n-gpu-layers', '99')):
            self.assertEqual(v[flag], value, flag)
        self.assertEqual((v['host'], v['port'], v['alias']), ('127.0.0.1', '18081', 'qwen3.8-27b-obliterated'))
        self.assertIn('no-context-shift', v)
        self.assertIn('no-cache-idle-slots', v)
        self.assertEqual(v['chat-template-file'], '/t/qwen35-tools.jinja')

    def test_four_profiles_validate_and_swift_stays_the_default(self):
        profiles = make_profiles(1)
        profiles['obliterated'] = obliterated()
        config = registry.normalize(dict(profiles=profiles, default_profile='swift'))
        self.assertEqual((config['default_profile'], len(config['profiles'])), ('swift', 4))
        self.assertEqual(registry.model_index(config)['qwen3.8-27b-obliterated'], 'obliterated')

    def test_public_facts_for_the_dashboard_have_no_paths(self):
        public = registry.public_profile('obliterated', obliterated('q8_0'))
        self.assertEqual((public['quantization'], public['kv_cache'], public['context'], public['qualification_status']), ('Q4_K_M', 'q8_0', 65536, 'experimental'))
        self.assertEqual(public['mtp'], dict(enabled=True, draft_n_max=3, draft_p_min=0.6))
        self.assertEqual((public['ram_cache']['enabled'], public['ram_cache']['mib']), (True, 8192))
        text = json.dumps(public)
        for private in ('/rt/', '/m/', '/t/', 'aaaa', 'bbbb'):
            self.assertNotIn(private, text)

    def test_missing_cache_ram_flag_is_the_runtime_default_not_off(self):
        p = dict(registry.normalize(dict(profiles=make_profiles(1), default_profile='swift'))['profiles']['inkling-balanced'])
        self.assertEqual(registry.public_profile('x', p)['ram_cache'], dict(enabled=True, mib=8192, label='runtime default (8,192 MiB)'))
        self.assertFalse(registry.public_profile('x', dict(p, argv=p['argv'] + ['--cache-ram', '0']))['ram_cache']['enabled'])
        idle = dict(p, argv=p['argv'] + ['--cache-ram', '8192', '--no-cache-idle-slots'])
        self.assertIn('idle slots not cached', registry.public_profile('x', idle)['ram_cache']['label'])

    def test_reasoning_defaults_to_the_measured_no_thinking_mode(self):
        spec = obliterated()['reasoning']
        self.assertEqual(registry.resolve_effort(spec, None)['kwargs'], dict(enable_thinking=False))
        self.assertEqual(registry.resolve_effort(spec, 'xhigh')['kwargs'], dict(reasoning_effort='xhigh'))
        with self.assertRaises(registry.ReasoningError):
            registry.resolve_effort(spec, 'high')

    def test_swift_cache_change_touches_only_cache_ram(self):
        import add_profiles
        profiles = make_profiles(1)
        profiles['swift']['argv'] = ['/x/llama-server', '--model', '/m/swift.gguf', '--ctx-size', '262144', '--cache-ram', '0', '--spec-type', 'draft-mtp',
                                     '--spec-draft-n-max', '3', '-ub', '128']
        config = dict(profiles=profiles, default_profile='swift', argv=list(profiles['swift']['argv']))
        before = list(profiles['swift']['argv'])
        add_profiles.apply_swift_cache(config, 8192)
        after = config['profiles']['swift']['argv']
        self.assertEqual([a for a in after if a not in ('8192', '--no-cache-idle-slots')], [a for a in before if a != '0'])
        self.assertIn('--no-cache-idle-slots', after)
        self.assertEqual((after[after.index('--cache-ram') + 1], config['argv']), ('8192', after))
        add_profiles.apply_swift_cache(config, 0)
        self.assertEqual(config['profiles']['swift']['argv'], before)

    def test_merge_adds_obliterated_and_leaves_the_other_profiles_identical(self):
        import add_profiles
        tuned = make_profiles(1)
        tuned['swift']['load_timeout'] = 777
        base = dict(profiles=tuned, default_profile='swift', backend_port=18081)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(base))
            plain = add_profiles.merge(path, dry_run=True)
            full = add_profiles.merge(path, dry_run=True, obliterated=dict(kv='q8_0', measured=MEASURED))
        self.assertEqual(sorted(full['profiles']), ['inkling-balanced', 'inkling-maximum', 'obliterated', 'swift'])
        for key in ('swift', 'inkling-balanced', 'inkling-maximum'):
            self.assertEqual(full['profiles'][key], plain['profiles'][key])
        self.assertEqual(full['default_profile'], 'swift')

    def test_store_cache_summary_separates_cached_follow_ups_from_cold_prompts(self):
        from store import Store
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / 'u.db')
            for i, (inp, cached, ttft) in enumerate([(20000, 0, 40.0), (21500, 20000, 4.0), (23000, 21500, 4.4), (100, 0, 0.3)]):
                store.start(str(i), 'opencode', 'm', True)
                store.tag(str(i), 'm', 'obliterated', 'none')
                store.finish(str(i), 'complete', 200, None, dict(input_tokens=inp, cached_tokens=cached, ttft_s=ttft, prefill_tps=480.0, decode_tps=30.0, usage_reported=1))
            summary = store.cache_summary('obliterated')
            self.assertEqual((summary['cached_requests'], summary['cold_requests']), (2, 1))
            self.assertEqual((summary['cached_ttft_s'], summary['cold_ttft_s']), (4.2, 40.0))
            self.assertEqual(summary['hit_rate_pct'], round(100 * 41500 / 64600, 1))
            self.assertEqual(store.cache_summary('swift')['requests'], 0)


class ObliteratedLifecycleTests(Base):
    def preflight(self, key, **changes):
        self.manager.profiles[key].update(changes)
        return Lifecycle._static_preflight(self.manager, key)

    async def test_blocked_profile_cannot_be_applied(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'm.gguf'
            model.write_bytes(b'x')
            stat = model.stat()
            with self.assertRaises(Unavailable) as raised:
                self.preflight('inkling-maximum', qualification_status='blocked', model=str(model), model_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
        self.assertEqual(str(raised.exception), 'profile_blocked')

    async def test_chat_template_file_is_verified_before_a_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / 'm.gguf'
            model.write_bytes(b'x')
            stat = model.stat()
            template = Path(folder) / 't.jinja'
            template.write_text('{{ x }}')
            common = dict(model=str(model), model_stat=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns), chat_template_file=str(template))
            self.assertIsNone(self.preflight('inkling-maximum', chat_template_sha256=registry.sha256_file(template), **common))
            with self.assertRaises(Unavailable) as raised:
                self.preflight('inkling-maximum', chat_template_sha256='0' * 64)
            self.assertEqual(str(raised.exception), 'chat_template_changed_reverification_required')
            template.unlink()
            with self.assertRaises(Unavailable) as raised:
                self.preflight('inkling-maximum')
            self.assertEqual(str(raised.exception), 'chat_template_missing')

    async def test_status_reports_cache_settings_and_host_memory(self):
        manager = self.app['manager']
        manager.profiles['swift']['argv'] = ['x', '--cache-ram', '8192']
        manager.public['swift'] = registry.public_profile('swift', manager.profiles['swift'])
        body = await self.status()
        self.assertEqual((body['cache']['enabled'], body['cache']['ram_limit_mib'], body['cache']['profile']), (True, 8192, 'swift'))
        self.assertIn('recent', body['cache'])
        self.assertIn('host_available_mib', body['cache'])
        for key in ('quantization', 'mtp', 'ram_cache', 'qualification_status'):
            self.assertIn(key, body['profiles'][0])


if __name__ == '__main__':
    unittest.main()
