"""Static checks for the generated Home Assistant package, theme and dashboard (no HA, gateway or GPU needed)."""
import base64
import datetime as dt
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import jinja2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ha_files import generate, runtime_facts

ARGV = ('--model /m/Swift.gguf --ctx-size 262144 --parallel 1 --n-gpu-layers 99 --flash-attn on --cache-type-k q4_0 '
        '--cache-type-v q4_0 --batch-size 512 --ubatch-size 128 --spec-type draft-mtp --spec-draft-n-max 3 --no-context-shift '
        '--cache-ram 0 --ctx-checkpoints 4 --reasoning on --reasoning-effort xhigh --reasoning-budget -1').split()
ALLOWED = {'grid', 'tile', 'heading', 'conditional', 'markdown', 'gauge', 'history-graph', 'entities'}
SECRET = 'eah_TEST_SECRET_TOKEN'


class Loader(yaml.SafeLoader):pass
Loader.add_constructor('!secret', lambda loader, node: 'SECRET:' + loader.construct_scalar(node))


def slug(name):
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():yield from walk(value)
    elif isinstance(node, list):
        for value in node:yield from walk(value)


def defined_entities(package):
    found = set()
    for resource in package['rest']:
        for sensor in resource['sensor']:found.add('sensor.' + slug(sensor['name']))
    for sensor in package['sensor']:found.add('sensor.' + slug(sensor['name']))
    for block in package['template']:
        for domain in ('sensor', 'switch'):
            for item in block.get(domain, []):found.add(domain + '.' + slug(item['name']))
    for domain in ('input_select', 'input_datetime', 'script'):
        for key in package[domain]:found.add(domain + '.' + key)
    return found


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        generate(root, 'http://private.example:8080', 'swift-1.5-qwen3.8-27b', SECRET, ARGV)
        cls.root = root
        cls.package = yaml.load((root / 'package.yaml').read_text(encoding='utf-8'), Loader)
        cls.dashboard = yaml.load((root / 'dashboard.yaml').read_text(encoding='utf-8'), Loader)
        cls.theme = yaml.safe_load((root / 'themes/engineathome.yaml').read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_three_views_with_stable_paths(self):
        views = self.dashboard['views']
        self.assertEqual([(v['title'], v['path']) for v in views], [('Overview', 'overview'), ('Requests', 'requests'), ('Settings', 'settings')])
        for view in views:
            self.assertEqual((view['type'], view['theme']), ('sections', 'ENGINEatHome'))
        self.assertIn('ENGINEatHome', self.theme)

    def test_every_referenced_entity_is_defined(self):
        defined = defined_entities(self.package)
        used = set()
        for node in walk(self.dashboard):
            if isinstance(node.get('entity'), str):used.add(node['entity'])
            for key in ('entities',):
                for item in node.get(key, []) if isinstance(node.get(key), list) else []:
                    used.add(item if isinstance(item, str) else item['entity'])
        for node in walk(self.dashboard):
            if node.get('type') == 'markdown':
                used.update(re.findall(r"((?:sensor|input_select|input_number|switch|script)\.engineathome_\w+)", node['content']))
        self.assertGreater(len(used), 40)
        self.assertEqual(sorted(used - defined), [])

    def test_only_native_cards_and_no_external_navigation_or_dependencies(self):
        for node in walk(self.dashboard):
            if 'type' in node and 'cards' not in node and 'sections' not in node and node.get('type') not in ('sections',) and 'condition' not in node and not isinstance(node.get('type'), type(None)):
                if node['type'] in ALLOWED or node.get('type') in ('sections', 'entity', 'numeric_state'):continue
                self.fail('non-native or unexpected card type: ' + str(node['type']))
        text = (self.root / 'dashboard.yaml').read_text(encoding='utf-8')
        for forbidden in ('custom:', 'iframe', 'apexcharts', 'mushroom', 'type: url', 'action: url', SECRET):self.assertNotIn(forbidden, text)

    def test_no_nonfunctional_profile_selector(self):
        text = (self.root / 'dashboard.yaml').read_text(encoding='utf-8') + (self.root / 'package.yaml').read_text(encoding='utf-8')
        self.assertNotIn('Inkling', text)
        self.assertNotIn('input_select.engineathome_profile', text)

    def test_secret_only_in_private_secrets_file(self):
        for name in ('package.yaml', 'dashboard.yaml', 'themes/engineathome.yaml'):
            self.assertNotIn(SECRET, (self.root / name).read_text(encoding='utf-8'))
        self.assertIn(SECRET, (self.root / 'secrets.yaml').read_text(encoding='utf-8'))
        self.assertTrue(all(r['headers']['Authorization'] == 'SECRET:engineathome_ha_authorization' for r in self.package['rest']))

    def test_lifecycle_controls_are_confirmed_and_cache_loss_is_stated(self):
        text = (self.root / 'dashboard.yaml').read_text(encoding='utf-8')
        self.assertEqual(text.count('confirmation:'), 2)
        self.assertIn('discards the prefix cache', text)
        self.assertIn('never interrupt running requests', text)

    def test_runtime_facts_come_from_real_arguments(self):
        facts = dict(runtime_facts(ARGV))
        self.assertEqual(facts['Context'], '262,144 tokens')
        self.assertEqual(facts['KV cache'], 'q4_0')
        self.assertEqual(facts['Speculative decoding'], 'draft-mtp, up to 3 draft tokens')
        self.assertEqual(facts['Context shifting'], 'off')
        self.assertIn('Swift.gguf', facts['Model file'])
        self.assertEqual(dict(runtime_facts(None))['Source'], 'not supplied when this dashboard was generated')

    def test_idle_timeout_helpers(self):
        options = self.package['input_select']['engineathome_idle_timeout']['options']
        self.assertEqual(options[0], '15 min')
        self.assertIn('1 hour', options)
        self.assertIn('rest_command', self.package)
        self.assertIn('engineathome_set_idle', self.package['rest_command'])


def environment(states, attrs):
    env = jinja2.Environment()
    env.filters['base64_encode'] = lambda text: base64.b64encode(text.encode()).decode()
    env.filters['timestamp_custom'] = lambda value, fmt='%Y-%m-%d', local=True: dt.datetime.fromtimestamp(value, dt.timezone.utc).strftime(fmt)
    env.globals['states'] = lambda entity: states.get(entity, 'unknown')
    env.globals['state_attr'] = lambda entity, attribute: (attrs.get(entity) or {}).get(attribute)
    return env


def request(started, status='complete', **values):
    row = dict(id='x', client='opencode', model='swift', started=started, status=status, error=None, input_tokens=1000, cached_tokens=600,
               fresh_input_tokens=400, output_tokens=50, total_tokens=1050, prefill_tps=800.123, decode_tps=44.456, ttft_s=1.234,
               latency_s=9.87)
    row.update(values)
    return row


def summary_row(period, client='opencode', **values):
    row = dict(period=period, client=client, model='swift', requests=3, completed=3, incomplete_or_failed=0, usage_reported_requests=3,
               input_tokens=3000, output_tokens=150, total_tokens=3150, cached_tokens=1800, fresh_input_tokens=1200, cache_known_requests=3,
               mean_latency_s=2.5, mean_ttft_s=0.8)
    row.update(values)
    return row


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        generate(Path(cls.tmp.name), 'http://private.example:8080', 'swift', SECRET, ARGV)
        dashboard = yaml.load((Path(cls.tmp.name) / 'dashboard.yaml').read_text(encoding='utf-8'), Loader)
        cls.cards = {}
        for node in walk(dashboard):
            if node.get('type') == 'markdown':cls.cards[node['content'].split('\n')[0][:40] + str(len(cls.cards))] = node['content']

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def render(self, content, states, attrs):
        return environment(states, attrs).from_string(content).render()

    def all_states(self, rows, sensor_state='3'):
        sensors = ('usage_history', 'request_history', 'recent_requests', 'usage_trend')
        states = {f'sensor.engineathome_{s}': sensor_state for s in sensors}
        states.update({'input_select.engineathome_status': 'All', 'input_select.engineathome_token_view': 'Total',
                       'input_select.engineathome_group_by': 'day', 'sensor.engineathome_context_capacity': '262144'})
        attrs = {f'sensor.engineathome_{s}': dict(rows=rows, timezone='Europe/Paris') for s in sensors}
        return states, attrs

    def svg(self, text):
        match = re.search(r'data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)', text)
        self.assertTrue(match, 'chart image missing')
        return ET.fromstring(base64.b64decode(match.group(1)))

    def test_every_card_renders_with_populated_nullable_empty_and_unavailable_data(self):
        t = 1_795_000_000
        populated = [summary_row('2026-11-21T10:00:00+01:00'), summary_row('2026-11-21T11:00:00+01:00', client='hermes', cached_tokens=None, fresh_input_tokens=None),
                     summary_row('2026-11-21T12:00:00+01:00', output_tokens=None, total_tokens=None)]
        requests = [request(t), request(t - 90, status='failed', error='backend_http_500', prefill_tps=None, decode_tps=None, ttft_s=None, input_tokens=None, cached_tokens=None)]
        for rows, state in ((populated + [r for r in requests], '3'), ([], '0'), ([], 'unavailable')):
            states, attrs = self.all_states(populated if rows else [], state)
            for key in ('recent_requests', 'request_history'):attrs[f'sensor.engineathome_{key}'] = dict(rows=rows and requests, timezone='Europe/Paris')
            for name, content in self.cards.items():
                text = self.render(content, states, attrs)
                self.assertNotIn('None', text.replace('\n', ' ').split('data:image')[0], name)
                self.assertNotIn('nan', text.lower().split('data:image')[0].replace('financ', ''), name)

    def test_token_chart_is_valid_svg_and_counts_unknown_as_missing_not_zero(self):
        rows = [summary_row('2026-11-21T10:00:00+01:00', cached_tokens=None, fresh_input_tokens=None, input_tokens=3000)]
        states, attrs = self.all_states(rows)
        chart = next(c for c in self.cards.values() if 'groupby' in c and 'usage_trend' in c)
        svg = self.svg(self.render(chart, states, attrs))
        self.assertEqual(svg.tag, '{http://www.w3.org/2000/svg}svg')
        fills = {r.get('fill') for r in svg.iter('{http://www.w3.org/2000/svg}rect')}
        self.assertIn('#7d8794', fills)  # input with no cache split is shown as its own series

    def test_throughput_has_separate_axes_and_ignores_requests_without_rates(self):
        t = 1_795_000_000
        rows = [request(t + 60 * i) for i in range(5)] + [request(t + 500, prefill_tps=None, decode_tps=None)]
        states, attrs = self.all_states(rows)
        card = next(c for c in self.cards.values() if 'polyline' in c)
        out = self.render(card, states, attrs)
        svg = self.svg(out)
        self.assertEqual(len(list(svg.iter('{http://www.w3.org/2000/svg}polyline'))), 2)
        self.assertIn('5 measured requests', out)

    def test_status_filter_applies_to_request_table(self):
        t = 1_795_000_000
        rows = [request(t), request(t - 60, status='failed', error='backend_http_500')]
        states, attrs = self.all_states(rows)
        states['input_select.engineathome_status'] = 'failed'
        card = next(c for c in self.cards.values() if 'Started' in c and 'input_select.engineathome_status' in c)
        out = self.render(card, states, attrs)
        self.assertIn('backend_http_500', out)
        self.assertEqual(out.count('● complete'), 0)
        self.assertIn('**1** matching request', out)


if __name__ == '__main__':unittest.main()
