"""Static checks for the generated Home Assistant package, theme and dashboard (no HA, gateway or GPU needed)."""
import base64
import datetime as dt
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import jinja2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ha_files import generate
import profiles as registry
from test_serving import make_profiles

CONFIG = registry.normalize(dict(profiles=make_profiles(1), default_profile='swift'))
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
        generate(root, 'http://private.example:8080', SECRET, CONFIG)
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

    def test_model_choices_come_from_the_profile_registry_not_from_constants(self):
        options=self.package['input_select']['engineathome_target_profile']['options']
        self.assertEqual(options,['Swift Production','Inkling Balanced','Inkling Maximum'])
        self.assertEqual(self.package['input_select']['engineathome_default_profile']['options'],options)
        self.assertEqual(self.package['input_select']['engineathome_model']['options'],['All','fixture-swift','fixture-balanced','fixture-maximum'])
        levels=self.package['input_select']['engineathome_reasoning_effort']['options']
        self.assertEqual(levels,['none','low','medium','xhigh','minimal','high'])   # union; trimmed to the active model by an automation
        other=registry.normalize(dict(profiles={'alpha':dict(make_profiles(1)['swift'],display_name='Alpha',model_id='alpha-1')},default_profile='alpha'))
        with tempfile.TemporaryDirectory() as folder:
            generate(Path(folder),'http://x',SECRET,other)
            self.assertEqual(yaml.load((Path(folder)/'package.yaml').read_text(encoding='utf-8'),Loader)['input_select']['engineathome_target_profile']['options'],['Alpha'])
            self.assertNotIn('Inkling',(Path(folder)/'dashboard.yaml').read_text(encoding='utf-8'))

    def test_selecting_never_applies_and_apply_is_guarded(self):
        automations={a['id']:a for a in self.package['automation']}
        select=automations['engineathome_select_target_profile']
        self.assertEqual([a['action'] for a in select['actions']],['rest_command.engineathome_select_profile','homeassistant.update_entity'])
        commands=self.package['rest_command']
        self.assertTrue(commands['engineathome_select_profile']['url'].endswith('/admin/profile/select'))
        self.assertTrue(commands['engineathome_apply_profile']['url'].endswith('/admin/profile/apply'))
        used=json.dumps(automations)
        self.assertNotIn('rest_command.engineathome_apply_profile',used)       # no automation can switch the model
        self.assertNotIn('script.engineathome_apply_profile',used)
        text=(self.root/'dashboard.yaml').read_text(encoding='utf-8')
        apply_card=text[text.index('script.engineathome_apply_profile')-1200:text.index('script.engineathome_apply_profile')+700]
        for guard in ('engineathome_selection_state','Pending','engineathome_lifecycle_protection','confirmation:'):self.assertIn(guard,apply_card)

    def test_reasoning_and_default_profile_changes_are_requests_to_the_gateway_only(self):
        commands=self.package['rest_command']
        self.assertEqual(commands['engineathome_set_reasoning']['method'],'patch')
        self.assertIn('reasoning_effort',commands['engineathome_set_reasoning']['payload'])
        self.assertTrue(commands['engineathome_set_default_profile']['url'].endswith('/admin/settings'))

    def test_secret_only_in_private_secrets_file(self):
        for name in ('package.yaml', 'dashboard.yaml', 'themes/engineathome.yaml'):
            self.assertNotIn(SECRET, (self.root / name).read_text(encoding='utf-8'))
        self.assertIn(SECRET, (self.root / 'secrets.yaml').read_text(encoding='utf-8'))
        self.assertTrue(all(r['headers']['Authorization'] == 'SECRET:engineathome_ha_authorization' for r in self.package['rest']))

    def test_lifecycle_controls_are_confirmed_and_cache_loss_is_stated(self):
        text = (self.root / 'dashboard.yaml').read_text(encoding='utf-8')
        self.assertEqual(text.count('confirmation:'), 3)      # apply, restart, unload
        self.assertIn('discards the prefix cache', text)
        self.assertIn('never interrupt running requests', text)

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
    env.filters['tojson'] = lambda value, indent=None: json.dumps(value, indent=indent)   # same signature as Jinja's filter, which HA 2026.10 uses
    env.globals['states'] = lambda entity: states.get(entity, 'unknown')
    env.globals['state_attr'] = lambda entity, attribute: (attrs.get(entity) or {}).get(attribute)
    return env


def request(started, status='complete', **values):
    row = dict(id='x', client='opencode', model='fixture-swift', profile='swift', effort='xhigh', started=started, status=status, error=None,
               input_tokens=1000, cached_tokens=600, fresh_input_tokens=400, output_tokens=50, total_tokens=1050, prefill_tps=800.123,
               decode_tps=44.456, ttft_s=1.234, latency_s=9.87)
    row.update(values)
    return row


def summary_row(period, client='opencode', **values):
    row = dict(period=period, client=client, model='fixture-swift', profile='swift', requests=3, completed=3, incomplete_or_failed=0, usage_reported_requests=3,
               input_tokens=3000, output_tokens=150, total_tokens=3150, cached_tokens=1800, fresh_input_tokens=1200, cache_known_requests=3,
               mean_latency_s=2.5, mean_ttft_s=0.8)
    row.update(values)
    return row


CATALOG = [registry.public_profile(k, p) for k, p in CONFIG['profiles'].items()]


class TemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        generate(Path(cls.tmp.name), 'http://private.example:8080', SECRET, CONFIG)
        dashboard = yaml.load((Path(cls.tmp.name) / 'dashboard.yaml').read_text(encoding='utf-8'), Loader)
        cls.cards = {}
        for node in walk(dashboard):
            if node.get('type') == 'markdown':cls.cards[node['content'].split('\n')[0][:40] + str(len(cls.cards))] = node['content']

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def render(self, content, states, attrs):
        return environment(states, attrs).from_string(content).render()

    def all_states(self, rows, sensor_state='3', selected='swift', active='swift', applied='swift', levels=None):
        sensors = ('usage_history', 'request_history', 'recent_requests', 'usage_trend')
        states = {f'sensor.engineathome_{s}': sensor_state for s in sensors}
        spec = CONFIG['profiles'][applied]['reasoning']
        states.update({'input_select.engineathome_status': 'All', 'input_select.engineathome_token_view': 'Total', 'input_select.engineathome_group_by': 'day',
                       'sensor.engineathome_context_capacity': '262144', 'sensor.engineathome_profiles': '3',
                       'sensor.engineathome_selected_profile': selected, 'sensor.engineathome_active_profile_key': active,
                       'sensor.engineathome_applied_profile': applied, 'sensor.engineathome_default_profile_key': 'swift',
                       'sensor.engineathome_active_model': CONFIG['profiles'][active]['display_name'] if active != 'none' else 'No model loaded',
                       'sensor.engineathome_applied_profile_name': CONFIG['profiles'][applied]['display_name'],
                       'sensor.engineathome_reasoning_default': spec['default']})
        attrs = {f'sensor.engineathome_{s}': dict(rows=rows, timezone='Europe/Paris') for s in sensors}
        attrs['sensor.engineathome_profiles'] = dict(profiles=CATALOG)
        attrs['sensor.engineathome_model_state'] = dict(api_base_url='http://private.example:8080/v1')
        attrs['sensor.engineathome_reasoning_default'] = dict(levels=list(spec['levels']) if levels is None else levels, default=spec['default'], default_source='profile',
                                                              parameter='reasoning_effort', request_override=True, note=spec['note'])
        return states, attrs

    def svg(self, text):
        match = re.search(r'data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)', text)
        self.assertTrue(match, 'chart image missing')
        return ET.fromstring(base64.b64decode(match.group(1)))

    def test_every_card_renders_with_populated_nullable_empty_and_unavailable_data(self):
        t = 1_795_000_000
        populated = [summary_row('2026-11-21T10:00:00+01:00'), summary_row('2026-11-21T11:00:00+01:00', client='hermes', cached_tokens=None, fresh_input_tokens=None),
                     summary_row('2026-11-21T12:00:00+01:00', output_tokens=None, total_tokens=None)]
        requests = [request(t), request(t - 90, status='failed', error='backend_http_500', prefill_tps=None, decode_tps=None, ttft_s=None, input_tokens=None, cached_tokens=None, effort=None)]
        for rows, state in ((populated + [r for r in requests], '3'), ([], '0'), ([], 'unavailable')):
            for selected, active, applied in (('swift', 'swift', 'swift'), ('inkling-maximum', 'swift', 'swift'), ('inkling-balanced', 'none', 'inkling-balanced'),
                                              ('swift', 'inkling-maximum', 'inkling-maximum')):
                states, attrs = self.all_states(populated if rows else [], state, selected, active, applied)
                for key in ('recent_requests', 'request_history'):attrs[f'sensor.engineathome_{key}'] = dict(rows=rows and requests, timezone='Europe/Paris')
                for name, content in self.cards.items():
                    text = self.render(content, states, attrs)
                    self.assertNotIn('None', text.replace('\n', ' ').split('data:image')[0], name)
                    self.assertNotIn('nan', text.lower().split('data:image')[0].replace('financ', ''), name)

    def test_selected_model_card_distinguishes_target_from_running_model_and_warns_about_the_cache(self):
        card = next(c for c in self.cards.values() if 'Selected target' in c and 'Running now' in c)
        states, attrs = self.all_states([], selected='inkling-maximum', active='swift', applied='swift')
        text = self.render(card, states, attrs)
        self.assertIn('Inkling Maximum', text);self.assertIn('fixture-maximum', text);self.assertIn('262,144', text)
        self.assertIn('Applying this stops Swift Production and discards its prefix cache', text)
        self.assertIn('| Microbatch | 256 | 128 |', text);self.assertIn('| KV cache | q8_0 | q4_0 |', text)
        states, attrs = self.all_states([], selected='swift', active='swift', applied='swift')
        self.assertIn('This is the model that is running', self.render(card, states, attrs))
        states, attrs = self.all_states([], selected='inkling-balanced', active='none', applied='swift')
        self.assertIn('No model is loaded', self.render(card, states, attrs))

    def test_thinking_card_lists_only_the_loaded_models_levels(self):
        card = next(c for c in self.cards.values() if 'Available levels' in c)
        states, attrs = self.all_states([], active='swift', applied='swift')
        text = self.render(card, states, attrs)
        self.assertIn('`none` `low` `medium` `xhigh`', text);self.assertNotIn('`minimal`', text)
        self.assertIn('Swift Production', text);self.assertIn('does not reload the model', text);self.assertIn('not a measurement', text)
        states, attrs = self.all_states([], selected='inkling-balanced', active='inkling-balanced', applied='inkling-balanced')
        text = self.render(card, states, attrs)
        self.assertIn('`minimal`', text);self.assertIn('Effective default: **high**', text);self.assertIn('Inkling Balanced', text)
        states, attrs = self.all_states([], levels=[])
        self.assertIn('unavailable', self.render(card, states, attrs))

    def test_connect_card_gives_a_valid_opencode_snippet_without_any_secret(self):
        card = next(c for c in self.cards.values() if '```json' in c)
        states, attrs = self.all_states([], selected='inkling-maximum', active='inkling-maximum', applied='inkling-maximum')
        text = self.render(card, states, attrs)
        snippet = json.loads(re.search(r'```json\n(.*?)\n```', text, re.S).group(1))
        model = snippet['provider']['engineathome']['models']['fixture-maximum']
        self.assertEqual((model['limit']['context'], snippet['model']), (262144, 'engineathome/fixture-maximum'))
        self.assertEqual(snippet['provider']['engineathome']['options']['apiKey'], '{env:ENGINEATHOME_API_KEY}')
        self.assertIn('http://private.example:8080/v1', text)
        self.assertNotIn(SECRET, text)

    def test_models_and_reasoning_tables_show_every_profile(self):
        states, attrs = self.all_states([])
        models = self.render(next(c for c in self.cards.values() if 'Loaded VRAM' in c), states, attrs)
        for needle in ('Swift Production (default)', 'Inkling Balanced', 'Inkling Maximum', 'fixture-balanced', '204,800', 'q8_0'):self.assertIn(needle, models)
        reasoning = self.render(next(c for c in self.cards.values() if '| Profile | Parameter |' in c), states, attrs)
        self.assertIn('none, low, medium, xhigh', reasoning);self.assertIn('none, minimal, low, medium, high, xhigh', reasoning);self.assertIn('0.0–0.99', reasoning)

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

    def test_status_filter_applies_to_request_table_and_effort_is_shown(self):
        t = 1_795_000_000
        rows = [request(t), request(t - 60, status='failed', error='backend_http_500', effort='low')]
        states, attrs = self.all_states(rows)
        states['input_select.engineathome_status'] = 'failed'
        card = next(c for c in self.cards.values() if 'Started' in c and 'input_select.engineathome_status' in c)
        out = self.render(card, states, attrs)
        self.assertIn('backend_http_500', out);self.assertIn('| low |', out)
        self.assertEqual(out.count('● complete'), 0)
        self.assertIn('**1** matching request', out)


if __name__ == '__main__':unittest.main()
