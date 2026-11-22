"""Run inside the HA container: isolated HA instance against the real read-only gateway API. It never changes the gateway.

usage: python validate_ha_history.py [config_dir]   (default /config/engineathome; use a candidate directory before installing)

Control services (rest_command.*) are replaced by recording stubs, so the model-selection, reasoning-default and idle-timeout
automations can be exercised without touching the production gateway. Status, usage and request sensors read the real API.
"""
import asyncio
import base64
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import xml.etree.ElementTree as ET

from homeassistant.core import HomeAssistant, ServiceCall, SupportsResponse
from homeassistant.bootstrap import async_load_base_functionality
from homeassistant.config_entries import ConfigEntries
from homeassistant.setup import async_setup_component
from homeassistant.util.yaml import load_yaml
from homeassistant.helpers.template import Template
from homeassistant.loader import async_setup as setup_loader
from annotatedyaml.loader import Secrets


def markdown_cards(node):
    if isinstance(node, dict):
        if node.get('type') == 'markdown' and 'content' in node:yield node
        for value in node.values():yield from markdown_cards(value)
    elif isinstance(node, list):
        for value in node:yield from markdown_cards(value)


async def main(folder_name):
    folder_path = Path(folder_name)
    package = load_yaml(str(folder_path / 'package.yaml'), Secrets(folder_path))
    dashboard = load_yaml(str(folder_path / 'dashboard.yaml'))
    result = {}
    calls = []
    with tempfile.TemporaryDirectory(prefix='engineathome-history-check-') as folder:
        hass = HomeAssistant(folder)
        setup_loader(hass)
        try:
            hass.config_entries = ConfigEntries(hass, {})
            assert await async_load_base_functionality(hass)

            async def stub(call: ServiceCall):
                calls.append((call.service, call.data))
            for name in package['rest_command']:hass.services.async_register('rest_command', name, stub)
            config = {name: package[name] for name in ('input_select', 'input_datetime', 'script', 'sensor', 'automation', 'template', 'rest')}
            for name in ('network', 'homeassistant', 'input_select', 'input_datetime', 'script', 'sensor', 'rest', 'template', 'automation'):
                assert await async_setup_component(hass, name, config), name
            await hass.async_start()
            await hass.async_block_till_done()

            async def select(entity, option):
                await hass.services.async_call('input_select', 'select_option', {'entity_id': 'input_select.engineathome_' + entity, 'option': option}, blocking=True)

            async def refresh():
                await hass.services.async_call('script', 'engineathome_refresh_usage', {}, blocking=True)
                await hass.services.async_call('homeassistant', 'update_entity', {'entity_id': ['sensor.engineathome_recent_requests', 'sensor.engineathome_usage_trend', 'sensor.engineathome_model_state']}, blocking=True)
                await hass.async_block_till_done()
                return [hass.states.get('sensor.engineathome_' + n) for n in ('usage_history', 'request_history', 'recent_requests', 'usage_trend')]

            history, requests, recent, trend = await refresh()
            for entity in (history, requests, recent, trend):
                assert entity and entity.state not in ('unknown', 'unavailable'), entity
            result['four_real_rest_entities'] = True

            # ---- profile / status sensors against the real gateway
            def state(name):
                return hass.states.get('sensor.engineathome_' + name)
            for name in ('model_state', 'selected_profile', 'applied_profile', 'active_profile_key', 'default_profile_key', 'switch_phase', 'switch_outcome',
                         'reasoning_default', 'profiles', 'selected_profile_name', 'applied_profile_name', 'default_profile_name', 'active_model',
                         'selection_state', 'gateway_health', 'runtime_state', 'lifecycle_protection', 'context_capacity'):
                assert state(name) and state(name).state not in ('unknown', 'unavailable'), name
            profiles = state('profiles').attributes['profiles']
            assert {p['key'] for p in profiles} >= {'swift'}, profiles
            names = {p['display_name'] for p in profiles}
            assert set(package['input_select']['engineathome_target_profile']['options']) == names, (names, package['input_select']['engineathome_target_profile']['options'])
            assert state('selected_profile_name').state in names
            assert state('reasoning_default').attributes['levels'], 'reasoning levels'
            result['profile_and_reasoning_sensors_real'] = dict(active=state('active_model').state, selected=state('selected_profile').state, levels=state('reasoning_default').attributes['levels'])

            # ---- selecting records a target but can never apply
            await hass.async_block_till_done()
            other = next(p for p in profiles if p['key'] != state('selected_profile').state)
            await select('target_profile', other['display_name'])
            await hass.async_block_till_done()
            assert any(service == 'engineathome_select_profile' for service, _ in calls), calls
            assert not any(service == 'engineathome_apply_profile' for service, _ in calls), calls
            result['target_change_calls_select_only'] = True
            calls.clear()
            await select('target_profile', next(p['display_name'] for p in profiles if p['key'] == state('selected_profile').state))   # back to the gateway's truth
            await hass.async_block_till_done()
            await hass.services.async_call('script', 'engineathome_apply_profile', {}, blocking=True)
            assert [service for service, _ in calls if service == 'engineathome_apply_profile'] == ['engineathome_apply_profile']
            result['apply_script_is_the_only_apply_path'] = True

            # ---- reasoning default: options follow the loaded model; a level change is sent once; unsupported levels cannot be chosen
            levels = state('reasoning_default').attributes['levels']
            await asyncio.sleep(1.5)
            await hass.async_block_till_done()
            options = hass.states.get('input_select.engineathome_reasoning_effort').attributes['options']
            assert options == levels, (options, levels)
            assert hass.states.get('input_select.engineathome_reasoning_effort').state == state('reasoning_default').state
            result['reasoning_options_follow_loaded_model'] = options
            calls.clear()
            choice = next(level for level in levels if level != state('reasoning_default').state)
            await select('reasoning_effort', choice)
            await asyncio.sleep(2.2)
            await hass.async_block_till_done()
            assert [s for s, _ in calls if s == 'engineathome_set_reasoning'] == ['engineathome_set_reasoning'], calls
            result['reasoning_change_sent_once'] = True
            calls.clear()
            unsupported = next((l for l in ('minimal', 'high', 'xhigh', 'low') if l not in levels), None)
            if unsupported:
                try:
                    await select('reasoning_effort', unsupported)
                    raise AssertionError('unsupported level accepted')
                except Exception as error:
                    assert 'AssertionError' not in type(error).__name__ or 'unsupported' not in str(error)
                result['unsupported_levels_cannot_be_selected'] = unsupported

            clients = history.attributes['clients']
            if clients:
                client = clients[0]
                model = history.attributes['model']
                await hass.services.async_call('input_select', 'set_options', {'entity_id': 'input_select.engineathome_client', 'options': ['All', *clients]}, blocking=True)
                await select('client', client)
                await select('model', model)
                for group in ('hour', 'day', 'week', 'month'):
                    await select('group_by', group)
                    history, requests, recent, trend = await refresh()
                    assert all(r['client'] == client and r['model'] == model for r in history.attributes['rows'])
                    assert all(r['client'] == client and r['model'] == model for r in requests.attributes['rows'])
                result['client_model_and_four_bucket_filters'] = True
                rows, detail = history.attributes['rows'], requests.attributes['rows']
                if len(detail) < 500:
                    for field in ('input_tokens', 'cached_tokens', 'output_tokens', 'total_tokens', 'fresh_input_tokens'):
                        assert sum(r[field] or 0 for r in rows) == sum(r[field] or 0 for r in detail), field
                    result['summary_matches_independent_request_sums'] = True
                assert all(r.get('profile') for r in detail) or not detail, 'every historical request is attributed to a profile'
                result['requests_carry_profile_attribution'] = True
            cards = list(markdown_cards(dashboard))
            assert len(cards) >= 12, len(cards)
            statuses = package['input_select']['engineathome_status']['options']
            for view in ('Total', 'Input', 'Output', 'Cached', 'Fresh'):
                await select('token_view', view)
                for status in statuses:
                    await select('status', status)
                    for card in cards:
                        rendered = Template(card['content'], hass).async_render(parse_result=False)
                        assert isinstance(rendered, str) and 'TemplateError' not in rendered and 'UndefinedError' not in rendered
                        assert 'None' not in rendered.split('data:image')[0].replace('NONE', ''), rendered[:300]
                        if 'data:image/svg+xml;base64,' in rendered:
                            ET.fromstring(base64.b64decode(re.search(r'base64,([A-Za-z0-9+/=]+)', rendered).group(1)))
            result['all_cards_render_five_token_views_and_every_status_filter_with_valid_svg'] = True
            connect = next(c for c in cards if '```json' in c['content'])
            text = Template(connect['content'], hass).async_render(parse_result=False)
            snippet = json.loads(re.search(r'```json\n(.*?)\n```', text, re.S).group(1))
            assert snippet['model'].startswith('engineathome/') and 'env:ENGINEATHOME_API_KEY' in json.dumps(snippet) and 'eah_' not in text
            result['opencode_snippet_valid_json_without_secrets'] = snippet['model']
            await select('status', 'All')
            await select('time_range', 'Custom')
            for suffix, stamp in (('from', '2000-01-01 00:00:00'), ('to', '2000-01-02 00:00:00')):
                await hass.services.async_call('input_datetime', 'set_datetime', {'entity_id': 'input_datetime.engineathome_' + suffix, 'datetime': stamp}, blocking=True)
            history, requests, recent, trend = await refresh()
            assert history.attributes['rows'] == [] and requests.attributes['rows'] == []
            for card in cards:
                Template(card['content'], hass).async_render(parse_result=False)
            result['custom_dates_and_empty_range'] = True
            await hass.services.async_call('input_datetime', 'set_datetime', {'entity_id': 'input_datetime.engineathome_to', 'datetime': '1999-12-31 00:00:00'}, blocking=True)
            history, requests, recent, trend = await refresh()
            assert history.state == 'unavailable' and requests.state == 'unavailable'
            for card in cards:
                Template(card['content'], hass).async_render(parse_result=False)
            result['invalid_dates_mark_data_unavailable_and_cards_still_render'] = True
            assert 'type: url' not in json.dumps(dashboard) and '"action": "url"' not in json.dumps(dashboard)
            result['no_external_links'] = True
        finally:
            await hass.async_stop()
    print(json.dumps(result))


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else '/config/engineathome'))
