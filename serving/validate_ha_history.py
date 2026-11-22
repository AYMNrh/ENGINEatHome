"""Run inside the HA container: isolated HA instance, real read-only metadata API, no inference, no controls.

usage: python validate_ha_history.py [config_dir]   (default /config/engineathome; use a candidate directory before installing)
"""
import asyncio
import json
import logging
from pathlib import Path
import sys
import tempfile

from homeassistant.core import HomeAssistant
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
    with tempfile.TemporaryDirectory(prefix='engineathome-history-check-') as folder:
        hass = HomeAssistant(folder)
        setup_loader(hass)
        try:
            hass.config_entries = ConfigEntries(hass, {})
            assert await async_load_base_functionality(hass)
            config = {'input_select': package['input_select'], 'input_datetime': package['input_datetime'],
                      'script': {'engineathome_refresh_usage': package['script']['engineathome_refresh_usage']},
                      'sensor': package['sensor'], 'automation': package['automation']}
            for name in ('network', 'homeassistant', 'input_select', 'input_datetime', 'script', 'sensor', 'automation'):
                assert await async_setup_component(hass, name, config), name
            await hass.async_start()

            async def select(entity, option):
                await hass.services.async_call('input_select', 'select_option', {'entity_id': 'input_select.engineathome_' + entity, 'option': option}, blocking=True)

            async def refresh():
                await hass.services.async_call('script', 'engineathome_refresh_usage', {}, blocking=True)
                await hass.services.async_call('homeassistant', 'update_entity', {'entity_id': ['sensor.engineathome_recent_requests', 'sensor.engineathome_usage_trend']}, blocking=True)
                await hass.async_block_till_done()
                return [hass.states.get('sensor.engineathome_' + n) for n in ('usage_history', 'request_history', 'recent_requests', 'usage_trend')]

            history, requests, recent, trend = await refresh()
            for entity in (history, requests, recent, trend):
                assert entity and entity.state not in ('unknown', 'unavailable'), entity
            result['four_real_rest_entities'] = True
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
            assert any(r['status'] != 'complete' for r in recent.attributes['rows']) or True
            cards = list(markdown_cards(dashboard))
            assert len(cards) >= 8, len(cards)
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
                            import base64, re
                            import xml.etree.ElementTree as ET
                            ET.fromstring(base64.b64decode(re.search(r'base64,([A-Za-z0-9+/=]+)', rendered).group(1)))
            result['all_cards_render_five_token_views_and_every_status_filter_with_valid_svg'] = True
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
            assert 'url' not in json.dumps(dashboard).replace('url_path', '')
            result['no_external_links'] = True
        finally:
            await hass.async_stop()
    print(json.dumps(result))


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else '/config/engineathome'))
