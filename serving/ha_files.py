"""Native Home Assistant REST package, theme and YAML dashboard, with secrets separated.

Everything profile-related (names, model IDs, levels, launch settings) is taken from the gateway's real profile registry
(`profiles.py`), so the dashboard never advertises a model or thinking level the backend does not have.
"""
from pathlib import Path
import json

from ha_dashboard import THEME, dashboard
import profiles as registry

GATEWAY = 'http://127.0.0.1:8080'
IDLE = {'15 min': 15, '30 min': 30, '1 hour': 60, '2 hours': 120, '4 hours': 240, '8 hours': 480, '24 hours': 1440}


def history_query(endpoint):
    """Render filters in HA, keeping the gateway credential entirely server-side."""
    return '''{%- set range = states('input_select.engineathome_time_range') -%}
{%- set end = as_timestamp(now()) -%}
{%- if range == 'Custom' -%}
  {%- set start = as_timestamp(states('input_datetime.engineathome_from'), 0) -%}
  {%- set end = as_timestamp(states('input_datetime.engineathome_to'), 0) -%}
{%- elif range == 'Today' -%}
  {%- set start = as_timestamp(today_at()) -%}
{%- elif range == 'Last 24 hours' -%}
  {%- set start = end - 86400 -%}
{%- elif range == 'Last 30 days' -%}
  {%- set start = end - 2592000 -%}
{%- else -%}
  {%- set start = end - 604800 -%}
{%- endif -%}
{%- set client = states('input_select.engineathome_client') -%}
{%- set model = states('input_select.engineathome_model') -%}
{{- 'http://127.0.0.1:8080/analytics/'''+endpoint+'''?' ~ (dict(start=start, end=end,
  bucket=states('input_select.engineathome_group_by') if states('input_select.engineathome_group_by') in ['hour','day','week','month'] else 'day',
  client=client if client not in ['All', 'unknown', 'unavailable'] else '',
  model=model if model not in ['All', 'unknown', 'unavailable'] else '') | urlencode) -}}'''


def window_query(endpoint, seconds, bucket):
    """Fixed unfiltered window ending now; independent of the Requests-view filters."""
    return ("{{- 'http://127.0.0.1:8080/analytics/"+endpoint+"?' ~ (dict(start=as_timestamp(now()) - "+str(seconds)+
            ", end=as_timestamp(now()) + 60, bucket='"+bucket+"') | urlencode) -}}")


def history_package():
    text='\nsensor:\n'
    sensors=[('summary','Usage History',history_query('summary'),60),('requests','Request History',history_query('requests'),60),
             ('requests','Recent Requests',window_query('requests',86400,'day'),30),
             ('summary','Usage Trend',window_query('summary',86400,'hour'),60)]
    for endpoint,name,query,interval in sensors:
        text+='  - platform: rest\n    resource_template: >-\n'+''.join('      '+line+'\n' for line in query.splitlines())
        text+=f'''    headers:
      Authorization: !secret engineathome_ha_authorization
    scan_interval: {interval}
    timeout: 15
'''
        text+=f'''    name: ENGINEatHome {name}
    unique_id: engineathome_{name.lower().replace(' ','_')}
    availability: "{{{{ value_json.rows is defined }}}}"
    value_template: "{{{{ value_json.rows | count }}}}"
    json_attributes:
      - rows
      - clients
      - model
      - models
      - timezone
      - accounting
      - unknown
'''
    return text


def jinja_dict(mapping, q="'"):
    """A Jinja dict literal. q is the quote character: single quotes inside YAML double-quoted strings and block scalars,
    double quotes inside YAML single-quoted strings (the rest_command payloads). Names contain no quote characters."""
    def lit(value):
        return value if isinstance(value, (int, float)) else q + str(value) + q
    return '{' + ', '.join(f'{lit(k)}: {lit(v)}' for k, v in mapping.items()) + '}'


def history_helpers(catalog):
    names = [p['display_name'] for p in catalog]
    models = ['All'] + [p['model_id'] for p in catalog]
    levels = []
    for p in catalog:
        for level in p['reasoning']['levels']:
            if level not in levels:levels.append(level)
    by_name = jinja_dict({p['display_name']: p['key'] for p in catalog})
    by_key = jinja_dict({p['key']: p['display_name'] for p in catalog})
    idle_by_label = jinja_dict(IDLE)
    idle_by_minutes = jinja_dict({v: k for k, v in IDLE.items()})
    return f'''
input_select:
  engineathome_time_range:
    name: Time range
    options: [Last 7 days, Today, Last 24 hours, Last 30 days, Custom]
    icon: mdi:calendar-range
  engineathome_group_by:
    name: Chart buckets
    options: [day, hour, week, month]
  engineathome_client:
    name: Client
    options: [All, opencode, hermes, python]
  engineathome_model:
    name: Model or profile
    options: {json.dumps(models)}
  engineathome_token_view:
    name: Token category
    options: [Total, Input, Output, Cached, Fresh]
  engineathome_idle_timeout:
    name: Idle unload timeout
    options: [15 min, 30 min, 1 hour, 2 hours, 4 hours, 8 hours, 24 hours]
    icon: mdi:timer-outline
  engineathome_status:
    name: Request status
    options: [All, complete, failed, incomplete, interrupted, timeout, client_disconnected]
    icon: mdi:filter-variant
  engineathome_target_profile:
    name: Target model
    options: {json.dumps(names)}
    icon: mdi:cube-outline
  engineathome_default_profile:
    name: Default launch profile
    options: {json.dumps(names)}
    icon: mdi:star-outline
  engineathome_reasoning_effort:
    name: Default thinking effort
    options: {json.dumps(levels)}
    icon: mdi:head-cog-outline
input_datetime:
  engineathome_from:
    name: Usage from
    has_date: true
    has_time: true
  engineathome_to:
    name: Usage to
    has_date: true
    has_time: true
automation:
  - id: engineathome_refresh_usage_filters
    alias: ENGINEatHome refresh usage filters
    mode: restart
    triggers:
      - trigger: state
        entity_id:
          - input_select.engineathome_time_range
          - input_select.engineathome_group_by
          - input_select.engineathome_client
          - input_select.engineathome_model
          - input_datetime.engineathome_from
          - input_datetime.engineathome_to
    actions:
      - action: homeassistant.update_entity
        target:
          entity_id:
            - sensor.engineathome_usage_history
            - sensor.engineathome_request_history
  - id: engineathome_discover_usage_clients
    alias: ENGINEatHome discover usage clients
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_usage_history
    conditions:
      - condition: template
        value_template: >-
          {{{{ states('input_select.engineathome_client') not in ['unknown', 'unavailable'] and
             state_attr('sensor.engineathome_usage_history', 'clients') is not none and
             state_attr('input_select.engineathome_client', 'options') !=
             (['All'] + state_attr('sensor.engineathome_usage_history', 'clients')) }}}}
    actions:
      - action: input_select.set_options
        target:
          entity_id: input_select.engineathome_client
        data:
          options: "{{{{ ['All'] + state_attr('sensor.engineathome_usage_history', 'clients') }}}}"
  - id: engineathome_discover_usage_models
    alias: ENGINEatHome discover usage models
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_usage_history
    conditions:
      - condition: template
        value_template: >-
          {{{{ state_attr('sensor.engineathome_usage_history', 'models') is not none and
             state_attr('input_select.engineathome_model', 'options') is not none and
             state_attr('sensor.engineathome_usage_history', 'models') | reject('in', state_attr('input_select.engineathome_model', 'options')) | list | count > 0 }}}}
    actions:
      - action: input_select.set_options
        target:
          entity_id: input_select.engineathome_model
        data:
          options: >-
            {{{{ (['All'] + (state_attr('input_select.engineathome_model', 'options')[1:] + state_attr('sensor.engineathome_usage_history', 'models')) | unique | list) }}}}
  - id: engineathome_apply_idle_timeout
    alias: ENGINEatHome apply idle timeout
    mode: restart
    triggers:
      - trigger: state
        entity_id: input_select.engineathome_idle_timeout
        not_from: [unavailable, unknown]
    conditions:
      - condition: template
        value_template: "{{{{ {idle_by_label}.get(trigger.to_state.state, 0) not in [0, states('sensor.engineathome_idle_unload_minutes') | int(-1)] }}}}"
    actions:
      - action: rest_command.engineathome_set_idle
      - action: homeassistant.update_entity
        target:
          entity_id: sensor.engineathome_idle_unload_minutes
  - id: engineathome_sync_idle_timeout
    alias: ENGINEatHome show idle timeout in effect
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_idle_unload_minutes
        not_to: [unavailable, unknown]
      - trigger: homeassistant
        event: start
      - trigger: time_pattern
        minutes: "/5"
    conditions:
      - condition: template
        value_template: >-
          {{% set m = {idle_by_minutes} %}}
          {{% set actual = states('sensor.engineathome_idle_unload_minutes') | int(0) %}}
          {{{{ actual in m and states('input_select.engineathome_idle_timeout') != m[actual] }}}}
    actions:
      - action: input_select.select_option
        target:
          entity_id: input_select.engineathome_idle_timeout
        data:
          option: >-
            {{% set m = {idle_by_minutes} %}}
            {{{{ m[states('sensor.engineathome_idle_unload_minutes') | int(0)] }}}}
  - id: engineathome_select_target_profile
    alias: ENGINEatHome record target model (never switches)
    mode: restart
    triggers:
      - trigger: state
        entity_id: input_select.engineathome_target_profile
        not_from: [unavailable, unknown]
    conditions:
      - condition: template
        value_template: "{{{{ {by_name}.get(states('input_select.engineathome_target_profile')) not in [none, states('sensor.engineathome_selected_profile')] }}}}"
    actions:
      - action: rest_command.engineathome_select_profile
      - action: homeassistant.update_entity
        target:
          entity_id: sensor.engineathome_model_state
  - id: engineathome_sync_target_profile
    alias: ENGINEatHome show the gateway's selected target
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_selected_profile
        not_to: [unavailable, unknown]
      - trigger: homeassistant
        event: start
      - trigger: time_pattern
        minutes: "/2"
    conditions:
      - condition: template
        value_template: >-
          {{% set names = {by_key} %}}
          {{{{ states('sensor.engineathome_selected_profile') in names and states('input_select.engineathome_target_profile') != names[states('sensor.engineathome_selected_profile')] }}}}
    actions:
      - action: input_select.select_option
        target:
          entity_id: input_select.engineathome_target_profile
        data:
          option: >-
            {{% set names = {by_key} %}}
            {{{{ names[states('sensor.engineathome_selected_profile')] }}}}
  - id: engineathome_apply_default_profile
    alias: ENGINEatHome set the default launch profile (never switches)
    mode: restart
    triggers:
      - trigger: state
        entity_id: input_select.engineathome_default_profile
        not_from: [unavailable, unknown]
    conditions:
      - condition: template
        value_template: "{{{{ {by_name}.get(states('input_select.engineathome_default_profile')) not in [none, states('sensor.engineathome_default_profile_key')] }}}}"
    actions:
      - action: rest_command.engineathome_set_default_profile
      - action: homeassistant.update_entity
        target:
          entity_id: sensor.engineathome_model_state
  - id: engineathome_sync_default_profile
    alias: ENGINEatHome show the gateway's default profile
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_default_profile_key
        not_to: [unavailable, unknown]
      - trigger: homeassistant
        event: start
      - trigger: time_pattern
        minutes: "/5"
    conditions:
      - condition: template
        value_template: >-
          {{% set names = {by_key} %}}
          {{{{ states('sensor.engineathome_default_profile_key') in names and states('input_select.engineathome_default_profile') != names[states('sensor.engineathome_default_profile_key')] }}}}
    actions:
      - action: input_select.select_option
        target:
          entity_id: input_select.engineathome_default_profile
        data:
          option: >-
            {{% set names = {by_key} %}}
            {{{{ names[states('sensor.engineathome_default_profile_key')] }}}}
  - id: engineathome_apply_reasoning_default
    alias: ENGINEatHome set the default thinking effort (no reload)
    mode: restart
    triggers:
      - trigger: state
        entity_id: input_select.engineathome_reasoning_effort
        not_from: [unavailable, unknown]
    actions:
      # Wait for the options/value sync below to settle, then judge the *current* selection.
      - delay: "00:00:01"
      - if:
          - condition: template
            value_template: >-
              {{{{ states('input_select.engineathome_reasoning_effort') in (state_attr('sensor.engineathome_reasoning_default', 'levels') or []) and
                 states('input_select.engineathome_reasoning_effort') != states('sensor.engineathome_reasoning_default') }}}}
        then:
          - action: rest_command.engineathome_set_reasoning
          - action: homeassistant.update_entity
            target:
              entity_id: sensor.engineathome_model_state
  - id: engineathome_sync_reasoning_default
    alias: ENGINEatHome offer only the loaded model's thinking levels
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_reasoning_default
        not_to: [unavailable, unknown]
      - trigger: state
        entity_id: sensor.engineathome_reasoning_default
        attribute: levels
      - trigger: homeassistant
        event: start
      - trigger: time_pattern
        minutes: "/2"
    conditions:
      - condition: template
        value_template: "{{{{ (state_attr('sensor.engineathome_reasoning_default', 'levels') or []) | count > 0 }}}}"
    actions:
      - if:
          - condition: template
            value_template: "{{{{ state_attr('input_select.engineathome_reasoning_effort', 'options') != state_attr('sensor.engineathome_reasoning_default', 'levels') }}}}"
        then:
          - action: input_select.set_options
            target:
              entity_id: input_select.engineathome_reasoning_effort
            data:
              options: "{{{{ state_attr('sensor.engineathome_reasoning_default', 'levels') }}}}"
      - if:
          - condition: template
            value_template: "{{{{ states('input_select.engineathome_reasoning_effort') != states('sensor.engineathome_reasoning_default') }}}}"
        then:
          - action: input_select.select_option
            target:
              entity_id: input_select.engineathome_reasoning_effort
            data:
              option: "{{{{ states('sensor.engineathome_reasoning_default') }}}}"
recorder:
  exclude:
    entities:
      - sensor.engineathome_usage_history
      - sensor.engineathome_request_history
      - sensor.engineathome_recent_requests
      - sensor.engineathome_usage_trend
      - sensor.engineathome_profiles
'''


# (name, expression, unit, device class); expressions are Jinja over the /admin/status JSON.
METRICS = [
    ('Active Requests', 'value_json.model.active_requests', None, None),
    ('GPU Temperature', 'value_json.gpu.temperature_c | round(0) | int', '°C', 'temperature'),
    ('GPU Utilization', 'value_json.gpu.utilization_percent | round(0) | int', '%', None),
    ('GPU VRAM Used', '(value_json.gpu.used / 1073741824) | round(2)', 'GiB', None),
    ('GPU VRAM Free', '(value_json.gpu.free / 1073741824) | round(2)', 'GiB', None),
    ('GPU Power', 'value_json.gpu.power_watts | round(0) | int', 'W', 'power'),
    ('GPU Power Limit', 'value_json.gpu.power_limit_watts | round(0) | int', 'W', 'power'),
    ('GPU Sample Time', "value_json.gpu_sample_at | timestamp_custom('%Y-%m-%dT%H:%M:%S+00:00', false)", None, 'timestamp'),
    ('Model Load Seconds', 'value_json.model.load_seconds | round(0) | int', 's', None),
    ('Model Uptime', '(value_json.model.uptime_seconds / 60) | round(0) | int', 'min', None),
    ('Gateway Uptime', '(value_json.gateway.uptime_seconds / 60) | round(0) | int', 'min', None),
    ('Context Capacity', 'value_json.model.context_capacity', 'tokens', None),
    ('Session Leases', 'value_json.model.session_leases', None, None),
    ('Idle Unload Minutes', '(value_json.idle_seconds / 60) | round(0) | int', 'min', None),
    ('Last Error', "value_json.model.last_error or 'none'", None, None),
    ('Selected Profile', 'value_json.profile.selected', None, None),
    ('Active Profile Key', "value_json.profile.active or 'none'", None, None),
    ('Applied Profile', 'value_json.profile.applied', None, None),
    ('Default Profile Key', 'value_json.profile.default', None, None),
    ('Switch Phase', "(value_json.profile.switching or {}).get('phase', 'idle')", None, None),
    ('Switch Outcome', "'none' if not value_json.profile.last_switch else ('ok' if value_json.profile.last_switch.ok else 'failed')", None, None),
    ('Switch Detail',
     "'No switch since the gateway started' if not value_json.profile.last_switch else ('Switched to ' ~ value_json.profile.last_switch.to ~ ' in ' ~ value_json.profile.last_switch.seconds ~ ' s' if value_json.profile.last_switch.ok else 'Switch to ' ~ value_json.profile.last_switch.to ~ ' failed: ' ~ value_json.profile.last_switch.error ~ ('; restored ' ~ value_json.profile.last_switch.restored if value_json.profile.last_switch.restored else '; nothing was restored'))",
     None, None),
    ('Last TTFT', 'value_json.last_inference.ttft_s | round(2)', 's', None),
    ('Last Prefill', 'value_json.last_inference.prefill_tps | round(1)', 'tokens/s', None),
    ('Last Decode', 'value_json.last_inference.decode_tps | round(1)', 'tokens/s', None),
    ('Last Request Latency', 'value_json.last_inference.latency_s | round(1)', 's', None),
    ('Last Request Status', 'value_json.last_inference.status', None, None),
    ('Last Request Time', "value_json.last_inference.finished | timestamp_custom('%Y-%m-%dT%H:%M:%S+00:00', false)", None, 'timestamp'),
    ('Last Cached Tokens', 'value_json.last_inference.cached_tokens', 'tokens', None),
    ('Prompt Cache Limit', 'value_json.cache.ram_limit_mib', 'MiB', None),
    ('Runtime RAM Used', 'value_json.cache.process_rss_mib', 'MiB', None),
    ('Host RAM Available', 'value_json.cache.host_available_mib', 'MiB', None),
    ('Recent Cache Hit Rate', 'value_json.cache.recent.hit_rate_pct', '%', None),
    ('Cached Follow-up TTFT', 'value_json.cache.recent.cached_ttft_s', 's', None),
    ('Cold Prompt TTFT', 'value_json.cache.recent.cold_ttft_s', 's', None),
    ('Recent Prefill Speed', 'value_json.cache.recent.prefill_tps', 'tokens/s', None),
    ('Recent Decode Speed', 'value_json.cache.recent.decode_tps', 'tokens/s', None),
]


def rest_package():
    package = '''rest:
  - resource: http://127.0.0.1:8080/admin/status
    headers:
      Authorization: !secret engineathome_ha_authorization
    scan_interval: 15
    timeout: 10
    sensor:
      - name: ENGINEatHome Model State
        unique_id: engineathome_model_state
        value_template: "{{ value_json.model.state }}"
        json_attributes_path: "$.model"
        json_attributes:
          - model_id
          - active_model_id
          - api_base_url
          - context_capacity
          - load_seconds
          - uptime_seconds
          - last_request
          - last_error
          - session_leases
      - name: ENGINEatHome Auto Unload State
        unique_id: engineathome_auto_unload_state
        value_template: "{{ 'on' if value_json.auto_unload else 'off' }}"
      - name: ENGINEatHome Reasoning Default
        unique_id: engineathome_reasoning_default
        value_template: "{{ value_json.reasoning.configured_default }}"
        json_attributes_path: "$.reasoning"
        json_attributes:
          - profile
          - parameter
          - levels
          - default
          - configured_default
          - default_source
          - aliases
          - numeric
          - request_override
          - note
      - name: ENGINEatHome Cache
        unique_id: engineathome_cache
        value_template: "{{ 'on' if value_json.cache.enabled else 'off' }}"
        json_attributes_path: "$.cache"
        json_attributes:
          - profile
          - enabled
          - ram_limit_mib
          - label
          - recent
          - host_total_mib
          - host_available_mib
          - process_rss_mib
          - note
      - name: ENGINEatHome Profiles
        unique_id: engineathome_profiles
        value_template: "{{ value_json.profiles | count }}"
        json_attributes:
          - profiles
'''
    for name, expression, unit, device in METRICS:
        package += f'      - name: ENGINEatHome {name}\n        unique_id: engineathome_{name.lower().replace(" ", "_")}\n'
        if 'value_json.gpu.' in expression:package += '        availability: "{{ value_json.gpu is not none }}"\n'
        base = expression.split(' |')[0]
        if 'value_json.cache.' in expression:package += '        availability: "{{ value_json.cache is not none and ' + base + ' is not none }}"\n'
        if 'last_inference' in expression and 'profile' not in expression:
            package += '        availability: "{{ value_json.last_inference is not none and ' + base + ' is not none }}"\n'
        for field in ('value_json.model.load_seconds', 'value_json.model.uptime_seconds', 'value_json.gpu_sample_at'):
            if field in expression:package += '        availability: "{{ ' + field + ' is not none }}"\n'
        package += '        value_template: "{{ ' + expression.replace('"', "'") + ' }}"\n'
        if unit:package += f'        unit_of_measurement: "{unit}"\n        state_class: measurement\n'
        if device:package += f'        device_class: {device}\n'
    package += '''  - resource: http://127.0.0.1:8080/analytics/totals
    headers:
      Authorization: !secret engineathome_ha_authorization
    scan_interval: 60
    sensor:
'''
    for name, key in (('Input', 'input_tokens'), ('Output', 'output_tokens'), ('Cached', 'cached_tokens'), ('Fresh', 'fresh_input_tokens'),
                      ('Total', 'total_tokens'), ('Requests', 'requests'), ('Usage Reported Requests', 'usage_reported_requests')):
        package += f'      - name: ENGINEatHome {name} Today\n        unique_id: engineathome_{key}_today\n        value_template: "{{{{ value_json.periods.day.{key} }}}}"\n        state_class: total_increasing\n'
    return package


def commands_package(catalog):
    by_name = jinja_dict({p['display_name']: p['key'] for p in catalog}, '"')
    package = '\nrest_command:\n'

    def command(name, path, method, payload, timeout=120):
        return (f'  engineathome_{name}:\n    url: {GATEWAY}{path}\n    method: {method}\n    timeout: {timeout}\n    headers:\n'
                '      Authorization: !secret engineathome_ha_authorization\n    content_type: application/json\n'
                f'    payload: {payload}\n')
    for action in ('load', 'unload', 'restart'):
        package += command(action, '/admin/' + action, 'post', "'{}'")
    for state in ('on', 'off'):
        package += command('auto_' + state, '/admin/settings', 'patch', f'\'{{"auto_unload": {"true" if state == "on" else "false"}}}\'')
    package += command('set_idle', '/admin/settings', 'patch',
                       "'{{ {\"idle_seconds\": " + jinja_dict(IDLE, '"') + ".get(states(\"input_select.engineathome_idle_timeout\"), 60) * 60} | tojson }}'")
    package += command('select_profile', '/admin/profile/select', 'post',
                       "'{{ {\"profile\": " + by_name + ".get(states(\"input_select.engineathome_target_profile\"))} | tojson }}'")
    package += command('apply_profile', '/admin/profile/apply', 'post', "'{}'", 300)
    package += command('set_default_profile', '/admin/settings', 'patch',
                       "'{{ {\"default_profile\": " + by_name + ".get(states(\"input_select.engineathome_default_profile\"))} | tojson }}'")
    package += command('set_reasoning', '/admin/settings', 'patch',
                       "'{{ {\"reasoning_effort\": states(\"input_select.engineathome_reasoning_effort\")} | tojson }}'")
    package += '\nscript:\n'
    refresh = '      - action: homeassistant.update_entity\n        target:\n          entity_id: sensor.engineathome_model_state\n'
    for action in ('load', 'unload', 'restart'):
        package += f'  engineathome_{action}:\n    alias: ENGINEatHome {action.title()}\n    sequence:\n      - action: rest_command.engineathome_{action}\n        continue_on_error: true\n' + refresh
    package += ('  engineathome_apply_profile:\n    alias: ENGINEatHome apply the selected model\n    sequence:\n'
                '      - action: rest_command.engineathome_apply_profile\n        continue_on_error: true\n' + refresh)
    package += '''  engineathome_refresh_usage:
    alias: ENGINEatHome refresh usage
    sequence:
      - action: homeassistant.update_entity
        target:
          entity_id:
            - sensor.engineathome_usage_history
            - sensor.engineathome_request_history
'''
    return package


def template_package(catalog):
    names = jinja_dict({p['key']: p['display_name'] for p in catalog})
    models = jinja_dict({p['key']: p['model_id'] for p in catalog})

    def name_sensor(title, source, none_text):
        return f'''      - name: ENGINEatHome {title}
        unique_id: engineathome_{title.lower().replace(' ', '_')}
        icon: mdi:cube-outline
        state: >-
          {{% set names = {names} %}}
          {{% set k = states('sensor.engineathome_{source}') %}}
          {{% if k in ['unavailable', 'unknown'] %}}Unknown{{% elif k == 'none' %}}{none_text}{{% else %}}{{{{ names.get(k, k) }}}}{{% endif %}}
'''
    package = '''
template:
  - switch:
      - name: ENGINEatHome Auto Unload
        unique_id: engineathome_auto_unload
        availability: "{{ states('sensor.engineathome_auto_unload_state') in ['on', 'off'] }}"
        state: "{{ is_state('sensor.engineathome_auto_unload_state', 'on') }}"
        turn_on:
          - action: rest_command.engineathome_auto_on
          - action: homeassistant.update_entity
            target:
              entity_id: sensor.engineathome_auto_unload_state
        turn_off:
          - action: rest_command.engineathome_auto_off
          - action: homeassistant.update_entity
            target:
              entity_id: sensor.engineathome_auto_unload_state
  - sensor:
      - name: ENGINEatHome GPU VRAM Used Percent
        unique_id: engineathome_gpu_vram_used_percent
        unit_of_measurement: "%"
        state_class: measurement
        availability: "{{ states('sensor.engineathome_gpu_vram_used') | is_number and states('sensor.engineathome_gpu_vram_free') | is_number }}"
        state: >-
          {{ (states('sensor.engineathome_gpu_vram_used') | float / (states('sensor.engineathome_gpu_vram_used') | float + states('sensor.engineathome_gpu_vram_free') | float) * 100) | round(0) | int }}
      - name: ENGINEatHome Cache Hit Rate Today
        unique_id: engineathome_cache_hit_rate_today
        unit_of_measurement: "%"
        availability: "{{ states('sensor.engineathome_input_today') | is_number and states('sensor.engineathome_input_today') | float > 0 and states('sensor.engineathome_cached_today') | is_number }}"
        state: "{{ (states('sensor.engineathome_cached_today') | float / states('sensor.engineathome_input_today') | float * 100) | round(0) | int }}"
      - name: ENGINEatHome Range Tokens
        unique_id: engineathome_range_tokens
        unit_of_measurement: tokens
        availability: "{{ state_attr('sensor.engineathome_usage_history', 'rows') is not none and states('sensor.engineathome_usage_history') not in ['unknown', 'unavailable'] and state_attr('sensor.engineathome_usage_history', 'rows') | map(attribute='total_tokens') | reject('none') | list | count > 0 }}"
        state: "{{ state_attr('sensor.engineathome_usage_history', 'rows') | map(attribute='total_tokens') | reject('none') | sum }}"
      - name: ENGINEatHome Gateway Health
        unique_id: engineathome_gateway_health
        icon: mdi:lan-connect
        state: >-
          {% if states('sensor.engineathome_model_state') in ['unavailable', 'unknown'] %}Unreachable
          {%- elif states('sensor.engineathome_model_state') == 'failed' or states('sensor.engineathome_gpu_temperature') in ['unavailable', 'unknown'] or states('sensor.engineathome_switch_outcome') == 'failed' %}Degraded
          {%- else %}Healthy{% endif %}
      - name: ENGINEatHome Runtime State
        unique_id: engineathome_runtime_state
        icon: mdi:power
        state: >-
          {% set s = states('sensor.engineathome_model_state') %}
          {{ 'Unreachable' if s in ['unavailable', 'unknown'] else s | capitalize }}
''' + name_sensor('Selected Profile Name', 'selected_profile', 'None') + name_sensor('Applied Profile Name', 'applied_profile', 'None') \
        + name_sensor('Default Profile Name', 'default_profile_key', 'None') + f'''      - name: ENGINEatHome Active Model
        unique_id: engineathome_active_model
        icon: mdi:cube-outline
        state: >-
          {{% set names = {names} %}}
          {{% set s = states('sensor.engineathome_model_state') %}}
          {{% set k = states('sensor.engineathome_active_profile_key') %}}
          {{% if s in ['unavailable', 'unknown'] %}}Unknown
          {{%- elif s == 'switching' %}}Switching…
          {{%- elif k in names %}}{{{{ names[k] }}}}
          {{%- else %}}No model loaded{{% endif %}}
        attributes:
          model_id: >-
            {{% set models = {models} %}}
            {{{{ models.get(states('sensor.engineathome_active_profile_key'), 'none') }}}}
      - name: ENGINEatHome Selection State
        unique_id: engineathome_selection_state
        icon: mdi:swap-horizontal-bold
        state: >-
          {{% if states('sensor.engineathome_selected_profile') in ['unavailable', 'unknown'] %}}Unknown
          {{%- elif states('sensor.engineathome_selected_profile') != states('sensor.engineathome_applied_profile') %}}Pending
          {{%- else %}}In sync{{% endif %}}
      - name: ENGINEatHome GPU Status
        unique_id: engineathome_gpu_status
        icon: mdi:expansion-card
        state: >-
          {{% set t = states('sensor.engineathome_gpu_temperature') %}}
          {{% set v = states('sensor.engineathome_gpu_vram_used_percent') %}}
          {{% if t | is_number and v | is_number %}}
          {{%- if t | float >= 85 %}}Hot{{% elif v | float >= 97 %}}Memory tight{{% elif t | float >= 75 %}}Warm{{% else %}}Normal{{% endif %}}
          {{%- else %}}Unavailable{{% endif %}}
      - name: ENGINEatHome Lifecycle Protection
        unique_id: engineathome_lifecycle_protection
        icon: mdi:shield-lock
        state: >-
          {{% set a = states('sensor.engineathome_active_requests') | int(0) %}}
          {{% set l = states('sensor.engineathome_session_leases') | int(0) %}}
          {{% if a > 0 or l > 0 %}}{{{{ a }}}} active, {{{{ l }}}} lease{{{{ 's' if l != 1 }}}} — unload, restart and model switches are refused{{% else %}}Clear{{% endif %}}
'''
    return package


def generate(root, address, token, config):
    """config: a gateway configuration (dict with profiles) or an already normalized one."""
    config = registry.normalize(config)
    catalog = [registry.public_profile(key, p) for key, p in config['profiles'].items()]
    root.mkdir(parents=True, exist_ok=True)
    package = rest_package() + history_package() + commands_package(catalog) + template_package(catalog) + history_helpers(catalog)
    (root / 'package.yaml').write_text(package, encoding='utf-8')
    (root / 'secrets.yaml').write_text('engineathome_ha_authorization: "Bearer ' + token + '"\n', encoding='utf-8')
    (root / 'secrets.yaml').chmod(0o600)
    (root / 'dashboard.yaml').write_text(dashboard(address, catalog, config['default_profile']), encoding='utf-8')
    theme = root / 'themes'
    theme.mkdir(exist_ok=True)
    (theme / 'engineathome.yaml').write_text(THEME, encoding='utf-8')
