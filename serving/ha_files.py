"""Native Home Assistant REST package and YAML dashboard, with secrets separated."""
from pathlib import Path
import json

from ha_dashboard import THEME, dashboard


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
      - timezone
      - accounting
      - unknown
'''
    return text


def history_helpers(model):
    return '''
input_select:
  engineathome_time_range:
    name: Usage time range
    options: [Last 7 days, Today, Last 24 hours, Last 30 days, Custom]
    icon: mdi:calendar-range
  engineathome_group_by:
    name: Group usage by
    options: [day, hour, week, month]
  engineathome_client:
    name: Usage client
    options: [All, opencode, hermes, python]
  engineathome_model:
    name: Usage model
    options: '''+json.dumps(['All',model])+'''
  engineathome_token_view:
    name: Token chart
    options: [Total, Input, Output, Cached, Fresh]
  engineathome_idle_timeout:
    name: Idle unload timeout
    options: [15 min, 30 min, 1 hour, 2 hours, 4 hours, 8 hours, 24 hours]
    icon: mdi:timer-outline
  engineathome_status:
    name: Request status
    options: [All, complete, failed, incomplete, interrupted, timeout, client_disconnected]
    icon: mdi:filter-variant
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
          {{ states('input_select.engineathome_client') not in ['unknown', 'unavailable'] and
             state_attr('sensor.engineathome_usage_history', 'clients') is not none and
             state_attr('input_select.engineathome_client', 'options') !=
             (['All'] + state_attr('sensor.engineathome_usage_history', 'clients')) }}
    actions:
      - action: input_select.set_options
        target:
          entity_id: input_select.engineathome_client
        data:
          options: "{{ ['All'] + state_attr('sensor.engineathome_usage_history', 'clients') }}"
  - id: engineathome_discover_usage_models
    alias: ENGINEatHome discover usage models
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.engineathome_usage_history
    conditions:
      - condition: template
        value_template: >-
          {{ state_attr('sensor.engineathome_usage_history', 'rows') is not none and
             state_attr('input_select.engineathome_model', 'options') is not none and
             (state_attr('sensor.engineathome_usage_history', 'rows') | map(attribute='model') | list | unique | list)
               | reject('in', state_attr('input_select.engineathome_model', 'options')) | list | count > 0 }}
    actions:
      - action: input_select.set_options
        target:
          entity_id: input_select.engineathome_model
        data:
          options: >-
            {{ (['All'] + (state_attr('input_select.engineathome_model', 'options')[1:] +
               (state_attr('sensor.engineathome_usage_history', 'rows') | map(attribute='model') | list)) | unique | list) }}
  - id: engineathome_apply_idle_timeout
    alias: ENGINEatHome apply idle timeout
    mode: restart
    triggers:
      - trigger: state
        entity_id: input_select.engineathome_idle_timeout
        not_from: [unavailable, unknown]
    conditions:
      - condition: template
        value_template: "{{ {'15 min':15,'30 min':30,'1 hour':60,'2 hours':120,'4 hours':240,'8 hours':480,'24 hours':1440}.get(trigger.to_state.state, 0) not in [0, states('sensor.engineathome_idle_unload_minutes') | int(-1)] }}"
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
          {% set m = {15:'15 min',30:'30 min',60:'1 hour',120:'2 hours',240:'4 hours',480:'8 hours',1440:'24 hours'} %}
          {% set actual = states('sensor.engineathome_idle_unload_minutes') | int(0) %}
          {{ actual in m and states('input_select.engineathome_idle_timeout') != m[actual] }}
    actions:
      - action: input_select.select_option
        target:
          entity_id: input_select.engineathome_idle_timeout
        data:
          option: >-
            {% set m = {15:'15 min',30:'30 min',60:'1 hour',120:'2 hours',240:'4 hours',480:'8 hours',1440:'24 hours'} %}
            {{ m[states('sensor.engineathome_idle_unload_minutes') | int(0)] }}
recorder:
  exclude:
    entities:
      - sensor.engineathome_usage_history
      - sensor.engineathome_request_history
      - sensor.engineathome_recent_requests
      - sensor.engineathome_usage_trend
'''


def runtime_facts(argv):
    """Launch settings as displayed on the Settings view, read from the gateway's real argv."""
    if not argv:return [('Source','not supplied when this dashboard was generated')]
    values={}
    for i,item in enumerate(argv):
        if item.startswith('--'):
            nxt=argv[i+1] if i+1<len(argv) and not argv[i+1].startswith('--') else None
            values[item[2:]]=nxt if nxt is not None else 'on'
    get=values.get
    kv=get('cache-type-k');vv=get('cache-type-v')
    rows=[('Model file',Path(get('model','')).name or 'unknown'),
          ('Context',f"{int(get('ctx-size',0)):,} tokens" if get('ctx-size') else 'unknown'),
          ('Parallel sequences',get('parallel','unknown')),
          ('GPU layers',get('n-gpu-layers','unknown')),
          ('Flash attention',get('flash-attn','unknown')),
          ('KV cache',kv if kv==vv else f'{kv} / {vv}'),
          ('Batch / microbatch',f"{get('batch-size','?')} / {get('ubatch-size','?')}"),
          ('Speculative decoding',f"{get('spec-type','none')}, up to {get('spec-draft-n-max','?')} draft tokens" if get('spec-type') else 'none'),
          ('Reasoning',f"{get('reasoning','?')}, effort {get('reasoning-effort','?')}, budget {get('reasoning-budget','?')}"),
          ('Prompt cache in RAM',get('cache-ram','unknown')),
          ('Context checkpoints',get('ctx-checkpoints','unknown')),
          ('Context shifting','off' if 'no-context-shift' in values else 'on')]
    return [(a,b) for a,b in rows if b is not None]


def generate(root,address,model,token,argv=None):
    root.mkdir(parents=True,exist_ok=True)
    package='''rest:
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
'''
    metrics=[('Active Requests','value_json.model.active_requests',None,None),
             ('GPU Temperature','value_json.gpu.temperature_c | round(0) | int','°C','temperature'),
             ('GPU Utilization','value_json.gpu.utilization_percent | round(0) | int','%',None),
             ('GPU VRAM Used','(value_json.gpu.used / 1073741824) | round(2)','GiB',None),
             ('GPU VRAM Free','(value_json.gpu.free / 1073741824) | round(2)','GiB',None),
             ('GPU Power','value_json.gpu.power_watts | round(0) | int','W','power'),
             ('GPU Power Limit','value_json.gpu.power_limit_watts | round(0) | int','W','power'),
             ('GPU Sample Time',"value_json.gpu_sample_at | timestamp_custom('%Y-%m-%dT%H:%M:%S+00:00', false)",None,'timestamp'),
             ('Model Load Seconds','value_json.model.load_seconds | round(0) | int','s',None),
             ('Model Uptime','(value_json.model.uptime_seconds / 60) | round(0) | int','min',None),
             ('Gateway Uptime','(value_json.gateway.uptime_seconds / 60) | round(0) | int','min',None),
             ('Context Capacity','value_json.model.context_capacity','tokens',None),
             ('Session Leases','value_json.model.session_leases',None,None),
             ('Idle Unload Minutes','(value_json.idle_seconds / 60) | round(0) | int','min',None),
             ('Last Error',"value_json.model.last_error or 'none'",None,None),
             ('Last TTFT','value_json.last_inference.ttft_s | round(2)','s',None),
             ('Last Prefill','value_json.last_inference.prefill_tps | round(1)','tokens/s',None),
             ('Last Decode','value_json.last_inference.decode_tps | round(1)','tokens/s',None),
             ('Last Request Latency','value_json.last_inference.latency_s | round(1)','s',None),
             ('Last Request Status','value_json.last_inference.status',None,None),
             ('Last Request Time',"value_json.last_inference.finished | timestamp_custom('%Y-%m-%dT%H:%M:%S+00:00', false)",None,'timestamp')]
    for name,expression,unit,device in metrics:
        package+=f'      - name: ENGINEatHome {name}\n        unique_id: engineathome_{name.lower().replace(" ","_")}\n'
        if 'value_json.gpu.' in expression:package+='        availability: "{{ value_json.gpu is not none }}"\n'
        base=expression.split(' |')[0]
        if 'last_inference' in expression:package+='        availability: "{{ value_json.last_inference is not none and '+base+' is not none }}"\n'
        for field in ('value_json.model.load_seconds','value_json.model.uptime_seconds','value_json.gpu_sample_at'):
            if field in expression:package+='        availability: "{{ '+field+' is not none }}"\n'
        package+='        value_template: "{{ '+expression+' }}"\n'
        if unit:package+=f'        unit_of_measurement: "{unit}"\n        state_class: measurement\n'
        if device:package+=f'        device_class: {device}\n'
    package+='''  - resource: http://127.0.0.1:8080/analytics/totals
    headers:
      Authorization: !secret engineathome_ha_authorization
    scan_interval: 60
    sensor:
'''
    for name,key in (('Input','input_tokens'),('Output','output_tokens'),('Cached','cached_tokens'),('Fresh','fresh_input_tokens'),('Total','total_tokens'),('Requests','requests'),('Usage Reported Requests','usage_reported_requests')):
        package+=f'      - name: ENGINEatHome {name} Today\n        unique_id: engineathome_{key}_today\n        value_template: "{{{{ value_json.periods.day.{key} }}}}"\n        state_class: total_increasing\n'
    package+=history_package()
    package+='\nrest_command:\n'
    for action in ('load','unload','restart'):
        package+=f'''  engineathome_{action}:
    url: http://127.0.0.1:8080/admin/{action}
    method: post
    timeout: 120
    headers:
      Authorization: !secret engineathome_ha_authorization
    content_type: application/json
    payload: '{{}}'
'''
    for state in ('on','off'):
        package+=f'''  engineathome_auto_{state}:
    url: http://127.0.0.1:8080/admin/settings
    method: patch
    headers:
      Authorization: !secret engineathome_ha_authorization
    content_type: application/json
    payload: '{{"auto_unload": {"true" if state=="on" else "false"}}}'
'''
    package+='''  engineathome_set_idle:
    url: http://127.0.0.1:8080/admin/settings
    method: patch
    headers:
      Authorization: !secret engineathome_ha_authorization
    content_type: application/json
    payload: '{{ {"idle_seconds": {"15 min":15,"30 min":30,"1 hour":60,"2 hours":120,"4 hours":240,"8 hours":480,"24 hours":1440}.get(states("input_select.engineathome_idle_timeout"), 60) * 60} | tojson }}'
'''
    package+='\nscript:\n'
    for action in ('load','unload','restart'):
        package+=f'  engineathome_{action}:\n    alias: ENGINEatHome {action.title()}\n    sequence:\n      - action: rest_command.engineathome_{action}\n      - action: homeassistant.update_entity\n        target:\n          entity_id: sensor.engineathome_model_state\n'
    package+='''  engineathome_refresh_usage:
    alias: ENGINEatHome refresh usage
    sequence:
      - action: homeassistant.update_entity
        target:
          entity_id:
            - sensor.engineathome_usage_history
            - sensor.engineathome_request_history
'''
    package+='''
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
'''
    package+='''  - sensor:
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
      - name: ENGINEatHome Gateway Health
        unique_id: engineathome_gateway_health
        icon: mdi:lan-connect
        state: >-
          {% if states('sensor.engineathome_model_state') in ['unavailable', 'unknown'] %}Unreachable
          {%- elif states('sensor.engineathome_model_state') == 'failed' or states('sensor.engineathome_gpu_temperature') in ['unavailable', 'unknown'] %}Degraded
          {%- else %}Healthy{% endif %}
      - name: ENGINEatHome Runtime State
        unique_id: engineathome_runtime_state
        icon: mdi:power
        state: >-
          {% set s = states('sensor.engineathome_model_state') %}
          {{ 'Unreachable' if s in ['unavailable', 'unknown'] else s | capitalize }}
      - name: ENGINEatHome Active Model
        unique_id: engineathome_active_model
        icon: mdi:cube-outline
        state: >-
          {% set s = states('sensor.engineathome_model_state') %}
          {% if s == 'ready' %}{{ state_attr('sensor.engineathome_model_state', 'model_id') }}
          {%- elif s in ['unavailable', 'unknown'] %}Unknown
          {%- elif s in ['loading', 'unloading'] %}Changing…
          {%- else %}No model loaded{% endif %}
      - name: ENGINEatHome GPU Status
        unique_id: engineathome_gpu_status
        icon: mdi:expansion-card
        state: >-
          {% set t = states('sensor.engineathome_gpu_temperature') %}
          {% set v = states('sensor.engineathome_gpu_vram_used_percent') %}
          {% if t | is_number and v | is_number %}
          {%- if t | float >= 85 %}Hot{% elif v | float >= 97 %}Memory tight{% elif t | float >= 75 %}Warm{% else %}Normal{% endif %}
          {%- else %}Unavailable{% endif %}
      - name: ENGINEatHome Lifecycle Protection
        unique_id: engineathome_lifecycle_protection
        icon: mdi:shield-lock
        state: >-
          {% set a = states('sensor.engineathome_active_requests') | int(0) %}
          {% set l = states('sensor.engineathome_session_leases') | int(0) %}
          {% if a > 0 or l > 0 %}{{ a }} active, {{ l }} lease{{ 's' if l != 1 }} — unload and restart refused{% else %}Clear{% endif %}
'''
    package+=history_helpers(model)
    (root/'package.yaml').write_text(package,encoding='utf-8')
    (root/'secrets.yaml').write_text('engineathome_ha_authorization: "Bearer '+token+'"\n',encoding='utf-8')
    (root/'secrets.yaml').chmod(0o600)
    (root/'dashboard.yaml').write_text(dashboard(address,model,runtime_facts(argv)),encoding='utf-8')
    theme=root/'themes';theme.mkdir(exist_ok=True)
    (theme/'engineathome.yaml').write_text(THEME,encoding='utf-8')
