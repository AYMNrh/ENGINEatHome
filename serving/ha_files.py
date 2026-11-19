"""Native Home Assistant REST package and YAML dashboard, with secrets separated."""
from pathlib import Path


def generate(root,address,model,token):
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
             ('GPU Temperature','value_json.gpu.temperature_c','°C','temperature'),
             ('GPU Utilization','value_json.gpu.utilization_percent','%',None),
             ('GPU VRAM Used','value_json.gpu.used / 1073741824','GiB',None),
             ('GPU VRAM Free','value_json.gpu.free / 1073741824','GiB',None),
             ('GPU Power','value_json.gpu.power_watts','W','power'),
             ('GPU Power Limit','value_json.gpu.power_limit_watts','W','power'),
             ('Model Load Seconds','value_json.model.load_seconds','s','duration'),
             ('Model Uptime','value_json.model.uptime_seconds','s','duration'),
             ('Last TTFT','value_json.last_inference.ttft_s','s','duration'),
             ('Last Prefill','value_json.last_inference.prefill_tps','tokens/s',None),
             ('Last Decode','value_json.last_inference.decode_tps','tokens/s',None)]
    for name,expression,unit,device in metrics:
        package+=f'      - name: ENGINEatHome {name}\n        unique_id: engineathome_{name.lower().replace(" ","_")}\n'
        if 'value_json.gpu.' in expression:package+='        availability: "{{ value_json.gpu is not none }}"\n'
        if 'last_inference' in expression:package+='        availability: "{{ value_json.last_inference is not none and '+expression+' is not none }}"\n'
        if expression in ('value_json.model.load_seconds','value_json.model.uptime_seconds'):package+='        availability: "{{ '+expression+' is not none }}"\n'
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
    package+='\nscript:\n'
    for action in ('load','unload','restart'):
        package+=f'  engineathome_{action}:\n    alias: ENGINEatHome {action.title()}\n    sequence:\n      - action: rest_command.engineathome_{action}\n      - action: homeassistant.update_entity\n        target:\n          entity_id: sensor.engineathome_model_state\n'
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
    (root/'package.yaml').write_text(package,encoding='utf-8')
    (root/'secrets.yaml').write_text('engineathome_ha_authorization: "Bearer '+token+'"\n',encoding='utf-8')
    (root/'secrets.yaml').chmod(0o600)
    dashboard=f'''title: ENGINEatHome
views:
  - title: Serving
    path: serving
    icon: mdi:brain
    cards:
      - type: markdown
        title: Stock Swift API
        content: |
          **Private OpenAI base URL**
          ```text
          {address}/v1
          ```
          **Model ID**
          ```text
          {model}
          ```
          Keys are held in protected files, never in sensor states.
      - type: horizontal-stack
        cards:
'''
    for action in ('load','unload','restart'):
        dashboard+=f'          - type: button\n            name: {action.title()}\n            icon: mdi:{"play" if action=="load" else "stop" if action=="unload" else "restart"}\n            tap_action:\n              action: perform-action\n              perform_action: script.engineathome_{action}\n'
    dashboard+='''      - type: entities
        title: Model and GPU
        entities:
          - sensor.engineathome_model_state
          - sensor.engineathome_active_requests
          - switch.engineathome_auto_unload
          - sensor.engineathome_model_load_seconds
          - sensor.engineathome_model_uptime
          - sensor.engineathome_gpu_temperature
          - sensor.engineathome_gpu_utilization
          - sensor.engineathome_gpu_vram_used
          - sensor.engineathome_gpu_vram_free
          - sensor.engineathome_gpu_power
          - sensor.engineathome_gpu_power_limit
          - sensor.engineathome_last_ttft
          - sensor.engineathome_last_prefill
          - sensor.engineathome_last_decode
      - type: entities
        title: Today — input includes cached; output includes reasoning
        entities:
          - sensor.engineathome_input_today
          - sensor.engineathome_fresh_today
          - sensor.engineathome_cached_today
          - sensor.engineathome_output_today
          - sensor.engineathome_total_today
          - sensor.engineathome_requests_today
          - sensor.engineathome_usage_reported_requests_today
      - type: history-graph
        title: GPU telemetry
        hours_to_show: 24
        entities:
          - sensor.engineathome_gpu_temperature
          - sensor.engineathome_gpu_power
          - sensor.engineathome_gpu_vram_used
'''
    dashboard+=f'''      - type: button
        name: Filter token history by time, client and model
        icon: mdi:chart-bar
        tap_action:
          action: url
          url_path: {address}/analytics
'''
    (root/'dashboard.yaml').write_text(dashboard,encoding='utf-8')
