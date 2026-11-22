"""ENGINEatHome multi-model inference console for Home Assistant: Overview, Requests, Settings.

Native Sections/Tile/Heading/Gauge/History cards only, plus a shipped theme and inline-SVG charts
rendered by HA templates from real gateway metadata. No HACS card, iframe or frontend resource.
Every referenced entity is defined in package.yaml (checked by test_ha_dashboard.py).
This module only renders configuration; it never touches inference, lifecycle logic or the gateway.
"""
import re
from textwrap import indent

ROUTE = '/engine-at-home'

# Chart palette: restrained, distinct from the green/amber/red status colours.
CACHED, FRESH, OUTPUT, SPLIT = '#6c8cff', '#35b6c8', '#b58cf0', '#7d8794'
MUTED = '#9aa3ad'

THEME = '''ENGINEatHome:
  primary-color: "#5b8def"
  accent-color: "#5b8def"
  primary-background-color: "#0f1114"
  secondary-background-color: "#15181c"
  card-background-color: "#1b1f24"
  ha-card-background: "#1b1f24"
  ha-card-border-width: 1px
  ha-card-border-color: "#2a3038"
  ha-card-border-radius: 14px
  ha-card-box-shadow: none
  primary-text-color: "#e8eaed"
  secondary-text-color: "#9aa3ad"
  text-primary-color: "#ffffff"
  disabled-text-color: "#6b7480"
  divider-color: "#2a3038"
  app-header-background-color: "#15181c"
  app-header-text-color: "#e8eaed"
  paper-item-icon-color: "#9aa3ad"
  state-icon-color: "#9aa3ad"
  success-color: "#3fb97f"
  warning-color: "#e8a33d"
  error-color: "#ef5b5b"
  info-color: "#5b8def"
  ha-view-sections-column-gap: 16px
  ha-view-sections-row-gap: 16px
  ha-color-form-background: "#232830"
  ha-color-form-background-hover: "#2a3038"
  ha-color-form-background-disabled: "#1b1f24"
  input-fill-color: "#232830"
  input-ink-color: "#e8eaed"
  input-label-ink-color: "#9aa3ad"
  input-idle-line-color: "#3a414b"
  input-hover-line-color: "#5b8def"
  input-disabled-ink-color: "#6b7480"
  input-dropdown-icon-color: "#9aa3ad"
  mdc-text-field-fill-color: "#232830"
  mdc-text-field-ink-color: "#e8eaed"
  mdc-text-field-label-ink-color: "#9aa3ad"
  mdc-select-fill-color: "#232830"
  mdc-select-ink-color: "#e8eaed"
  mdc-select-label-ink-color: "#9aa3ad"
  mdc-select-dropdown-icon-color: "#9aa3ad"
  mdc-select-idle-line-color: "#3a414b"
  mdc-theme-surface: "#1b1f24"
  mdc-theme-on-surface: "#e8eaed"
  mdc-theme-primary: "#5b8def"
  mdc-theme-text-primary-on-background: "#e8eaed"
  mdc-ripple-color: "#5b8def"
  switch-checked-color: "#5b8def"
  switch-checked-track-color: "#5b8def"
  switch-unchecked-color: "#9aa3ad"
  switch-unchecked-track-color: "#3a414b"
  slider-color: "#5b8def"
  table-row-background-color: "#1b1f24"
  table-row-alternative-background-color: "#1f242a"
  code-editor-background-color: "#15181c"
'''

# ----------------------------------------------------------------------------------------------
# Jinja building blocks
# ----------------------------------------------------------------------------------------------
NUM = ("{%- macro n(v) -%}{{ '—' if v is none else '{:,}'.format(v | int) }}{%- endmacro -%}\n"
       "{%- macro d(v, p=1) -%}{{ '—' if v is none else (v | float | round(p)) }}{%- endmacro -%}\n"
       "{%- macro k(v) -%}{%- if v >= 1000000 -%}{{ '%.1f' | format(v / 1000000) }}M{%- elif v >= 1000 -%}{{ '%.0f' | format(v / 1000) }}k{%- else -%}{{ v | int }}{%- endif -%}{%- endmacro -%}\n")

STATUS = ("{%- macro st(s) -%}{{ {'complete':'● complete','running':'◐ running','failed':'✕ failed','incomplete':'◌ incomplete',"
          "'interrupted':'◌ interrupted','timeout':'✕ timeout','client_disconnected':'◌ client left'}.get(s, s) }}{%- endmacro -%}\n")

ROW = ('| {{ r.started | timestamp_custom(\'%d %b %H:%M:%S\') }} | {{ r.client }} | {{ r.model }} | {{ r.effort or \'—\' }} | {{ st(r.status) }}'
       '{% if r.error %} `{{ r.error }}`{% endif %} | {{ n(r.input_tokens) }} | {{ n(r.cached_tokens) }} | {{ n(r.fresh_input_tokens) }} '
       '| {{ n(r.output_tokens) }} | {{ d(r.ttft_s, 2) }} | {{ d(r.prefill_tps) }} | {{ d(r.decode_tps) }} | {{ d(r.latency_s, 1) }} |')
HEAD = ('| Started | Client | Model | Effort | Status | Input | Cached | Fresh | Output | TTFT s | Prefill t/s | Decode t/s | Duration s |\n'
        '| :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |')

RECENT = NUM + STATUS + '''{% set rows = state_attr('sensor.engineathome_recent_requests', 'rows') or [] %}
{% if states('sensor.engineathome_recent_requests') in ['unknown','unavailable'] %}
Request metadata unavailable.
{% elif not rows %}
No requests in the last 24 hours.
{% else %}
| Time | Client | Model | Status | Input | Cached | Output | TTFT s | Prefill t/s | Decode t/s | Duration s |
| :-- | :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: |
{% for r in rows[:5] -%}
| {{ r.started | timestamp_custom('%H:%M:%S') }} | {{ r.client }} | {{ r.model }} | {{ st(r.status) }} | {{ n(r.input_tokens) }} | {{ n(r.cached_tokens) }} | {{ n(r.output_tokens) }} | {{ d(r.ttft_s, 2) }} | {{ d(r.prefill_tps) }} | {{ d(r.decode_tps) }} | {{ d(r.latency_s, 1) }} |
{% endfor %}
{% endif %}

<sub>Newest five of the last 24 h, each attributed to the model that served it. — means not reported. Metadata only; prompts, outputs and keys are never stored.</sub>

**[Open the full request explorer →](''' + ROUTE + '''/requests)**'''

REQUESTS = NUM + STATUS + '''{% set all = state_attr('sensor.engineathome_request_history', 'rows') or [] %}
{% set wanted = states('input_select.engineathome_status') %}
{% set rows = all if wanted in ['All','unknown','unavailable'] else all | selectattr('status', 'eq', wanted) | list %}
{% if states('sensor.engineathome_request_history') in ['unknown','unavailable'] %}
Request history unavailable. Check that a custom range ends after it starts and spans at most 367 days.
{% elif not rows %}
No requests match these filters.
{% else %}
**{{ rows | length }}** matching request{{ 's' if rows | length != 1 }}{% if all | length >= 500 %} (newest 500 searched){% endif %} · showing the latest {{ [rows | length, 50] | min }}

''' + HEAD + '''
{% for r in rows[:50] -%}
''' + ROW + '''
{% endfor %}
{% endif %}

<sub>Input includes cached; fresh = input − cached when both are reported. Nonstream TTFT is unobservable and interrupted streams may lack final usage — shown as —, never as zero.</sub>'''

PERIODS = NUM + '''{% set rows = state_attr('sensor.engineathome_usage_history', 'rows') or [] %}
{% if states('sensor.engineathome_usage_history') in ['unknown','unavailable'] %}
Usage unavailable for the selected range.
{% elif not rows %}
No requests match these filters.
{% else %}
Usage reported for **{{ n(rows | sum(attribute='usage_reported_requests')) }}** of **{{ n(rows | sum(attribute='requests')) }}** requests; cache split reported for **{{ n(rows | sum(attribute='cache_known_requests')) }}**. Buckets use **{{ state_attr('sensor.engineathome_usage_history', 'timezone') }}** time.

| Period | Client | Model | Requests | Input | Cached | Fresh | Output | Total | Mean TTFT s | Mean duration s |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
{% for r in rows | reverse -%}
| {{ r.period[:16] | replace('T', ' ') }} | {{ r.client }} | {{ r.model }} | {{ n(r.requests) }} | {{ n(r.input_tokens) }} | {{ n(r.cached_tokens) }} | {{ n(r.fresh_input_tokens) }} | {{ n(r.output_tokens) }} | {{ n(r.total_tokens) }} | {{ d(r.mean_ttft_s, 2) }} | {{ d(r.mean_latency_s, 1) }} |
{% endfor %}
{% endif %}'''

SUMMARY = NUM + '''{% set rows = state_attr('sensor.engineathome_usage_history', 'rows') or [] %}
{% if rows %}
**{{ n(rows | sum(attribute='requests')) }}** requests · **{{ n(rows | sum(attribute='completed')) }}** complete · **{{ n(rows | sum(attribute='incomplete_or_failed')) }}** not complete

Input **{{ n(rows | map(attribute='input_tokens') | reject('none') | sum) }}** of which cached **{{ n(rows | map(attribute='cached_tokens') | reject('none') | sum) }}** · Output **{{ n(rows | map(attribute='output_tokens') | reject('none') | sum) }}**
{% else %}
No requests match these filters.
{% endif %}'''


def token_chart(sensor, view, unit, empty):
    """Stacked bars: cached input, fresh input, uncategorised input, output — one unit (tokens), one axis."""
    return NUM + '''{% set rows = state_attr('@SENSOR@', 'rows') or [] %}
{% set view = @VIEW@ %}
{% set unit = @UNIT@ %}
{% if states('@SENSOR@') in ['unknown','unavailable'] %}
Usage unavailable.
{% elif not rows %}
@EMPTY@
{% else %}
{% set mask = {'Total':[1,1,1,1],'Input':[1,1,1,0],'Output':[0,0,0,1],'Cached':[1,0,0,0],'Fresh':[0,1,0,0]}.get(view, [1,1,1,1]) %}
{% set colors = ['@CACHED@','@FRESH@','@SPLIT@','@OUTPUT@'] %}
{% set names = ['Cached input','Fresh input','Input, split unreported','Output'] %}
{% set groups = (rows | groupby('period'))[-60:] %}
{% set ns = namespace(data=[], peak=1, requests=0, used=[0,0,0,0]) %}
{% for g in groups %}
  {% set l = g.list %}
  {% set c = l | map(attribute='cached_tokens') | reject('none') | sum %}
  {% set f = l | map(attribute='fresh_input_tokens') | reject('none') | sum %}
  {% set i = l | map(attribute='input_tokens') | reject('none') | sum %}
  {% set o = l | map(attribute='output_tokens') | reject('none') | sum %}
  {% set u = [i - c - f, 0] | max %}
  {% set v = [c * mask[0], f * mask[1], u * mask[2], o * mask[3]] %}
  {% set ns.data = ns.data + [[g.grouper, v]] %}
  {% set ns.used = [ns.used[0] + v[0], ns.used[1] + v[1], ns.used[2] + v[2], ns.used[3] + v[3]] %}
  {% set ns.peak = [ns.peak, v | sum] | max %}
  {% set ns.requests = ns.requests + (l | map(attribute='requests') | sum) %}
{% endfor %}
{% set count = ns.data | length %}
{% set left = 46 %}{% set top = 10 %}{% set pw = 664 %}{% set ph = 190 %}
{% set slot = pw / count %}{% set bar = [slot * 0.72, 2] | max %}
{% set ns.svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 262" font-family="sans-serif" font-size="11">' %}
{% for t in [0, 0.5, 1] %}
  {% set y = top + ph * (1 - t) %}
  {% set ns.svg = ns.svg ~ '<line x1="' ~ left ~ '" x2="710" y1="' ~ y ~ '" y2="' ~ y ~ '" stroke="@MUTED@" stroke-opacity=".25"/><text x="' ~ (left - 6) ~ '" y="' ~ (y + 4) ~ '" text-anchor="end" fill="@MUTED@">' ~ (k(ns.peak * t) | trim) ~ '</text>' %}
{% endfor %}
{% for item in ns.data %}
  {% set x = left + loop.index0 * slot + (slot - bar) / 2 %}
  {% set st = namespace(y=top + ph) %}
  {% for amount in item[1] %}
    {% set h = amount / ns.peak * ph %}
    {% if h > 0 %}
      {% set st.y = st.y - h %}
      {% set ns.svg = ns.svg ~ '<rect x="' ~ ('%.1f' | format(x)) ~ '" y="' ~ ('%.1f' | format(st.y)) ~ '" width="' ~ ('%.1f' | format(bar)) ~ '" height="' ~ ('%.1f' | format(h)) ~ '" rx="1.5" fill="' ~ colors[loop.index0] ~ '"/>' %}
    {% endif %}
  {% endfor %}
{% endfor %}
{% for ix in [0, (count - 1) // 2, count - 1] | unique | list %}
  {% set p = ns.data[ix][0] %}
  {% set text = p[:7] if unit == 'month' else (p[11:16] ~ ' ' ~ p[5:10]) if unit == 'hour' else p[5:10] %}
  {% set anchor = 'start' if ix == 0 and count > 1 else 'end' if ix == count - 1 and count > 1 else 'middle' %}
  {% set ns.svg = ns.svg ~ '<text x="' ~ ('%.1f' | format(left + ix * slot + slot / 2)) ~ '" y="' ~ (top + ph + 16) ~ '" text-anchor="' ~ anchor ~ '" fill="@MUTED@">' ~ text ~ '</text>' %}
{% endfor %}
{% set lg = namespace(x=left) %}
{% for name in names %}{% if mask[loop.index0] and ns.used[loop.index0] > 0 %}
  {% set ns.svg = ns.svg ~ '<rect x="' ~ lg.x ~ '" y="246" width="10" height="10" rx="2" fill="' ~ colors[loop.index0] ~ '"/><text x="' ~ (lg.x + 15) ~ '" y="255" fill="@MUTED@">' ~ name ~ '</text>' %}
  {% set lg.x = lg.x + 22 + name | length * 6.2 %}
{% endif %}{% endfor %}
{% set ns.svg = ns.svg ~ '</svg>' %}
![@TITLE@](data:image/svg+xml;base64,{{ ns.svg | base64_encode }})

<sub>{{ view }} tokens per {{ unit }}, all matching requests ({{ n(ns.requests) }}), {{ count }} period{{ 's' if count != 1 }}. Peak {{ k(ns.peak) | trim }}. Input includes cached; total = input + output; reasoning is part of output. Unreported counts are left out, not drawn as zero.</sub>
{% endif %}'''.replace('@SENSOR@', sensor).replace('@VIEW@', view).replace('@UNIT@', unit).replace('@EMPTY@', empty) \
        .replace('@CACHED@', CACHED).replace('@FRESH@', FRESH).replace('@OUTPUT@', OUTPUT).replace('@SPLIT@', SPLIT) \
        .replace('@MUTED@', MUTED).replace('@TITLE@', 'Token usage')


THROUGHPUT = NUM + '''{% set rows = state_attr('sensor.engineathome_recent_requests', 'rows') or [] %}
{% set pts = rows | selectattr('decode_tps', 'ne', none) | selectattr('prefill_tps', 'ne', none) | list | reverse | list %}
{% if states('sensor.engineathome_recent_requests') in ['unknown','unavailable'] %}
Throughput unavailable.
{% elif not pts %}
No throughput measurements in the last 24 hours.
{% else %}
{% set pts = pts[-300:] %}
{% set t0 = pts[0].started %}{% set t1 = pts[-1].started %}{% set span = [t1 - t0, 1] | max %}
{% set left = 46 %}{% set pw = 664 %}{% set ph = 84 %}
{% set ns = namespace(svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 262" font-family="sans-serif" font-size="11">') %}
{% for s in [['prefill_tps','Prefill','@PREFILL@',0],['decode_tps','Decode','@DECODE@',1]] %}
  {% set vals = pts | map(attribute=s[0]) | list %}
  {% set peak = [vals | max, 1] | max %}
  {% set y0 = 10 + s[3] * 112 %}
  {% for t in [0, 0.5, 1] %}
    {% set y = y0 + ph * (1 - t) %}
    {% set ns.svg = ns.svg ~ '<line x1="' ~ left ~ '" x2="710" y1="' ~ y ~ '" y2="' ~ y ~ '" stroke="@MUTED@" stroke-opacity=".25"/><text x="' ~ (left - 6) ~ '" y="' ~ (y + 4) ~ '" text-anchor="end" fill="@MUTED@">' ~ (k(peak * t) | trim) ~ '</text>' %}
  {% endfor %}
  {% set pl = namespace(points='', dots='') %}
  {% for p in pts %}
    {% set x = '%.1f' | format(left + loop.index0 / ([pts | length - 1, 1] | max) * pw) %}
    {% set y = '%.1f' | format(y0 + ph - p[s[0]] / peak * ph) %}
    {% set pl.points = pl.points ~ x ~ ',' ~ y ~ ' ' %}
    {% set pl.dots = pl.dots ~ '<circle cx="' ~ x ~ '" cy="' ~ y ~ '" r="2.4" fill="' ~ s[2] ~ '"/>' %}
  {% endfor %}
  {% set ns.svg = ns.svg ~ '<polyline points="' ~ pl.points ~ '" fill="none" stroke="' ~ s[2] ~ '" stroke-width="1.6" stroke-opacity=".75"/>' ~ pl.dots %}
  {% set ns.svg = ns.svg ~ '<rect x="' ~ (left + s[3] * 120) ~ '" y="246" width="10" height="10" rx="2" fill="' ~ s[2] ~ '"/><text x="' ~ (left + s[3] * 120 + 15) ~ '" y="255" fill="@MUTED@">' ~ s[1] ~ ' tok/s (own axis)</text>' %}
{% endfor %}
{% set ns.svg = ns.svg ~ '<text x="' ~ left ~ '" y="232" fill="@MUTED@">' ~ (t0 | timestamp_custom('%d %b %H:%M')) ~ '</text><text x="710" y="232" text-anchor="end" fill="@MUTED@">' ~ (t1 | timestamp_custom('%d %b %H:%M')) ~ '</text></svg>' %}
{% set pre = pts | map(attribute='prefill_tps') | list %}{% set dec = pts | map(attribute='decode_tps') | list %}
![Throughput](data:image/svg+xml;base64,{{ ns.svg | base64_encode }})

Mean prefill **{{ d(pre | sum / pre | length) }}** tok/s · mean decode **{{ d(dec | sum / dec | length) }}** tok/s · {{ pts | length }} measured request{{ 's' if pts | length != 1 }}

<sub>Each point is one completed request's native rate, oldest to newest (evenly spaced), last 24 h. Prefill and decode use separate axes. Requests without reported rates are omitted.</sub>
{% endif %}'''.replace('@PREFILL@', FRESH).replace('@DECODE@', OUTPUT).replace('@MUTED@', MUTED)


PROFILES = "{% set profiles = state_attr('sensor.engineathome_profiles', 'profiles') or [] %}"


def profile_lookup(var, key_expr):
    return f"{{% set {var} = (profiles | selectattr('key', 'eq', {key_expr}) | list | first) if (profiles | selectattr('key', 'eq', {key_expr}) | list) else none %}}"


SELECTED = NUM + PROFILES + '''
{% set sel = states('sensor.engineathome_selected_profile') %}
{% set act = states('sensor.engineathome_active_profile_key') %}
''' + profile_lookup('p', 'sel') + '\n' + profile_lookup('a', 'act') + '''
{% if p is none %}
Model details unavailable.
{% else %}
**{{ p.display_name }}** · `{{ p.model_id }}`

| | Selected target | Running now |
| :-- | :-- | :-- |
| Context | {{ n(p.context) }} tokens | {{ (n(a.context) ~ ' tokens') if a else '—' }} |
| Microbatch | {{ p.microbatch }} | {{ a.microbatch if a else '—' }} |
| KV cache | {{ p.kv_cache }} | {{ a.kv_cache if a else '—' }} |
| Prefix cache | {{ 'enabled' if p.prefix_cache else 'off' }} | {{ ('enabled' if a.prefix_cache else 'off') if a else '—' }} |
| Expected VRAM | {{ ('%.1f' | format(p.expected_vram_mib / 1024) ~ ' GiB') if p.expected_vram_mib else '—' }} | {{ ('%.1f' | format(a.expected_vram_mib / 1024) ~ ' GiB') if a and a.expected_vram_mib else '—' }} |
| Default thinking | {{ p.reasoning.default }} | {{ a.reasoning.default if a else '—' }} |

{{ p.description }}
{% if p.qualification %}

<sub>{{ p.qualification }}</sub>
{% endif %}
{% if sel == act %}

✓ This is the model that is running.
{% elif act != 'none' and a %}

⚠ **Applying this stops {{ a.display_name }} and discards its prefix cache.** The first request afterwards re-reads its whole prompt.
{% else %}

No model is loaded. Applying loads this one.
{% endif %}
{% endif %}'''

THINKING = PROFILES + '''
{% set levels = state_attr('sensor.engineathome_reasoning_default', 'levels') or [] %}
{% set source = state_attr('sensor.engineathome_reasoning_default', 'default_source') %}
{% if not levels %}
Thinking controls unavailable.
{% else %}
Applies to **{{ states('sensor.engineathome_active_model') if states('sensor.engineathome_active_profile_key') != 'none' else states('sensor.engineathome_applied_profile_name') }}**

Available levels: {% for l in levels %}`{{ l }}` {% endfor %}

Effective default: **{{ states('sensor.engineathome_reasoning_default') }}** ({% if source == 'dashboard' %}set on this dashboard; the model's own default is `{{ state_attr('sensor.engineathome_reasoning_default', 'default') }}`{% else %}the model's own default{% endif %})

Per-request override: **{{ 'supported' if state_attr('sensor.engineathome_reasoning_default', 'request_override') else 'not supported' }}** — send `{{ state_attr('sensor.engineathome_reasoning_default', 'parameter') }}` (or `chat_template_kwargs.{{ state_attr('sensor.engineathome_reasoning_default', 'parameter') }}`). A request's value affects only that request; unsupported values are rejected, never changed.

<sub>{{ state_attr('sensor.engineathome_reasoning_default', 'note') }} The effort is forwarded to the chat template as requested; it is not a measurement of how much the model thinks.</sub>

Higher effort may help difficult coding tasks, but can increase latency and generated tokens. Changing the default does not reload the model.
{% endif %}'''

CONNECT_SHORT = PROFILES + '''
{% set key = states('sensor.engineathome_applied_profile') %}
''' + profile_lookup('p', 'key') + '''
{% if p is none %}
Connection details unavailable.
{% else %}
**OpenAI base URL**
```text
{{ state_attr('sensor.engineathome_model_state', 'api_base_url') }}
```
**Model ID** · context {{ '{:,}'.format(p.context | int) }} tokens · default thinking {{ states('sensor.engineathome_reasoning_default') }}
```text
{{ p.model_id }}
```
<sub>Requests must name this model ID; another profile's ID is refused with 409 while this one is running. The OpenCode snippet is under Settings.</sub>
{% endif %}'''

CONNECT = PROFILES + '''
{% set key = states('sensor.engineathome_applied_profile') %}
''' + profile_lookup('p', 'key') + '''
{% if p is none %}
Connection details unavailable.
{% else %}
**OpenAI base URL**
```text
{{ state_attr('sensor.engineathome_model_state', 'api_base_url') }}
```
**Active model ID** · context {{ '{:,}'.format(p.context | int) }} tokens · default thinking {{ states('sensor.engineathome_reasoning_default') }}
```text
{{ p.model_id }}
```
**OpenCode** — merge into your config and set `ENGINEATHOME_API_KEY` in the launching environment (no key is ever shown here)
```json
{
  "provider": {
    "engineathome": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Homelab",
      "options": {
        "baseURL": "{{ state_attr('sensor.engineathome_model_state', 'api_base_url') }}",
        "apiKey": "{env:ENGINEATHOME_API_KEY}",
        "timeout": 14400000
      },
      "models": {
        "{{ p.model_id }}": {
          "name": "{{ p.display_name }}",
          "reasoning": true,
          "tool_call": true,
          "limit": { "context": {{ p.context | int }}, "output": 32768 }
        }
      }
    }
  },
  "model": "engineathome/{{ p.model_id }}"
}
```
<sub>Requests must name this model ID. Without an <code>options.reasoningEffort</code> on the model entry, the thinking default shown above applies; add <code>"options": {"reasoningEffort": "high"}</code> to pin a level for that client. Verified against OpenCode 1.18: it sends <code>reasoning_effort</code> only when configured (plus <code>low</code> on its own title requests).</sub>
{% endif %}'''


MODELS_TABLE = NUM + PROFILES + '''
{% if not profiles %}
Model list unavailable.
{% else %}
| Profile | Model ID | Context | KV | Microbatch | Loaded VRAM | Thinking default |
| :-- | :-- | --: | :-- | --: | --: | :-- |
{% for p in profiles -%}
| {{ p.display_name }}{% if p.key == states('sensor.engineathome_default_profile_key') %} (default){% endif %} | `{{ p.model_id }}` | {{ n(p.context) }} | {{ p.kv_cache }} | {{ p.microbatch }} | {{ ('%.1f' | format(p.expected_vram_mib / 1024) ~ ' GiB') if p.expected_vram_mib else '—' }} | {{ p.reasoning.default }} |
{% endfor %}

{% for p in profiles -%}
- **{{ p.display_name }}:** {{ p.qualification or p.description }}
{% endfor %}
{% endif %}'''

REASONING_TABLE = PROFILES + '''
{% if not profiles %}
Unavailable.
{% else %}
| Profile | Parameter | Levels | Default | Per-request override |
| :-- | :-- | :-- | :-- | :-- |
{% for p in profiles -%}
| {{ p.display_name }} | `{{ p.reasoning.parameter }}` | {{ p.reasoning.levels | join(', ') }}{% if p.reasoning.aliases %} (aliases: {% for k, v in p.reasoning.aliases.items() %}{{ k }}→{{ v }}{% endfor %}){% endif %}{% if p.reasoning.numeric %}; or a number {{ p.reasoning.numeric[0] }}–{{ p.reasoning.numeric[1] }}{% endif %} | {{ p.reasoning.default }} | {{ 'yes' if p.reasoning.request_override else 'no' }} |
{% endfor %}

{% for p in profiles -%}
- **{{ p.display_name }}:** {{ p.reasoning.note }}
{% endfor %}
{% endif %}

<sub>Only levels the model's own chat template accepts are offered. A request's value overrides the dashboard default for that request only.</sub>'''

LAUNCH = PROFILES + '''
{% set key = states('sensor.engineathome_selected_profile') %}
''' + profile_lookup('p', 'key') + '''
{% if p is none %}
Unavailable.
{% else %}
Launch settings of **{{ p.display_name }}** (the selected target):

| Setting | Value |
| :-- | :-- |
{% for row in p.launch -%}
| {{ row.label }} | {{ row.value }} |
{% endfor %}

<sub>Read from the gateway's configured launch arguments; executable paths and checksums are not shown.</sub>
{% endif %}'''

PROTECTIONS = '''**Switching models.** Choosing a target only records the choice. **Apply** checks that nothing is running, then stops the current model, waits for the GPU memory, starts the new one and sends it a real test request before reporting success. If that fails the previous model (or Swift) is restored and the reason is shown. Unload, Restart and Apply all discard the prefix cache, so the next request re-reads its entire prompt (a cold start with a long first-token wait).

**What is protected**
- Apply, Unload and Restart are refused while any request is running or queued, while a session lease is held, or while the native server is still processing (HTTP 409). Nothing is interrupted and leases are never bypassed.
- Only one model occupies the GPU at a time; only one lifecycle operation runs at a time.
- A request names exactly one model ID. A request for a model that is not the one being served is refused with `model_not_active`; it is never answered by a different model.
- The gateway admits at most 8 requests; the model decodes one sequence at a time. A request may run up to 4 hours.
- A load is refused unless the GPU power limit is 175 W, enough VRAM is free, no other CUDA workload is running and the model file is unchanged. A failed load waits 15 seconds before retrying.

**Privacy.** Only metadata is stored: counts, timings, client label, model, profile, forwarded thinking effort, status and error code. Prompts, outputs, reasoning text and API keys are never stored or shown here.'''


# ----------------------------------------------------------------------------------------------
# Card builders
# ----------------------------------------------------------------------------------------------
def block(text, spaces):
    return indent(text, ' ' * spaces)


def section(cards, span=1):
    # A section spanning N columns has a 12*N-unit grid, so scale the card widths to keep their proportions.
    cards = [re.sub(r'(columns: )(\d+)', lambda m: m.group(1) + str(int(m.group(2)) * span), c) for c in cards]
    head = '      - type: grid\n' + (f'        column_span: {span}\n' if span > 1 else '')
    return head + '        cards:\n' + ''.join(cards)


def view(title, path, icon, sections, columns=4):
    return (f'  - title: {title}\n    path: {path}\n    icon: {icon}\n    type: sections\n    max_columns: {columns}\n'
            f'    theme: ENGINEatHome\n    sections:\n' + ''.join(sections))


def heading(text, icon, style='title'):
    return f'          - type: heading\n            heading: {text}\n            heading_style: {style}\n            icon: {icon}\n'


def _place(cfg, opts):
    lines = (cfg.rstrip('\n') + '\n' + opts).splitlines()
    return '          - ' + lines[0] + '\n' + ''.join('            ' + line + '\n' for line in lines[1:])


def tile_cfg(entity, name=None, icon=None, color=None, content=None, tap=None):
    text = f'type: tile\nentity: {entity}\n'
    if entity.startswith('script.'):text += 'hide_state: true\n'
    if name:text += f'name: {name}\n'
    if icon:text += f'icon: {icon}\n'
    if color:text += f'color: {color}\n'
    if content:text += 'state_content:\n' + ''.join(f'  - {c}\n' for c in content)
    if tap:text += tap
    return text


def tile(entity, name=None, icon=None, color=None, columns=6, content=None, tap=None):
    return _place(tile_cfg(entity, name, icon, color, content, tap), f'grid_options:\n  columns: {columns}\n')


def conditional_all(conditions, inner, columns=12):
    """conditions: [(entity, 'state'|'state_not'|'above', value)] — all must hold."""
    text = ''
    for entity, op, value in conditions:
        if op == 'above':
            text += f'  - condition: numeric_state\n    entity: {entity}\n    above: {value}\n'
        else:
            shown = value if isinstance(value, str) else '[' + ', '.join(value) + ']'
            text += f'  - condition: state\n    entity: {entity}\n    {op}: {shown}\n'
    cfg = 'type: conditional\nconditions:\n' + text + 'card:\n' + indent(inner.rstrip('\n') + '\n', '  ')
    return _place(cfg, f'grid_options:\n  columns: {columns}\n')


def conditional(entity, state, inner, columns=12, negate=False, above=None):
    if above is not None:return conditional_all([(entity, 'above', above)], inner, columns)
    return conditional_all([(entity, 'state_not' if negate else 'state', state)], inner, columns)


def md(content, columns=12):
    return _place('type: markdown\ncontent: |\n' + block(content, 2) + '\n', f'grid_options:\n  columns: {columns}\n')


def entities_card(rows, columns=12):
    body = ''.join(f'  - entity: {e}\n    name: {n}\n' for e, n in rows)
    return _place('type: entities\nshow_header_toggle: false\nentities:\n' + body, f'grid_options:\n  columns: {columns}\n')


def history(entity, hours=24):
    return _place(f'type: history-graph\nhours_to_show: {hours}\nentities:\n  - {entity}\n', 'grid_options:\n  columns: 12\n')


def script_tile(script, name, icon, color, confirm=None):
    tap = f'tap_action:\n  action: perform-action\n  perform_action: script.{script}\n'
    if confirm:tap += f'  confirmation:\n    text: "{confirm}"\n'
    return tile_cfg('script.' + script, name, icon, color, tap=tap)


def traffic(entity, name, icon, rules, content=('state',), columns=12):
    """One tile per state group, shown conditionally, so colour tracks health without custom cards."""
    return ''.join(conditional(entity, states, tile_cfg(entity, name, icon, color, list(content)), columns) for states, color in rules)


def dashboard(address, catalog, default_key):
    names = {p['key']: p['display_name'] for p in catalog}
    out = 'title: ENGINEatHome\nviews:\n'
    health = 'sensor.engineathome_gateway_health'
    state = 'sensor.engineathome_runtime_state'
    raw = 'sensor.engineathome_model_state'
    gpu = 'sensor.engineathome_gpu_status'
    active = 'sensor.engineathome_active_model'
    protection = 'sensor.engineathome_lifecycle_protection'
    selection = 'sensor.engineathome_selection_state'
    moving = ['loading', 'unloading', 'switching']

    # ================================================================ Overview
    overview = [section([traffic(health, 'Gateway', 'mdi:lan-connect', [('Healthy', 'green'), ('Degraded', 'amber'), ('Unreachable', 'red')])]),
                section([tile(active, 'Active model', 'mdi:cube-outline', 'blue', 12, ['state', 'model_id'])]),
                section([
                    conditional('sensor.engineathome_active_requests', None, tile_cfg('sensor.engineathome_active_requests', 'Active requests', 'mdi:swap-horizontal', 'blue'), above=0),
                    conditional('sensor.engineathome_active_requests', '"0"', tile_cfg('sensor.engineathome_active_requests', 'Active requests', 'mdi:swap-horizontal', 'grey')),
                    conditional('sensor.engineathome_active_requests', ['unavailable', 'unknown'], tile_cfg('sensor.engineathome_active_requests', 'Active requests', 'mdi:swap-horizontal', 'red'))]),
                section([traffic(gpu, 'GPU', 'mdi:expansion-card', [('Normal', 'green'), (['Warm', 'Memory tight'], 'amber'), (['Hot', 'Unavailable'], 'red')])])]

    control = [heading('Model control', 'mdi:tune-variant'),
               # What is running (blue) and what is selected (below) are separate controls and separate facts.
               tile(active, 'Running model', 'mdi:cube-outline', 'blue', 6, ['state', 'model_id']),
               ''.join(conditional(raw, s, tile_cfg(state, 'Runtime', icon, color, ['state', 'last_changed']), 6) for s, icon, color in
                       [('ready', 'mdi:check-circle', 'green'), (moving, 'mdi:progress-clock', 'amber'), ('unloaded', 'mdi:power-sleep', 'grey'),
                        ('failed', 'mdi:alert-circle', 'red'), (['unavailable', 'unknown'], 'mdi:lan-disconnect', 'red')]),
               conditional(raw, 'failed', tile_cfg('sensor.engineathome_last_error', 'Load failed', 'mdi:alert', 'red', ['state', 'last_changed'])),
               conditional('sensor.engineathome_switch_outcome', 'failed', tile_cfg('sensor.engineathome_switch_detail', 'Last model switch', 'mdi:alert-circle-outline', 'red', ['state'])),
               heading('Switch model', 'mdi:swap-horizontal-bold', 'subtitle'),
               entities_card([('input_select.engineathome_target_profile', 'Target model')]),
               md(SELECTED),
               conditional(raw, 'switching', tile_cfg('sensor.engineathome_switch_phase', 'Switching model…', 'mdi:progress-wrench', 'amber', ['state'])),
               conditional_all([(selection, 'state', 'Pending'), (raw, 'state_not', moving), (protection, 'state', 'Clear')],
                               script_tile('engineathome_apply_profile', 'Apply model', 'mdi:check-bold', 'blue',
                                           'Apply the selected model? The running model is stopped and its prefix cache is discarded. Refused while requests or session leases are active.')),
               heading('Runtime', 'mdi:power-settings', 'subtitle'),
               conditional(raw, ['unloaded', 'failed'], script_tile('engineathome_load', 'Load model', 'mdi:play', 'blue')),
               conditional(raw, 'ready', script_tile('engineathome_restart', 'Restart', 'mdi:restart', 'blue',
                           'Restart the model? This discards the prefix cache, so the next request starts cold. Refused while requests or session leases are active.'), 6),
               conditional(raw, 'ready', script_tile('engineathome_unload', 'Unload', 'mdi:eject', 'blue',
                           'Unload the model and free the GPU? This discards the prefix cache. Refused while requests or session leases are active.'), 6),
               conditional(protection, 'Clear', tile_cfg(protection, 'Unload, restart and switches blocked', 'mdi:shield-lock', 'amber'), negate=True),
               md('Apply, Unload and Restart discard the prefix cache — the next request re-reads its full prompt. They never interrupt running requests or bypass session leases.')]
    overview.append(section(control, 2))
    overview.append(section([heading('Thinking effort', 'mdi:head-cog-outline'),
                             entities_card([('input_select.engineathome_reasoning_effort', 'Default thinking effort')]),
                             md(THINKING),
                             heading('Connect', 'mdi:connection'),
                             md(CONNECT_SHORT)], 2))

    kpis = [heading('Key metrics', 'mdi:view-dashboard-outline'),
            tile('sensor.engineathome_gpu_vram_used', 'VRAM used', 'mdi:memory', None, 3),
            tile('sensor.engineathome_gpu_vram_free', 'VRAM free', 'mdi:memory', None, 3),
            tile('sensor.engineathome_gpu_utilization', 'GPU utilization', 'mdi:gauge', None, 3),
            tile('sensor.engineathome_gpu_power', 'GPU power', 'mdi:flash', None, 3),
            tile('sensor.engineathome_gpu_temperature', 'GPU temperature', 'mdi:thermometer', None, 3),
            tile('sensor.engineathome_active_requests', 'Active requests', 'mdi:swap-horizontal', None, 3),
            tile('sensor.engineathome_last_prefill', 'Last prefill', 'mdi:speedometer-medium', None, 3),
            tile('sensor.engineathome_last_decode', 'Last decode', 'mdi:speedometer', None, 3),
            tile('sensor.engineathome_last_ttft', 'Last time to first token', 'mdi:timer-play-outline', None, 3),
            tile('sensor.engineathome_cache_hit_rate_today', 'Cache efficiency today', 'mdi:cached', None, 3),
            tile('sensor.engineathome_total_today', 'Tokens today', 'mdi:sigma', None, 3),
            tile('sensor.engineathome_last_request_latency', 'Last request latency', 'mdi:timer-outline', None, 3),
            tile('sensor.engineathome_session_leases', 'Session leases', 'mdi:link-lock', None, 3),
            entities_card([('input_select.engineathome_time_range', 'Time range')], 3),
            tile('sensor.engineathome_range_tokens', 'Tokens in selected range', 'mdi:sigma-lower', None, 3)]
    overview.append(section(kpis, 4))
    overview.append(section([heading('Token usage — last 24 hours', 'mdi:chart-bar-stacked'),
                             md(token_chart('sensor.engineathome_usage_trend', "'Total'", "'hour'", 'No requests in the last 24 hours.'))], 2))
    overview.append(section([heading('Throughput — completed requests, 24 hours', 'mdi:speedometer'), md(THROUGHPUT)], 2))
    for title, entity, icon in (('GPU temperature', 'sensor.engineathome_gpu_temperature', 'mdi:thermometer'),
                                ('GPU power', 'sensor.engineathome_gpu_power', 'mdi:flash'),
                                ('GPU VRAM used', 'sensor.engineathome_gpu_vram_used', 'mdi:memory'),
                                ('GPU utilization', 'sensor.engineathome_gpu_utilization', 'mdi:gauge')):
        overview.append(section([heading(title, icon), history(entity)]))
    overview.append(section([heading('Recent requests', 'mdi:history'), md(RECENT)], 4))
    out += view('Overview', 'overview', 'mdi:view-dashboard', overview)

    # ================================================================ Requests
    requests = [section([
        heading('Filters', 'mdi:filter-variant'),
        entities_card([('input_select.engineathome_time_range', 'Time range'), ('input_select.engineathome_client', 'Client'),
                       ('input_select.engineathome_model', 'Model / profile (active or historical)'), ('input_select.engineathome_status', 'Status'),
                       ('input_select.engineathome_token_view', 'Token category (chart)'), ('input_select.engineathome_group_by', 'Chart buckets')]),
        conditional('input_select.engineathome_time_range', 'Custom',
                    'type: entities\ntitle: Custom range — end after start, at most 367 days\nshow_header_toggle: false\nentities:\n'
                    '  - input_datetime.engineathome_from\n  - input_datetime.engineathome_to\n'),
        _place(tile_cfg('script.engineathome_refresh_usage', 'Refresh now', 'mdi:refresh', 'blue',
                        tap='tap_action:\n  action: perform-action\n  perform_action: script.engineathome_refresh_usage\n'), 'grid_options:\n  columns: 12\n'),
        md(SUMMARY)]),
        section([heading('Tokens in the selected range', 'mdi:chart-bar-stacked'),
                 md(token_chart('sensor.engineathome_usage_history', "states('input_select.engineathome_token_view')",
                                "states('input_select.engineathome_group_by')", 'No requests match these filters.'))], 3),
        section([heading('Requests', 'mdi:format-list-bulleted'), md(REQUESTS)], 4),
        section([heading('By period, client and model', 'mdi:table'), md(PERIODS)], 4)]
    out += view('Requests', 'requests', 'mdi:format-list-bulleted', requests)

    # ================================================================ Settings
    settings = [
        section([heading('Launch profile', 'mdi:star-outline'),
                 entities_card([('input_select.engineathome_default_profile', 'Default launch profile')]),
                 md('The profile that is applied after a clean gateway start and that an on-demand load starts. Changing it never switches the running model; use **Apply model** on the Overview.'),
                 tile('sensor.engineathome_default_profile_name', 'Default', 'mdi:star', None, 6),
                 tile('sensor.engineathome_applied_profile_name', 'Served now', 'mdi:cube-outline', None, 6),
                 tile('sensor.engineathome_selected_profile_name', 'Selected target', 'mdi:cursor-default-click', None, 6),
                 conditional('sensor.engineathome_active_profile_key', 'none', tile_cfg(active, 'Loaded', 'mdi:memory', None, ['state', 'model_id']), 6, negate=True),
                 heading('Automatic unload', 'mdi:timer-sand'),
                 tile('switch.engineathome_auto_unload', 'Unload when idle', 'mdi:power-sleep', 'blue', 12),
                 entities_card([('input_select.engineathome_idle_timeout', 'Idle unload timeout')]),
                 tile('sensor.engineathome_idle_unload_minutes', 'Timeout in effect', 'mdi:timer-outline', None, 12),
                 md('The gateway unloads the model after the selected idle time. Active requests and session leases always postpone it. Changing a setting never interrupts a request.')]),
        section([heading('Models and API', 'mdi:api'),
                 tile('sensor.engineathome_context_capacity', 'Context of the served model', 'mdi:text-long', None, 12),
                 md(MODELS_TABLE), md(CONNECT)]),
        section([heading('Thinking controls', 'mdi:head-cog-outline'), md(REASONING_TABLE),
                 heading('Runtime configuration', 'mdi:cog-outline'), md(LAUNCH)]),
        section([heading('Diagnostics', 'mdi:stethoscope'),
                 tile(health, 'Gateway health', 'mdi:lan-connect', None, 12),
                 tile(state, 'Runtime', 'mdi:power', None, 6),
                 tile('sensor.engineathome_session_leases', 'Session leases', 'mdi:link-lock', None, 6),
                 tile('sensor.engineathome_gateway_uptime', 'Gateway uptime', 'mdi:clock-outline', None, 6),
                 tile('sensor.engineathome_model_uptime', 'Model loaded for', 'mdi:clock-check-outline', None, 6),
                 tile('sensor.engineathome_model_load_seconds', 'Last load took', 'mdi:download', None, 6),
                 tile('sensor.engineathome_gpu_sample_time', 'GPU sampled', 'mdi:expansion-card', None, 6),
                 tile('sensor.engineathome_last_request_time', 'Last request', 'mdi:clock-check-outline', None, 6),
                 tile('sensor.engineathome_switch_phase', 'Switch phase', 'mdi:swap-horizontal-bold', None, 6),
                 tile('sensor.engineathome_switch_detail', 'Last model switch', 'mdi:history', None, 12, ['state']),
                 tile('sensor.engineathome_last_error', 'Last lifecycle error', 'mdi:alert-circle-outline', None, 12, ['state', 'last_changed'])]),
        section([heading('Lifecycle protections', 'mdi:shield-lock'), md(PROTECTIONS)], 3)]
    out += view('Settings', 'settings', 'mdi:cog', settings, 3)
    return out
