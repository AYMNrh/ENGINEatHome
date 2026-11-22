"""Metadata-only SQLite usage ledger. Cached/reasoning tokens are subsets, never added twice."""
import datetime as dt
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from zoneinfo import ZoneInfo


FIELDS=('input_tokens','output_tokens','total_tokens','cached_tokens','fresh_input_tokens','reasoning_tokens',
        'prefill_tps','decode_tps','ttft_s','upstream_ttft_s','latency_s','upstream_s','usage_reported')


def bucket(timestamp,unit,timezone):
    value=dt.datetime.fromtimestamp(timestamp,dt.timezone.utc).astimezone(ZoneInfo(timezone))
    if unit=='hour':value=value.replace(minute=0,second=0,microsecond=0)
    elif unit=='day':value=value.replace(hour=0,minute=0,second=0,microsecond=0)
    elif unit=='week':value=(value-dt.timedelta(days=value.weekday())).replace(hour=0,minute=0,second=0,microsecond=0)
    elif unit=='month':value=value.replace(day=1,hour=0,minute=0,second=0,microsecond=0)
    else:raise ValueError('invalid bucket')
    return value.isoformat()


class Store:
    def __init__(self,path,legacy_profiles=None):
        """legacy_profiles: {model_id: profile_key}; rows written before profiles existed are attributed through it."""
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript('''CREATE TABLE IF NOT EXISTS requests(
                id TEXT PRIMARY KEY,client TEXT NOT NULL,model TEXT NOT NULL,started REAL NOT NULL,finished REAL,
                status TEXT NOT NULL,http_status INTEGER,error TEXT,stream INTEGER,
                input_tokens INTEGER,output_tokens INTEGER,total_tokens INTEGER,cached_tokens INTEGER,
                fresh_input_tokens INTEGER,reasoning_tokens INTEGER,prefill_tps REAL,decode_tps REAL,
                ttft_s REAL,upstream_ttft_s REAL,latency_s REAL,upstream_s REAL,usage_reported INTEGER DEFAULT 0);
                CREATE INDEX IF NOT EXISTS request_time ON requests(started);
                CREATE INDEX IF NOT EXISTS request_client_time ON requests(client,model,started);
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS leases(id TEXT PRIMARY KEY,client TEXT NOT NULL,expires REAL NOT NULL);''')
            columns={row[1] for row in db.execute('PRAGMA table_info(requests)')}
            migrating='profile' not in columns
            for column in ('profile','effort'):
                if column not in columns:db.execute(f'ALTER TABLE requests ADD COLUMN {column} TEXT')
            if migrating:   # once: later rows without a profile are rejected requests and must never be credited to a model
                for model,profile in (legacy_profiles or {}).items():
                    db.execute('UPDATE requests SET profile=? WHERE profile IS NULL AND model=?',(profile,model))
            db.execute("UPDATE requests SET status='interrupted',error='gateway_restart',finished=? WHERE status='running'",(time.time(),))

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=15);db.row_factory=sqlite3.Row
        db.create_function('period',3,bucket)
        try:
            with db:yield db
        finally:db.close()

    def start(self,id,client,model,stream):
        with self.connect() as db:db.execute('INSERT INTO requests(id,client,model,started,status,stream) VALUES(?,?,?,?,?,?)',
                                             (id,client,model,time.time(),'running',int(stream)))

    def tag(self,id,model,profile,effort):
        """Attribute a request to the model that actually served it; effort is what was forwarded to the template."""
        with self.connect() as db:db.execute('UPDATE requests SET model=?,profile=?,effort=? WHERE id=?',(model,profile,effort,id))

    def finish(self,id,status,http_status,error,values):
        names=','.join(name+'=?' for name in FIELDS)
        with self.connect() as db:db.execute(f'UPDATE requests SET finished=?,status=?,http_status=?,error=?,{names} WHERE id=?',
            (time.time(),status,http_status,error,*[values.get(name) for name in FIELDS],id))

    def setting(self,key,default=None):
        with self.connect() as db:row=db.execute('SELECT value FROM settings WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self,key,value):
        with self.connect() as db:db.execute('INSERT INTO settings VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',(key,json.dumps(value)))

    def leases(self):
        with self.connect() as db:
            db.execute('DELETE FROM leases WHERE expires<=?',(time.time(),))
            return [dict(row) for row in db.execute('SELECT id,client,expires FROM leases ORDER BY expires')]

    def lease(self,id,client,expires):
        with self.connect() as db:db.execute('INSERT INTO leases VALUES(?,?,?) ON CONFLICT(id) DO UPDATE SET expires=excluded.expires',(id,client,expires))

    def release(self,id):
        with self.connect() as db:db.execute('DELETE FROM leases WHERE id=?',(id,))

    def query(self,start,end,unit,timezone,client=None,model=None,detail=False,profile=None):
        if unit not in ('hour','day','week','month'):raise ValueError('invalid bucket')
        ZoneInfo(timezone)
        where='started>=? AND started<?';args=[start,end]
        for name,value in (('client',client),('model',model),('profile',profile)):
            if value:where+=' AND '+name+'=?';args.append(value)
        with self.connect() as db:
            if detail:
                rows=db.execute('SELECT * FROM requests WHERE '+where+' ORDER BY started DESC LIMIT 500',args)
            else:
                rows=db.execute('''SELECT period(started,?,?) AS period,client,model,profile,COUNT(*) AS requests,
                  SUM(status='complete') AS completed,SUM(status!='complete') AS incomplete_or_failed,
                  SUM(usage_reported) AS usage_reported_requests,
                  SUM(input_tokens) AS input_tokens,SUM(output_tokens) AS output_tokens,SUM(total_tokens) AS total_tokens,
                  SUM(cached_tokens) AS cached_tokens,SUM(fresh_input_tokens) AS fresh_input_tokens,
                  COUNT(input_tokens) AS input_known_requests,COUNT(cached_tokens) AS cache_known_requests,
                  COUNT(output_tokens) AS output_known_requests,COUNT(fresh_input_tokens) AS fresh_known_requests,
                  AVG(CASE WHEN status='complete' THEN latency_s END) AS mean_latency_s,
                  AVG(CASE WHEN status='complete' THEN ttft_s END) AS mean_ttft_s
                  FROM requests WHERE '''+where+' GROUP BY period,client,model,profile ORDER BY period,client,model,profile',
                  [unit,timezone,*args])
            return [dict(row) for row in rows]

    def totals(self,timezone):
        now=time.time();result={}
        for unit in ('hour','day','week','month'):
            start=dt.datetime.fromisoformat(bucket(now,unit,timezone)).timestamp()
            rows=self.query(start,now+1,unit,timezone)
            result[unit]={key:sum(row.get(key) or 0 for row in rows) for key in
                          ('requests','completed','incomplete_or_failed','usage_reported_requests','input_known_requests','output_known_requests','cache_known_requests','fresh_known_requests','input_tokens','output_tokens','cached_tokens','fresh_input_tokens','total_tokens')}
        return result

    def models(self):
        # Only models that actually served requests; names from rejected requests are client input, not models.
        with self.connect() as db:return [row[0] for row in db.execute('SELECT DISTINCT model FROM requests WHERE profile IS NOT NULL ORDER BY model')]

    def clients(self):
        with self.connect() as db:return [row[0] for row in db.execute('SELECT DISTINCT client FROM requests ORDER BY client')]

    def latest(self):
        with self.connect() as db:
            row=db.execute("SELECT * FROM requests WHERE status!='running' ORDER BY started DESC LIMIT 1").fetchone()
            return dict(row) if row else None


class Counters:
    def __init__(self):self.values={};self.terminal=False;self.done=False;self.first_token=False

    def observe(self,event):
        usage=event.get('usage') or {}
        timings=event.get('timings') or {}
        def take(key,value):
            if isinstance(value,(int,float)) and not isinstance(value,bool) and value>=0:self.values[key]=value
        take('input_tokens',usage.get('prompt_tokens',usage.get('input_tokens')))
        take('output_tokens',usage.get('completion_tokens',usage.get('output_tokens')))
        take('cached_tokens',(usage.get('prompt_tokens_details') or usage.get('input_tokens_details') or {}).get('cached_tokens'))
        take('reasoning_tokens',(usage.get('completion_tokens_details') or usage.get('output_tokens_details') or {}).get('reasoning_tokens'))
        if 'cache_n' in timings:
            take('cached_tokens',timings['cache_n'])
            if 'prompt_n' in timings:take('input_tokens',timings['cache_n']+timings['prompt_n'])
        take('output_tokens',timings.get('predicted_n',event.get('tokens_predicted')))
        take('prefill_tps',timings.get('prompt_per_second'));take('decode_tps',timings.get('predicted_per_second'))
        if usage or timings:self.values['usage_reported']=1
        for choice in event.get('choices') or []:
            delta=choice.get('delta') or choice.get('message') or {}
            if choice.get('finish_reason'):self.terminal=True
            if any(delta.get(k) for k in ('content','reasoning_content','tool_calls')) or choice.get('text'):self.first_token=True
        if event.get('content') or event.get('tokens'):self.first_token=True
        if event.get('stop') is True:self.terminal=True;self.done=True

    def finish(self):
        data=dict(self.values)
        prompt,output,cached=(data.get(k) for k in ('input_tokens','output_tokens','cached_tokens'))
        if prompt is not None and output is not None:data['total_tokens']=prompt+output
        if prompt is not None and cached is not None and cached<=prompt:data['fresh_input_tokens']=prompt-cached
        data.setdefault('usage_reported',0)
        return data
