"""Generate a private serving configuration from validated stock artifacts; never print keys."""
import hashlib
import json
import os
from pathlib import Path
import secrets


def write(path,data):
    path.write_text(json.dumps(data,indent=2)+'\n');path.chmod(0o600)


def main():
    os.umask(0o077);root=Path(__file__).resolve().parent;experiments=Path.home()/'llm/experiments'
    state=json.loads((experiments/'state.json').read_text())
    measured=json.loads(Path(state['jobs']['stock-262144-r1']['result']).read_text())
    baseline=json.loads((experiments/'builds/stock/manifest.json').read_text())
    binary=Path.home()/'llm/llama.cpp/build/bin/llama-server'
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=baseline['sha256']:raise RuntimeError('original stock binary differs')
    for name,digest in baseline['libraries'].items():
        if hashlib.sha256((binary.parent/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('original stock library differs')
    model=Path(json.loads((experiments/'config.json').read_text())['model'])
    sha=hashlib.sha256()
    with model.open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):sha.update(block)
    expected='9aecf1cd41b2cb2f32a74e0d889e33855ebef43b26f43b43feb5720239e677e5'
    if sha.hexdigest()!=expected:raise RuntimeError('model checksum mismatch')
    alias='swift-1.5-qwen3.8-27b';port=18081;argv=list(measured['argv']);argv[0]=str(binary)
    for flag,value in (('--port',str(port)),('--alias',alias),('--verbosity','0')):argv[argv.index(flag)+1]=value
    argv.append('--log-disable')
    credentials_path=root/'credentials.json'
    credentials=json.loads(credentials_path.read_text()) if credentials_path.exists() else {}
    scopes={'opencode':['inference'],'hermes':['inference'],'python':['inference'],
            'admin':['control','analytics'],'home-assistant':['control','analytics'],'analytics':['analytics']}
    for name in scopes:credentials.setdefault(name,'eah_'+secrets.token_urlsafe(32))
    write(credentials_path,credentials)
    address='http://homelab.tail5bda2d.ts.net:8080'
    stat=model.stat();buffers=measured['placement']['buffers']
    config=dict(version=1,port=8080,backend_port=port,backend_url=f'http://127.0.0.1:{port}',
        tailscale_ip='100.75.206.4',api_base_url=address+'/v1',model_id=alias,model=str(model),context=262144,
        argv=argv,database='/var/lib/engineathome/usage.sqlite3',idle_seconds=3600,request_timeout=14400,
        max_active_requests=8,timezone='Europe/Paris',load_timeout=240,
        minimum_load_free_bytes=measured['startup']['device']['used']+512*2**20,
        minimum_process_device_bytes=sum(buffers['CUDA0_'+k+'_bytes'] for k in ('model','KV','RS')),
        model_stat=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns),checkpoint_sha256=expected,
        sampling=dict(temperature=1.0,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,repeat_penalty=1.,seed=42),
        keys={hashlib.sha256(credentials[name].encode()).hexdigest():dict(client=name,scopes=scope) for name,scope in scopes.items()})
    if (root/'config.json').exists():
        previous=json.loads((root/'config.json').read_text())
        if previous['model_stat']!=config['model_stat']:raise RuntimeError('existing serving model changed')
    write(root/'config.json',config)
    write(root/'runtime-manifest.json',dict(checkpoint_sha256=expected,binary_sha256=baseline['sha256'],
          source_revision=baseline['revision'],original_argv=measured['argv'],production_argv=argv,
          changes='original binary path, fixed localhost port, stable alias, private logging; inference settings unchanged'))
    # Keep the validated Inkling profiles in the regenerated configuration (installs the pinned runtime, verifies hashes).
    import add_profiles
    try:config=add_profiles.merge(root/'config.json')
    except (FileNotFoundError,RuntimeError) as exc:
        print('warning: Inkling profiles not added ('+type(exc).__name__+': '+str(exc)[:120]+'); Swift-only configuration kept')
        config=json.loads((root/'config.json').read_text())
    from ha_files import generate
    generate(root/'ha-generated',address,credentials['home-assistant'],config)
    print('Configured validated stock serving. Credentials retained in a mode-0600 file; no keys printed.')


if __name__=='__main__':main()
