"""Install only the ENGINEatHome package, preserve configuration, validate before restart."""
import datetime as dt
from pathlib import Path
import shutil
import subprocess


def main():
    source=Path(__file__).resolve().parent/'ha-generated'
    config=Path('/srv/containers/homeassistant/config')
    path=config/'configuration.yaml';original=path.read_text();changed=original
    marker='    engine-at-home:\n'
    if marker not in original:
        anchor='  dashboards:\n'
        if original.count(anchor)!=1:raise RuntimeError('inspect Lovelace dashboards before installing')
        entry='''    engine-at-home:
      mode: yaml
      title: ENGINEatHome
      icon: mdi:brain
      show_in_sidebar: true
      filename: engineathome/dashboard.yaml
'''
        changed=changed.replace(anchor,anchor+entry,1)
    if '    engineathome: !include engineathome/package.yaml' not in original:
        if '\nhomeassistant:' in '\n'+original:raise RuntimeError('existing homeassistant block needs a targeted merge')
        changed+='\nhomeassistant:\n  packages:\n    engineathome: !include engineathome/package.yaml\n'
    target=config/'engineathome';target.mkdir(exist_ok=True);target.chmod(0o750)
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup=target/('backup-'+stamp);backup.mkdir(mode=0o700)
    shutil.copy2(path,backup/'configuration.yaml')
    for name in ('package.yaml','dashboard.yaml','secrets.yaml'):
        if (target/name).exists():shutil.copy2(target/name,backup/name)
        shutil.copyfile(source/name,target/name);(target/name).chmod(0o600 if name=='secrets.yaml' else 0o644)
    # The theme lives in HA's existing themes folder (configuration.yaml already includes it); only our own file is touched.
    themes=config/'themes';created_themes=not themes.exists();themes.mkdir(exist_ok=True)
    theme=themes/'engineathome.yaml';had_theme=theme.exists()
    if had_theme:shutil.copy2(theme,backup/'theme-engineathome.yaml')
    shutil.copyfile(source/'themes/engineathome.yaml',theme);theme.chmod(0o644)
    path.write_text(changed)
    check=subprocess.run(['docker','exec','homeassistant','python','-m','homeassistant','--script','check_config','--config','/config'],capture_output=True,text=True)
    # Do not copy Home Assistant diagnostics into usage logs; they can contain unrelated secrets.
    (backup/'check-config.log').write_text(check.stdout+check.stderr);(backup/'check-config.log').chmod(0o600)
    if check.returncode:
        path.write_text(original)
        for name in ('package.yaml','dashboard.yaml','secrets.yaml'):
            if (backup/name).exists():shutil.copy2(backup/name,target/name)
        if had_theme:shutil.copy2(backup/'theme-engineathome.yaml',theme)
        else:
            theme.unlink()
            if created_themes:themes.rmdir()
        raise RuntimeError('HA config validation failed; previous configuration restored; inspect protected check-config.log')
    print('Home Assistant configuration validated; backup: '+str(backup))


if __name__=='__main__':main()
