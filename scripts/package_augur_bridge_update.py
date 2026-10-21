"""Package an unchanged native release with a separately verified Python overlay.

No native receipt is rewritten. All copies are explicit manifest-selected files;
private local backups, model weights, session data and run logs are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import augur_bridge_entry as E


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_copy(source, destination, digest):
    if source.is_symlink() or sha(source) != digest:
        raise ValueError('input identity drift: ' + source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha(source) != digest or sha(destination) != digest:
        raise ValueError('input changed during copy: ' + source.name)


def package(base, overlay, out, version, source_commit):
    base, overlay, out = (Path(p).resolve() for p in (base, overlay, out))
    if out.exists() or any(out.is_relative_to(p) or p.is_relative_to(out) for p in (base, overlay)):
        raise ValueError('output must be a new disjoint directory')
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-' for c in version):
        raise ValueError('invalid version')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('full source commit SHA required')
    original = json.loads((base / 'windows-distribution.json').read_bytes())
    deployment = json.loads((overlay / 'deployment-manifest.json').read_bytes())
    if deployment.get('schema') != 'augur-python-bridge-overlay-v1':
        raise ValueError('verified overlay manifest required')
    out.mkdir(parents=True)
    for name, digest in original['files'].items():
        if not E.safe_name(name) or not (base / name).resolve().is_relative_to(base):
            raise ValueError('unsafe original package member')
        checked_copy(base / name, out / 'native-base' / name, digest)
    checked_copy(base / 'windows-distribution.json', out / 'native-base/windows-distribution.json', sha(base / 'windows-distribution.json'))
    overlay_names = set()
    for original_name, digest in deployment['overlay_files_sha256'].items():
        name = original_name.replace('\\', '/')
        if name in overlay_names:
            raise ValueError('duplicate normalized overlay member: ' + name)
        overlay_names.add(name)
        if not E.safe_name(name) or not name.startswith(('source/scripts/', 'source/model-cards/')):
            raise ValueError('non-public overlay member: ' + name)
        if not (overlay / name).resolve().is_relative_to(overlay):
            raise ValueError('overlay path escape')
        checked_copy(overlay / name, out / 'bridge' / name, digest)
    entry = Path(__file__).with_name('augur_bridge_entry.py')
    checked_copy(entry, out / 'bridge-entry.py', sha(entry))
    commands = {'Start-Augur.cmd': ('configure', '--autostart'),
                'Start-Augur-Engine.cmd': ('engine', ''),
                'Start-Monitor.cmd': ('monitor', ''),
                'Verify-Package.cmd': ('verify', '')}
    for filename, (action, flags) in commands.items():
        command = ('@echo off\r\nsetlocal\r\n"%~dp0native-base\\runtime\\python\\python.exe" '
            '-I -X utf8 -B "%~dp0bridge-entry.py" ' + action + ' ' + flags + ' %*\r\nexit /b %errorlevel%\r\n')
        (out / filename).write_bytes(command.encode('ascii'))
    (out / 'README-START-HERE.txt').write_text(
        'Augur ' + version + ' — Windows SM70 bridge update\n\n'
        '1. Extract the complete archive. Run Verify-Package.cmd.\n'
        '2. Stop an older Augur service normally before switching; never overlap GPU owners.\n'
        '3. Double-click Start-Augur.cmd. The local setup page opens and uses saved settings.\n'
        '4. Keep the launcher window open; use Quit Augur to stop cleanly.\n\n'
        '中文：完整解压后先运行 Verify-Package.cmd，正常退出旧服务，再双击 Start-Augur.cmd。\n'
        '请勿直接运行 native-base 内的旧启动入口；它不包含桥接修复。\n\n'
        'The original native package/receipts remain byte-for-byte under native-base/.\n'
        'The tested Python patch lives under bridge/. This is NOT a native rebuild.\n'
        'Nerkyor research code is in the source release, NOT these shipped binaries.\n'
        'Model weights and NVIDIA drivers are separate. Settings, keys, caches and sessions are not included.\n'
        'Set DSH requiresAssistantAfterToolResult=false only AFTER the new backend advertises support.\n'
        'Restore that flag to true before rolling back to the old backend.\n'
        'An existing conversation may need one complete prefill after migration.\n'
        'This package is not code signed or independently clean-machine qualified.\n',encoding='utf8')
    record = dict(schema=E.SCHEMA, version=version, source_release_commit=source_commit,
        base_distribution_sha256=sha(base / 'windows-distribution.json'),
        base_build_receipt_sha256=sha(base / 'build-receipt.json'),
        native_rebuilt=False, native_receipts_unchanged=True, research_native_changes_shipped=False,
        model_weights_included=False, private_data_included=False, clean_machine_verified=False,
        files={p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    (out / 'bridge-update.json').write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    E.verify(out)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','overlay','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--version',required=True);p.add_argument('--source-commit',required=True)
    a=p.parse_args();record=package(a.base,a.overlay,a.out,a.version,a.source_commit)
    print(json.dumps(dict(status='PACKAGED_VERIFIED',files=len(record['files']),native_rebuilt=False)))


if __name__=='__main__': main()
