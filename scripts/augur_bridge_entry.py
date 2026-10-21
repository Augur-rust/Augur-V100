"""Verified entry point for a side-by-side Augur Windows bridge update.

The original native package and receipts stay under native-base/. Only the
separately identified Python bridge executes from bridge/source/scripts/.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import sys

SCHEMA = 'augur-windows-bridge-update-v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    return (isinstance(name, str) and bool(name) and '\\' not in name and ':' not in name
            and not PurePosixPath(name).is_absolute()
            and all(part not in ('', '.', '..') for part in name.split('/')))


def verify(root):
    root = Path(root).resolve(strict=True)
    record = json.loads((root / 'bridge-update.json').read_bytes())
    if record.get('schema') != SCHEMA or not isinstance(record.get('files'), dict):
        raise ValueError('invalid bridge update manifest')
    expected = set(record['files']) | {'bridge-update.json'}
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('bundle link or path escape')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if expected != actual:
        raise ValueError('unexpected or missing bridge update files')
    for name, digest in record['files'].items():
        if not safe_name(name) or sha(root / name) != digest:
            raise ValueError('bridge update file drift: ' + name)
    base = root / 'native-base'
    original = json.loads((base / 'windows-distribution.json').read_bytes())
    if sha(base / 'windows-distribution.json') != record['base_distribution_sha256']:
        raise ValueError('original distribution identity changed')
    for name, digest in original['files'].items():
        if not safe_name(name) or record['files'].get('native-base/' + name) != digest:
            raise ValueError('original package changed: ' + name)
    if sha(base / 'build-receipt.json') != record['base_build_receipt_sha256']:
        raise ValueError('original native build receipt changed')
    return record


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ('configure', 'engine', 'monitor', 'stop', 'verify'):
        raise SystemExit('usage: bridge-entry.py configure|engine|monitor|stop|verify [options]')
    root = Path(__file__).resolve().parent
    record = verify(root)
    base = root / 'native-base'
    scripts = root / 'bridge/source/scripts'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts))
    entry = scripts / 'augur_portable.py'
    if args[0] == 'verify':
        # Retains original runtime, build, CPU and source verification. No GPU.
        sys.argv = [str(entry), 'verify', '--runtime-root', str(base)]
        runpy.run_path(str(entry), run_name='__main__')
        import qwen35_chat as C
        if C.MAX_MESSAGES != 65536:
            raise ValueError('wrong bridge loaded')
        print(json.dumps(dict(status='PASS_BRIDGE_UPDATE_INTEGRITY_AND_IMPORTS',
            version=record['version'], native_rebuilt=False, gpu_started=False,
            files=len(record['files']), max_messages=C.MAX_MESSAGES)))
        return
    if '--runtime-root' in args:
        raise ValueError('runtime root is pinned to this verified bundle')
    sys.argv = [str(entry), args[0], '--runtime-root', str(base), *args[1:]]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
