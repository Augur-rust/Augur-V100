"""Collect Windows dependency notices from both Cargo locks and a verified runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tomllib
import prepare_augur_release as R
import prepare_augur_notices as N
import prepare_augur_windows_runtime as W


def validate(root):
    root=Path(root).resolve();record=json.loads((root/'notice-bundle.json').read_bytes())
    if record.get('schema')!='quasar-windows-notices-v1': raise ValueError('unexpected notice schema')
    actual=set()
    for p in root.rglob('*'):
        if p.is_symlink() or not p.resolve().is_relative_to(root): raise ValueError('notice symlink/escape')
        if p.is_file(): actual.add(p.relative_to(root).as_posix())
    if actual!=set(record['files'])|{'notice-bundle.json'}: raise ValueError('notice file set drift')
    for name,row in record['files'].items():
        if not W.safe_member(name) or R.sha(root/name)!=row['sha256'] or (root/name).stat().st_size!=row['bytes']:
            raise ValueError('notice file hash drift')
    return record


def prepare(source,runtime,out,cache):
    source=source.resolve();runtime=runtime.resolve();out=out.resolve()
    if out.is_relative_to(source) or out.is_relative_to(runtime): raise ValueError('use a separate notice directory')
    rt=W.validate(runtime);out.mkdir(parents=True,exist_ok=False)
    all_packages={};all_locks={};inputs={}
    for project in (source,source/'tools/augur-quantize'):
        command=['cargo','metadata','--locked','--offline','--filter-platform','x86_64-pc-windows-msvc','--format-version','1']
        if project==source: command+=['--features','augur-model-qwen35-runtime/qwen35-augur-vision']
        metadata=json.loads(subprocess.check_output(command,cwd=project))
        for p in metadata['packages']: all_packages[(p['name'],p['version'],p.get('source'))]=p
        lock=tomllib.loads((project/'Cargo.lock').read_text(encoding='utf8'))
        for p in lock['package']:
            key=(p['name'],p['version'],p.get('source'))
            if key in all_locks and all_locks[key].get('checksum')!=p.get('checksum'): raise ValueError('Cargo locks disagree')
            all_locks[key]=p
        inputs[(project/'Cargo.lock').relative_to(source).as_posix()]=R.sha(project/'Cargo.lock')
    files={}
    rows=N.cargo_notices(dict(packages=list(all_packages.values())),dict(package=list(all_locks.values())),cache,out,files)
    for name in R.NATIVE_NOTICES: N.retain(out,'source/'+name,(source/name).read_bytes(),files)
    for name,identity in rt['files'].items():
        if name.startswith('notices/'):
            N.retain(out,'runtime/'+name[len('notices/'):],(runtime/name).read_bytes(),files)
    W.save(out/'dependency-inventory.json',dict(schema='quasar-windows-dependencies-v1',
        scope='resolved Windows Cargo workspace and quantizer, Python closure and shipped native runtime; not exact Rust linked-symbol inventory',
        cargo=rows,python=rt['python_wheels'],native_archives=rt['archives'],msvc=rt['msvc'],
        windows_and_nvidia_driver='provided by the user, not redistributed',models_included=False))
    p=out/'dependency-inventory.json';files[p.name]=dict(sha256=R.sha(p),bytes=p.stat().st_size)
    N.retain(out,'README.md',b'# Augur Windows dependency notices\n\n'
        b'Cargo archives match both committed Cargo.lock files. Python wheels match exact PyPI release hashes.\n'
        b'Tokenizers wheel omits LICENSE; the notice is supplied from its exact PyPI source release, with proof.\n'
        b'Vendor archive hashes, signed MSVC REDIST identities, CUDA EULA and redistribution lists are retained.\n'
        b'Third-party terms remain applicable; the project license does not replace them.\n'
        b'The inventory is resolved dependency scope, not a claim that every Rust crate is linked.\n'
        b'Model weights, Windows and NVIDIA drivers are not bundled. This is not legal advice.\n',files)
    result=dict(schema='quasar-windows-notices-v1',source_lock_hashes=inputs,
        source_notice_hashes={name:R.sha(source/name) for name in R.NATIVE_NOTICES},
        runtime_manifest_sha256=R.sha(runtime/'runtime-manifest.json'),cargo_packages=len(rows),files=files,
        weights_included=False,clean_machine_verified=False)
    W.save(out/'notice-bundle.json',result);validate(out)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cargo-cache',type=Path,default=Path(os.environ.get('CARGO_HOME',Path.home()/'.cargo'))/'registry/cache')
    a=p.parse_args();r=prepare(a.source,a.runtime,a.out,a.cargo_cache)
    print(json.dumps(dict(files=len(r['files']),cargo_packages=r['cargo_packages'])))

if __name__=='__main__': main()
