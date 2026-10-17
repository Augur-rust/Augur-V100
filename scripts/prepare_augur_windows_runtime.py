"""Build a hash-locked Windows runtime from authenticated upstream inputs, no models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import urllib.request
import zipfile
import prepare_augur_release as R
import prepare_augur_notices as N
from prepare_augur_source_archive import safe_member, source_notices, release_entry

# Exact upstream redistributables selected for the Windows SM70 build.
ARCHIVES = {
    'python-3.13.16-embed-amd64.zip': ('97dae5274cc54867065e8d5a3226e48c35017ed332a0fdb0e27d5b5821961297',
        'https://www.python.org/ftp/python/3.13.16/python-3.13.16-embed-amd64.zip'),
    'cuda_cudart-windows-x86_64-12.8.90-archive.zip': ('4a39058fd8519444a81cfc7ae055d136f48d1a31ffa41ae255b35b2edd61e13b',
        'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip'),
    'libcublas-windows-x86_64-12.8.4.1-archive.zip': ('57a470112cec7e112c95253dde8b3c7184d795dbd92b0bde77a4cb7f8c94c8aa',
        'https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.8.4.1-archive.zip'),
}
CUDA_DLLS = {'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll'}
VC_DLLS = {'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll',
           'msvcp140_2.dll', 'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll', 'concrt140.dll'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=True, indent=2)
        f.write('\n')


def members(archive):
    """Reject Windows aliases, escapes and links before extracting any member."""
    seen = set()
    for info in archive.infolist():
        name = info.filename.rstrip('/') if info.is_dir() else info.filename
        if (not safe_member(name) or name.casefold() in seen
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError('unsafe/duplicate ZIP member: ' + name)
        seen.add(name.casefold())
        if not info.is_dir():
            yield info


def extract(archive, info, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, path.open('xb') as dest:
        shutil.copyfileobj(source, dest, 1024 * 1024)
    if path.stat().st_size != info.file_size:
        raise ValueError('archive member size drift')


def verify_pypi(filename, identity, raw, size):
    metadata = json.loads(raw)
    def normalize(v): return v.lower().replace('_', '-').replace('.', '-')
    if (normalize(metadata['info']['name']) != normalize(identity['name'])
            or metadata['info']['version'] != identity['version']):
        raise ValueError('PyPI package/version mismatch')
    rows = [v for v in metadata['urls'] if v['filename'] == filename and v['packagetype'] == 'bdist_wheel']
    if (len(rows) != 1 or rows[0]['digests']['sha256'] != identity['sha256']
            or rows[0]['size'] != size or rows[0].get('yanked') is not False
            or not rows[0]['url'].startswith('https://files.pythonhosted.org/')):
        raise ValueError('wheel not authenticated by the exact PyPI release')
    return rows[0]['url']


def validate(root):
    root = Path(root).resolve()
    record = json.loads((root/'runtime-manifest.json').read_bytes())
    if record.get('schema') != 'quasar-windows-runtime-v1':
        raise ValueError('invalid runtime manifest')
    actual = set()
    for p in root.rglob('*'):
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError('runtime symlink/escape')
        if p.is_file(): actual.add(p.relative_to(root).as_posix())
    if actual != set(record['files']) | {'runtime-manifest.json'}:
        raise ValueError('unexpected/missing runtime files')
    for name, row in record['files'].items():
        if (not safe_member(name) or R.sha(root/name) != row['sha256']
                or (root/name).stat().st_size != row['bytes']):
            raise ValueError('runtime file drift: ' + name)
    required = {'python/python.exe', 'python/python313._pth', 'requirements-windows.lock'}
    required |= {'native/'+n for n in CUDA_DLLS | VC_DLLS}
    if not required <= actual: raise ValueError('incomplete runtime')
    return record


def prepare(inputs, vc_redist, out, download_metadata=False):
    inputs, vc_redist, out = inputs.resolve(), vc_redist.resolve(), out.resolve()
    if out.is_relative_to(inputs) or inputs.is_relative_to(out):
        raise ValueError('runtime output and inputs must be disjoint')
    if sys.version_info[:2] != (3, 13) or os.name != 'nt':
        raise ValueError('build this cp313 Windows runtime with Python 3.13 on Windows')
    for name, (digest, _) in ARCHIVES.items():
        if R.sha(inputs/'downloads'/name) != digest: raise ValueError('upstream archive hash drift: '+name)
    out.mkdir(parents=True, exist_ok=False)
    notices, notice_files, archives = out/'notices', {}, []
    found = set()
    for name, (digest, url) in ARCHIVES.items():
        with zipfile.ZipFile(inputs/'downloads'/name) as archive:
            infos = list(members(archive))
            for info in infos:
                base = Path(info.filename).name
                if name.startswith('python-'):
                    extract(archive, info, out/'python'/info.filename)
                elif base in CUDA_DLLS:
                    extract(archive, info, out/'native'/base); found.add(base)
                if N.notice_name(info.filename):
                    N.retain(notices, 'upstream/'+name+'/'+info.filename, archive.read(info), notice_files)
        archives.append(dict(filename=name, sha256=digest, url=url))
    if found != CUDA_DLLS: raise ValueError('CUDA DLL set incomplete')
    # Only installed vendor REDIST inputs are eligible, never System32 DLLs.
    if vc_redist.name != 'Microsoft.VC143.CRT' or 'Redist' not in vc_redist.parts:
        raise ValueError('use the installed Visual Studio x64 CRT REDIST directory')
    vc_rows = []
    for name in sorted(VC_DLLS):
        source = vc_redist/name
        literal = str(source).replace("'", "''")
        command = ("$p=Get-Item -LiteralPath '"+literal+"';$s=Get-AuthenticodeSignature -LiteralPath $p.FullName;"
                   "[pscustomobject]@{status=$s.Status.ToString();subject=$s.SignerCertificate.Subject;"
                   "version=$p.VersionInfo.FileVersion}|ConvertTo-Json -Compress")
        row = json.loads(subprocess.check_output([shutil.which('pwsh.exe') or 'powershell.exe','-NoProfile','-Command',command], text=True))
        if row['status'] != 'Valid' or 'Microsoft Corporation' not in row['subject']:
            raise ValueError('MSVC runtime signer validation failed')
        shutil.copyfile(source, out/'native'/name)
        if R.sha(source) != R.sha(out/'native'/name): raise ValueError('VC copy drift')
        row.update(filename=name, sha256=R.sha(source), origin='VisualStudio2022/VC/Redist/MSVC/'+vc_redist.parent.parent.name+'/x64/Microsoft.VC143.CRT')
        vc_rows.append(row)
        # CPython ships older CRT DLLs next to python.exe; use one vendor version.
        if name.startswith('vcruntime'):
            shutil.copyfile(source, out/'python'/name)
    metadata_root = inputs/'pypi'; metadata_root.mkdir(exist_ok=True)
    lock, wheels = R.wheel_lock(inputs/'wheels')
    for name, identity in wheels.items():
        metadata_file = metadata_root/(identity['name']+'-'+identity['version']+'.json')
        if not metadata_file.exists():
            if not download_metadata: raise ValueError('missing PyPI metadata: '+str(metadata_file))
            url='https://pypi.org/pypi/'+identity['name']+'/'+identity['version']+'/json'
            with urllib.request.urlopen(url, timeout=60) as response: raw=response.read()
            metadata_file.write_bytes(raw)
        raw = metadata_file.read_bytes()
        identity['url'] = verify_pypi(name, identity, raw, (inputs/'wheels'/name).stat().st_size)
        identity['pypi_metadata_sha256'] = hashlib.sha256(raw).hexdigest()
        (out/'provenance').mkdir(exist_ok=True)
        shutil.copyfile(metadata_file, out/'provenance'/metadata_file.name)
    (out/'requirements-windows.lock').write_text(lock, encoding='ascii')
    with (inputs/('pip-install-'+out.name+'.log')).open('xb') as log:
        subprocess.run([sys.executable, '-m','pip','--isolated','install','--no-index','--no-deps',
            '--no-compile','--require-hashes','--only-binary=:all:', '--find-links',str(inputs/'wheels'),
            '--target',str(out/'python/Lib/site-packages'),'-r',str(out/'requirements-windows.lock')],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    # sys.path is isolated from registry, PYTHONPATH, cwd and user site.
    (out/'python/python313._pth').write_text('python313.zip\n.\nLib/site-packages\n../../source/scripts\n',encoding='ascii')
    wheel_rows=N.wheel_notices(inputs/'wheels',wheels,notices,notice_files)
    for wheel in wheel_rows:
        if wheel['embedded_notices_present']: continue
        raw=(metadata_root/(wheel['name']+'-'+wheel['version']+'.json')).read_bytes()
        metadata=json.loads(raw)
        sources=[v for v in metadata['urls'] if v['packagetype']=='sdist']
        if len(sources)!=1: raise ValueError('one exact source release required for missing wheel license')
        row=release_entry(metadata,sources[0]['filename'],'sdist')
        source=inputs/'downloads'/row['filename']
        if not source.exists():
            if not download_metadata: raise ValueError('missing license source archive: '+row['filename'])
            with urllib.request.urlopen(row['url'],timeout=60) as response: data=response.read()
            source.write_bytes(data)
        recovered,proof=source_notices(wheel,wheel['filename'],raw,
            hashlib.sha256(raw).hexdigest(),source.read_bytes(),source.name)
        wheel['supplemental_source_proof']=proof
        wheel['notice_files'] += [N.retain(notices,'python-sdist/'+name,data,notice_files)
                                  for name,data in recovered.items()]

    for filename in ('vc-redistribution.html','cuda-eula.html'):
        source=inputs/'upstream'/filename
        N.retain(notices,'vendor/'+filename,source.read_bytes(),notice_files)
    (out/'README.txt').write_text('Private application runtime for Augur, not a standalone CUDA SDK.\n'
        'CPython, Python wheels, NVIDIA and Microsoft files retain their respective licenses.\n'
        'The NVIDIA driver, Windows/UCRT, GPU hardware and model weights are not included.\n'
        'The embedded interpreter has an explicit isolated sys.path; no pip is shipped.\n'
        'MSVC vcruntime DLLs beside Python are replaced by the recorded signed REDIST version.\n',encoding='utf8')
    record=dict(schema='quasar-windows-runtime-v1',archives=archives,python_wheels=wheel_rows,
        msvc=vc_rows,files={p.relative_to(out).as_posix():dict(sha256=R.sha(p),bytes=p.stat().st_size)
        for p in sorted(out.rglob('*')) if p.is_file()},
        platform='win_amd64',python='3.13.16',cuda='12.8.1',clean_machine_verified=False,
        driver_included=False,weights_included=False,publicly_published=False)
    save(out/'runtime-manifest.json',record)
    validate(out)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs', type=Path)
    ap.add_argument('--vc-redist', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--download-metadata', action='store_true')
    ap.add_argument('--verify-only', action='store_true')
    a=ap.parse_args()
    if not a.verify_only and (a.inputs is None or a.vc_redist is None): ap.error('inputs and vc-redist required')
    result=validate(a.out) if a.verify_only else prepare(a.inputs,a.vc_redist,a.out,a.download_metadata)
    print(json.dumps(dict(files=len(result['files']),python=result['python'],clean_machine_verified=False)))

if __name__=='__main__': main()
