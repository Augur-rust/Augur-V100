"""Retain authenticated dependency notices; this is not legal or image clearance."""
import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tarfile
import tomllib
import zipfile
import prepare_augur_release as R

MAX_NOTICE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def notice_name(name):
    return PurePosixPath(name).name.upper().startswith(
        ('LICENSE', 'LICENCE', 'COPYING', 'COPYRIGHT', 'NOTICE'))


def package_directory(name, version):
    if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+!-]{0,127}', part or '')
               for part in (name, version)):
        raise ValueError('unsafe dependency name/version')
    return name + '-' + version


def retain(out, name, data, files):
    if not R.safe_name(name) or name.casefold() in {key.casefold() for key in files}:
        raise ValueError('unsafe or duplicate notice path')
    if not data or len(data) > MAX_NOTICE_BYTES:
        raise ValueError('empty or oversized notice')
    if sum(row['bytes'] for row in files.values()) + len(data) > MAX_BUNDLE_BYTES:
        raise ValueError('notice bundle size limit')
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)
    files[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return name


def cargo_notices(metadata, lock, cache, out, files):
    locked = {(p['name'], p['version'], p.get('source')): p.get('checksum')
              for p in lock['package']}
    rows, seen = [], set()
    for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        if package['source'] is None:
            continue
        key = (package['name'], package['version'], package['source'])
        digest = locked.get(key)
        if key in seen or not package['source'].startswith('registry+') or not re.fullmatch(r'[a-f0-9]{64}', digest or ''):
            raise ValueError('dependency is not a unique checksum-locked registry package')
        seen.add(key)
        directory = package_directory(*key[:2])
        candidates = sorted(cache.glob('*/' + directory + '.crate'))
        archive_path = next((p for p in candidates if p.is_file() and R.sha(p) == digest), None)
        if archive_path is None:
            raise ValueError('authenticated Cargo archive absent: ' + directory)
        notices, member_names = [], set()
        with tarfile.open(archive_path, mode='r:gz') as archive:
            for member in archive:
                name = member.name.rstrip('/') if member.isdir() else member.name
                parts = PurePosixPath(name).parts
                if not R.safe_name(name) or parts[0] != directory or name in member_names:
                    raise ValueError('unsafe/duplicate Cargo archive member')
                member_names.add(name)
                if not member.isfile() and not member.isdir():
                    raise ValueError('Cargo archive link or special member rejected')
                if not member.isfile() or not notice_name(name):
                    continue
                if member.size <= 0 or member.size > MAX_NOTICE_BYTES or len(parts) < 2:
                    raise ValueError('invalid Cargo notice size/path')
                with archive.extractfile(member) as stream:
                    data = stream.read(MAX_NOTICE_BYTES + 1)
                if len(data) != member.size:
                    raise ValueError('Cargo notice truncated')
                notices.append(retain(out, 'cargo/' + name, data, files))
        if not notices or R.sha(archive_path) != digest:
            raise ValueError('Cargo notice absent or archive changed: ' + directory)
        rows.append(dict(name=key[0], version=key[1], source=key[2],
            locked_checksum=digest, license_expression=package.get('license'), notice_files=notices))
    return rows


def wheel_notices(root, inventory, out, files):
    rows = []
    for filename, identity in sorted(inventory.items()):
        if not R.safe_name(filename) or '/' in filename:
            raise ValueError('unsafe wheel filename')
        path = root / filename
        if R.sha(path) != identity['sha256']:
            raise ValueError('wheel changed before notice collection')
        directory = package_directory(identity['name'], identity['version'])
        notices, names = [], set()
        with zipfile.ZipFile(path) as archive:
            metadata = [p for p in archive.infolist() if p.filename.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('one wheel metadata record required')
            value = BytesParser().parsebytes(archive.read(metadata[0]))
            if (value['Name'], value['Version']) != (identity['name'], identity['version']):
                raise ValueError('wheel metadata identity mismatch')
            for member in archive.infolist():
                name = member.filename.rstrip('/') if member.is_dir() else member.filename
                if not R.safe_name(name) or name in names:
                    raise ValueError('unsafe/duplicate wheel member')
                names.add(name)
                if member.is_dir() or not (notice_name(name) or '.dist-info/licenses/' in name):
                    continue
                if stat.S_ISLNK(member.external_attr >> 16) or member.file_size <= 0 or member.file_size > MAX_NOTICE_BYTES:
                    raise ValueError('wheel notice link/size rejected')
                notices.append(retain(out, 'python/' + directory + '/' + name, archive.read(member), files))
        if R.sha(path) != identity['sha256']:
            raise ValueError('wheel changed while collecting notices')
        license_field = value.get('License', '')
        rows.append(dict(identity, filename=filename, license_expression=value.get('License-Expression'),
            license_field=license_field if len(license_field) <= 160 else None,
            license_classifiers=[c for c in value.get_all('Classifier', []) if c.startswith('License ::')],
            declared_license_files=value.get_all('License-File', []), notice_files=notices,
            embedded_notices_present=bool(notices)))
    return rows


def prepare(context, cache, out):
    preparation = R.validate_context(context)
    out.mkdir(exist_ok=False)
    try:
        metadata_bytes = subprocess.check_output(['cargo', 'metadata', '--locked', '--offline',
            '--features', 'augur-model-qwen35-runtime/qwen35-augur-vision',
            '--filter-platform', 'x86_64-unknown-linux-gnu', '--format-version', '1'], cwd=context)
        files = {}
        cargo = cargo_notices(json.loads(metadata_bytes),
            tomllib.loads((context / 'Cargo.lock').read_text(encoding='utf8')), cache, out, files)
        wheels = wheel_notices(context / 'runtime-wheels', preparation['wheels'], out, files)
        native = [retain(out, 'native/' + name, (context / name).read_bytes(), files)
                  for name in R.NATIVE_NOTICES]
        missing = [dict(name=row['name'], version=row['version'], filename=row['filename'])
                   for row in wheels if not row['embedded_notices_present']]
        readme = ('# Augur Dependency Notices\n\n'
            'Cargo notices come from archives authenticated against Cargo.lock.\n'
            'Python notices come from the exact locked wheel bytes. Native project,\n'
            'llama/ggml, NInfer and Lucide notices are retained from the source context.\n'
            'No missing wheel notice is silently replaced with a Cargo package license.\n'
            'The inventory is resolved workspace scope, not proof of linked binary scope.\n'
            'This bundle is not legal clearance. Cargo/Wheel declarations, native binary\n'
            'transitive dependencies, CUDA/base OS conditions and model/draft rights\n'
            'still require review. No model, credentials or absolute host paths are included.\n'
            '\nMissing embedded wheel notices:\n' +
            ''.join('- ' + row['name'] + '==' + row['version'] + '\n' for row in missing))
        retain(out, 'README.md', readme.encode('ascii'), files)
        result = dict(schema='quasar-notice-bundle-v1',
            source_manifest_sha256=preparation['source_manifest_sha256'],
            preparation_sha256=R.sha(context / 'release-preparation.json'),
            collector_sha256=R.sha(Path(__file__)), cargo=cargo, python_wheels=wheels,
            native_notice_files=native, files=files, missing_python_embedded_notices=missing,
            license_review_complete=False, base_os_cuda_inventory_complete=False,
            model_rights_verified=False, linux_build_verified=False, product_ready=False)
        R.validate_context(context)
        with (out / 'notice-bundle.json').open('x', encoding='utf8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        validate_bundle(out, context)
        return result
    except BaseException as error:
        with (out / 'failure.json').open('x', encoding='utf8') as stream:
            json.dump(dict(type=type(error).__name__, error=str(error)), stream, indent=2)
        raise


def validate_bundle(root, context):
    preparation = R.validate_context(context)
    result = json.loads((root / 'notice-bundle.json').read_bytes())
    if (result['schema'] != 'quasar-notice-bundle-v1'
            or result['source_manifest_sha256'] != preparation['source_manifest_sha256']
            or result['preparation_sha256'] != R.sha(context / 'release-preparation.json')):
        raise ValueError('unbound notice context')
    for name, identity in result['files'].items():
        if not R.safe_name(name) or R.sha(root / name) != identity['sha256'] or (root / name).stat().st_size != identity['bytes']:
            raise ValueError('notice path/byte drift')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('notice symlink/escape')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(result['files']) | {'notice-bundle.json'}:
        raise ValueError('unexpected/missing notice files')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--cargo-cache', type=Path,
        default=Path(os.environ.get('CARGO_HOME', Path.home() / '.cargo')) / 'registry/cache')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    context, out = args.context.resolve(), args.out.resolve()
    result = validate_bundle(out, context) if args.verify_only else prepare(context, args.cargo_cache.resolve(), out)
    print(json.dumps(dict(notice_bundle=str(out), files=len(result['files']),
        cargo_packages=len(result['cargo']), python_wheels=len(result['python_wheels']),
        missing_python_embedded_notices=result['missing_python_embedded_notices'],
        bundle_sha256=R.sha(out / 'notice-bundle.json'), license_review_complete=False)))


if __name__ == '__main__':
    main()
