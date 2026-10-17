"""Export the selected source without weights/binaries; Linux support is deferred."""
import argparse
from email.parser import BytesParser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import urllib.parse
import zipfile
import prepare_augur_release as R
import prepare_augur_notices as N

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 256 * 1024 * 1024
MAX_MEMBERS = 10000
MANIFEST = 'source-distribution.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + '\n').encode('ascii')


def safe_member(name):
    if not R.safe_name(name):
        return False
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {p + str(n) for p in ('COM', 'LPT') for n in range(1, 10)}
    return all(part == part.rstrip(' .') and part.split('.')[0].upper() not in reserved
               and not any(ord(c) < 32 or c in '<>"|?*' for c in part) for part in name.split('/'))


def add(files, name, data):
    names = {key.casefold() for key in files}
    folded = name.casefold()
    if (not safe_member(name) or folded in names
            or any(folded.startswith(key + '/') or key.startswith(folded + '/') for key in names)):
        raise ValueError('unsafe, duplicate or aliased distribution member')
    if not data or len(data) > MAX_FILE or len(files) >= MAX_MEMBERS:
        raise ValueError('distribution file/member bound')
    if sum(map(len, files.values())) + len(data) > MAX_TOTAL:
        raise ValueError('distribution total bound')
    files[name] = data


def read_checked(root, name, expected):
    path = root / name
    if not safe_member(name) or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('input path escape')
    if path.stat().st_size > MAX_FILE:
        raise ValueError('input file bound')
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError('input byte drift: ' + name)
    return raw


def release_entry(release, filename, kind):
    matches = [r for r in release['urls'] if r['filename'] == filename and r['packagetype'] == kind]
    if len(matches) != 1:
        raise ValueError('one exact PyPI release file required')
    row = matches[0]
    url = urllib.parse.urlsplit(row['url'])
    if (url.scheme != 'https' or url.netloc != 'files.pythonhosted.org' or url.query or url.fragment
            or PurePosixPath(url.path).name != filename or row.get('yanked') is not False
            or not re.fullmatch('[a-f0-9]{64}', row['digests']['sha256'])
            or type(row['size']) is not int or not 0 < row['size'] <= MAX_FILE):
        raise ValueError('invalid PyPI distribution identity')
    return row


def source_notices(wheel, wheel_filename, metadata_raw, expected_metadata_sha, archive_raw, archive_name):
    """Bind a source notice to the exact wheel's PyPI release, never a Rust crate."""
    if len(metadata_raw) > MAX_FILE or digest(metadata_raw) != expected_metadata_sha:
        raise ValueError('PyPI metadata identity mismatch')
    release = strict_json(metadata_raw)
    if (release['info']['name'], release['info']['version']) != (wheel['name'], wheel['version']):
        raise ValueError('PyPI package identity mismatch')
    wheel_row = release_entry(release, wheel_filename, 'bdist_wheel')
    if wheel_row['digests']['sha256'] != wheel['sha256']:
        raise ValueError('PyPI wheel does not match the locked wheel')
    sdist = release_entry(release, archive_name, 'sdist')
    if len(archive_raw) != sdist['size'] or digest(archive_raw) != sdist['digests']['sha256']:
        raise ValueError('sdist size/hash mismatch')
    directory = N.package_directory(wheel['name'], wheel['version'])
    members, total, found = set(), 0, {}
    with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as archive:
        for member in archive:
            name = member.name.rstrip('/') if member.isdir() else member.name
            if (not safe_member(name) or PurePosixPath(name).parts[0] != directory
                    or name.casefold() in members or len(members) >= MAX_MEMBERS):
                raise ValueError('unsafe/duplicate sdist member')
            members.add(name.casefold())
            if not member.isfile() and not member.isdir():
                raise ValueError('sdist link or special member rejected')
            total += member.size
            if member.size < 0 or member.size > MAX_FILE or total > MAX_TOTAL:
                raise ValueError('sdist member/total bound')
            if member.isfile() and (name == directory + '/PKG-INFO' or N.notice_name(name)):
                with archive.extractfile(member) as stream:
                    data = stream.read(MAX_FILE + 1)
                if len(data) != member.size or not data:
                    raise ValueError('empty or truncated sdist notice/metadata')
                found[name] = data
    metadata = BytesParser().parsebytes(found.pop(directory + '/PKG-INFO', b''))
    if (metadata['Name'], metadata['Version']) != (wheel['name'], wheel['version']) or not found:
        raise ValueError('Python sdist identity/notice absent')
    proof = dict(name=wheel['name'], version=wheel['version'], wheel_filename=wheel_filename,
        wheel_sha256=wheel['sha256'], metadata_sha256=expected_metadata_sha,
        pypi_url='https://pypi.org/pypi/' + wheel['name'] + '/' + wheel['version'] + '/json',
        sdist_filename=archive_name, sdist_sha256=sdist['digests']['sha256'], sdist_url=sdist['url'],
        notice_sha256={name: digest(data) for name, data in found.items()},
        notice_origin='same Python release sdist; not embedded in wheel',
        wheel_rebuilt_or_equivalent_verified=False, source_executed=False, legal_clearance=False)
    return found, proof


def write_archive(path, files, facts):
    if MANIFEST in files:
        raise ValueError('manifest is reserved')
    manifest = dict(facts, schema='quasar-source-distribution-v1',
        files={name: dict(sha256=digest(data), bytes=len(data)) for name, data in sorted(files.items())})
    checked = {}
    for name, data in files.items():
        add(checked, name, data)
    add(checked, MANIFEST, encoded(manifest))
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(checked.items()):
            member = zipfile.ZipInfo(name, (2026, 10, 10, 0, 0, 0))
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o644) << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(member, data, compresslevel=9)
    return manifest


def verify_archive(path, expected_sha, destination=None):
    if R.sha(path) != expected_sha:
        raise ValueError('archive identity mismatch')
    files = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if (member.is_dir() or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)
                    or not 0 < member.file_size <= MAX_FILE
                    or sum(map(len, files.values())) + member.file_size > MAX_TOTAL):
                raise ValueError('archive type/size rejected')
            add(files, member.filename, archive.read(member))
    manifest = strict_json(files.pop(MANIFEST))
    if manifest.get('schema') != 'quasar-source-distribution-v1' or set(files) != set(manifest['files']):
        raise ValueError('archive manifest/extent mismatch')
    for name, raw in files.items():
        if manifest['files'][name] != dict(sha256=digest(raw), bytes=len(raw)):
            raise ValueError('archive member identity mismatch')
    source_raw = files['augur-release-source.json']
    if digest(source_raw) != manifest['source_manifest_sha256']:
        raise ValueError('archive source identity mismatch')
    source = strict_json(source_raw)
    if source.get('schema') != 'quasar-release-source-v1' or source.get('default_main_model') != 'quasar-27b':
        raise ValueError('QUASAR source manifest required')
    for name, expected in source['files'].items():
        if not R.selected(name) or digest(files[name]) != expected:
            raise ValueError('archive frozen source drift')
    if R.sha(path) != expected_sha:
        raise ValueError('archive changed during verification')
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
        # Validate every entry before creating anything; never use extractall.
        for name, data in dict(files, **{MANIFEST: encoded(manifest)}).items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('extraction path escape')
            with target.open('xb') as stream:
                stream.write(data)
    return manifest


def prepare(context, notices, out, metadata, metadata_sha, sdist, readme):
    preparation = R.validate_context(context)
    bundle = N.validate_bundle(notices, context)
    source = strict_json((context / 'augur-release-source.json').read_bytes())
    files = {}
    for name, expected in source['files'].items():
        add(files, name, read_checked(context, name, expected))
    add(files, 'augur-release-source.json', read_checked(context, 'augur-release-source.json', preparation['source_manifest_sha256']))
    for name, identity in bundle['files'].items():
        add(files, 'distribution/notices/' + name, read_checked(notices, name, identity['sha256']))
    add(files, 'distribution/notices/notice-bundle.json', (notices / 'notice-bundle.json').read_bytes())
    missing = bundle['missing_python_embedded_notices']
    if len(missing) != 1 or (missing[0]['name'], missing[0]['version']) != ('tokenizers', '0.22.2'):
        raise ValueError('review changed missing-notice scope before exporting')
    filename = missing[0]['filename']
    meta_raw, archive_raw = metadata.read_bytes(), sdist.read_bytes()
    supplement, proof = source_notices(preparation['wheels'][filename], filename, meta_raw, metadata_sha, archive_raw, sdist.name)
    for name, raw in supplement.items():
        add(files, 'distribution/source-notices/' + name, raw)
    add(files, 'distribution/source-notices/provenance.json', encoded(proof))
    add(files, 'distribution/source-notices/pypi-release.json', meta_raw)
    add(files, 'distribution/source-notices/' + sdist.name, archive_raw)
    add(files, 'README.source.md', readme.read_bytes())
    add(files, 'distribution/exporter.py', Path(__file__).read_bytes())
    facts = dict(source_manifest_sha256=preparation['source_manifest_sha256'],
        context_preparation_sha256=R.sha(context / 'release-preparation.json'),
        notice_bundle_sha256=R.sha(notices / 'notice-bundle.json'),
        exporter_sha256=R.sha(Path(__file__)), default_main_model='quasar-27b',
        source_files=len(source['files']), model_weights_included=False, draft_weights_included=False,
        native_binaries_included=False, runtime_wheels_included=False, credentials_included=False,
        linux_support='deferred_by_user', registry_published=False,
        missing_embedded_wheel_notices=missing, supplemented_from_same_python_release=['tokenizers==0.22.2'],
        legal_clearance=False, product_ready=False, numerical_qualified=False, dflash_active=False)
    R.validate_context(context)
    N.validate_bundle(notices, context)
    write_archive(out, files, facts)
    return verify_archive(out, R.sha(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', required=True, type=Path)
    parser.add_argument('--notices', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--pypi-json', required=True, type=Path)
    parser.add_argument('--pypi-json-sha256', required=True)
    parser.add_argument('--sdist', required=True, type=Path)
    parser.add_argument('--readme', required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.context.resolve(), args.notices.resolve(), args.out.resolve(),
        args.pypi_json.resolve(), args.pypi_json_sha256, args.sdist.resolve(), args.readme.resolve())
    print(json.dumps(dict(archive_sha256=R.sha(args.out), files=len(result['files']),
        source_files=result['source_files'], source_preparation_pass=True, linux_support='deferred_by_user')))


if __name__ == '__main__':
    main()
