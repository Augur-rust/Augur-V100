"""Export the exact Windows release source and notices without Git, models or runtime binaries."""
import argparse
import json
from pathlib import Path
import augur_product as P
import prepare_augur_windows_notices as N
import prepare_augur_source_archive as A
import prepare_augur_release as R


def export(build,notices,out):
    build=Path(build).resolve();notices=Path(notices).resolve();out=Path(out).resolve()
    if out.is_relative_to(build) or out.is_relative_to(notices): raise ValueError('archive must be outside inputs')
    P.verified_build(build);notice=N.validate(notices)
    identity=json.loads((build/'source-identity.json').read_bytes())
    for name,digest in (notice['source_lock_hashes'] | notice['source_notice_hashes']).items():
        if identity['files'].get(name)!=digest: raise ValueError('notice/source lock mismatch')
    files={}
    for name,digest in identity['files'].items():
        if not R.selected(name): raise ValueError('non-public source file: '+name)
        A.add(files,name,A.read_checked(build/'source',name,digest))
    # This is the manifest consumed by the no-Git independent builder.
    A.add(files,'augur-release-source.json',A.encoded(dict(schema='quasar-release-source-v1',
        git_head=identity['git_head'],files=identity['files'],default_main_model='quasar-27b')))
    for name,row in notice['files'].items(): A.add(files,'THIRD_PARTY/'+name,A.read_checked(notices,name,row['sha256']))
    A.add(files,'THIRD_PARTY/notice-bundle.json',(notices/'notice-bundle.json').read_bytes())
    result = A.write_archive(out,files,dict(source_files=len(identity['files']),
        source_manifest_sha256=A.digest(files['augur-release-source.json']),
        bound_build_receipt_sha256=P.sha(build/'build-receipt.json'),
        source_identity_sha256=P.sha(build/'source-identity.json'),
        platform='Windows x64 / SM70',weights_included=False,binaries_included=False,
        default_main_model='quasar-27b',default_vision_residency='on_demand',published=False,
        clean_machine_verified=False))
    A.verify_archive(out,P.sha(out))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('build','notices','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=export(a.build,a.notices,a.out)
    print(json.dumps(dict(files=len(result['files']),sha256=P.sha(a.out),bytes=a.out.stat().st_size)))

if __name__=='__main__':main()
