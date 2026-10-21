"""Isolated entry point for the bundled Windows runtime (no system Python/Conda)."""
import argparse
import os
from pathlib import Path
import runpy
import sys

_DLL_HANDLES=[]

def activate(root):
    root=Path(root).resolve()
    native=root/'runtime/native'
    if not native.is_dir(): raise RuntimeError('bundled native runtime missing; verify or re-extract the package')
    for name in list(os.environ):
        if name.upper().startswith(('PYTHON','CUDA_PATH')): os.environ.pop(name)
    system=Path(os.environ.get('SystemRoot',r'C:\Windows'))
    os.environ['PATH']=os.pathsep.join(map(str,[native,root/'bin',system/'System32',system]))
    os.environ['PYTHONNOUSERSITE']='1'
    sys.dont_write_bytecode=True
    if os.name=='nt': _DLL_HANDLES.append(os.add_dll_directory(str(native)))
    return root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('action',choices=('engine','monitor','stop','verify','configure'))
    parser.add_argument('--runtime-root', type=Path, help='explicit unchanged native/runtime package for a separate bridge')
    options,remaining=parser.parse_known_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    root=activate(options.runtime_root or Path(__file__).resolve().parents[2])
    if options.action=='verify':
        import json
        from prepare_augur_windows_runtime import validate
        import augur_product as P
        runtime=validate(root/'runtime'); build=P.verified_build(root)
        record=json.loads((root/'windows-distribution.json').read_bytes())
        for name,digest in record['files'].items():
            from prepare_augur_source_archive import safe_member
            if not safe_member(name) or P.sha(root/name)!=digest:
                raise ValueError('package file drift: '+name)
        import tokenizers, jinja2, PIL, jsonschema
        print(json.dumps(dict(status='PASS_PACKAGE_INTEGRITY_AND_IMPORTS',
            runtime_files=len(runtime['files']),binaries=len(build['binaries']),
            python=sys.version.split()[0],python_isolated=bool(sys.flags.isolated),
            gpu_started=False,clean_machine_verified=False)))
        return
    script={'engine':'run_augur_windows.py','monitor':'augur_monitor.py','configure':'augur_launcher.py',
            'stop':'augur_stop.py'}[options.action]
    sys.argv=[str(Path(__file__).with_name(script)),*(['--build',str(root)] if options.action in ('engine','configure') else []),*remaining]
    runpy.run_path(sys.argv[0],run_name='__main__')

if __name__=='__main__': main()
