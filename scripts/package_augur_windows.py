"""Create a relocatable Windows binary candidate without altering build/CPU receipts."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import augur_product as P

SCHEMA='quasar-windows-portable-build-v1'
LAUNCH_EXE='启动服务.exe'

def copy_checked(source, destination, expected):
    if P.sha(source)!=expected: raise ValueError('source changed before package copy')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
    if P.sha(source)!=expected or P.sha(destination)!=expected:
        raise ValueError('source changed during package copy')


def package(build, out, runtime=None, notices=None):
    build=Path(build).resolve();out=Path(out).resolve()
    if out.is_relative_to(build) or build.is_relative_to(out):
        raise ValueError('package and build directories must be disjoint')
    original=P.verified_build(build)
    version=(build/'source/VERSION').read_text(encoding='ascii').strip()
    if not version or any(c.isspace() for c in version): raise ValueError('invalid release version')
    if runtime is not None:
        from prepare_augur_windows_runtime import validate
        runtime=Path(runtime).resolve(); runtime_record=validate(runtime)
        if out.is_relative_to(runtime) or runtime.is_relative_to(out):
            raise ValueError('runtime and package directories must be disjoint')
    if notices is not None:
        from prepare_augur_windows_notices import validate as validate_notices
        notices=Path(notices).resolve(); notice_record=validate_notices(notices)
        if out.is_relative_to(notices) or notices.is_relative_to(out):
            raise ValueError('notices and package directories must be disjoint')
    identity=json.loads((build/'source-identity.json').read_bytes())
    if notices is not None:
        for name,digest in (notice_record['source_lock_hashes'] | notice_record['source_notice_hashes']).items():
            if identity['files'].get(name)!=digest: raise ValueError('notice/source identity mismatch')
        if runtime is not None and notice_record['runtime_manifest_sha256']!=P.sha(runtime/'runtime-manifest.json'):
            raise ValueError('notice/runtime identity mismatch')
    out.mkdir(parents=True,exist_ok=False)
    copy_checked(build/'source/VERSION',out/'VERSION',P.sha(build/'source/VERSION'))
    for name in ('build-receipt.json','cpu-complete.json','source-identity.json'):
        copy_checked(build/name,out/name,P.sha(build/name))
    for name,digest in identity['files'].items():
        copy_checked(build/'source'/name,out/'source'/name,digest)
    binaries={}
    for name,record in original['binaries'].items():
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
            raise ValueError('invalid binary name')
        relative='bin/'+name+'.exe'
        copy_checked(Path(record['path']),out/relative,record['sha256'])
        binaries[name]=dict(path=relative,sha256=record['sha256'])
        if name=='augur-launch' and runtime is not None:
            # Double-click entry at the package root; it resolves runtime/ beside itself.
            # The bin/ record stays the verified identity; this copy is hashed in windows-distribution.json.
            copy_checked(Path(record['path']),out/LAUNCH_EXE,record['sha256'])
    P.save(out/'portable-build.json',dict(schema=SCHEMA,
        build_receipt_sha256=P.sha(out/'build-receipt.json'),
        cpu_receipt_sha256=P.sha(out/'cpu-complete.json'),
        source_identity_sha256=P.sha(out/'source-identity.json'),binaries=binaries))
    for name in ('LICENSE-MIT','LICENSE-APACHE','NOTICE'):
        if (out/'source'/name).is_file():
            copy_checked(out/'source'/name,out/name,P.sha(out/'source'/name))
    if runtime is not None:
        for name,record in runtime_record['files'].items():
            copy_checked(runtime/name,out/'runtime'/name,record['sha256'])
            # Windows searches the executable directory before System32 and PATH.
            # Keep the signed CRT app-local to native EXEs, not just on PATH.
            from prepare_augur_windows_runtime import VC_DLLS
            if name in {'native/'+dll for dll in VC_DLLS}:
                copy_checked(runtime/name,out/'bin'/Path(name).name,record['sha256'])
        copy_checked(runtime/'runtime-manifest.json',out/'runtime/runtime-manifest.json',P.sha(runtime/'runtime-manifest.json'))
    if notices is not None:
        for name,record in notice_record['files'].items():
            copy_checked(notices/name,out/'THIRD_PARTY'/name,record['sha256'])
        copy_checked(notices/'notice-bundle.json',out/'THIRD_PARTY/notice-bundle.json',P.sha(notices/'notice-bundle.json'))
    actions=[('Start-Augur.cmd','augur_launcher.py','configure',' --build "%~dp0."'),
             ('Start-Augur-Engine.cmd','run_augur_windows.py','engine',' --build "%~dp0."'),
             ('Start-Monitor.cmd','augur_monitor.py','monitor','')]
    if runtime is not None: actions.append(('Verify-Package.cmd',None,'verify',''))
    for filename,script,action,arguments in actions:
        text='@echo off\r\nsetlocal\r\n'
        if runtime is None:
            text+='if not defined AUGUR_PYTHON set "AUGUR_PYTHON=python"\r\n'
            text+='"%AUGUR_PYTHON%" -X utf8 -B "%~dp0source\\scripts\\'+script+'"'+arguments+' %*\r\n'
        else:
            text+='if not exist "%~dp0runtime\\python\\python.exe" (echo Missing bundled runtime. Re-extract the full archive. & exit /b 2)\r\n'
            text+='"%~dp0runtime\\python\\python.exe" -I -X utf8 -B "%~dp0source\\scripts\\augur_portable.py" '+action+' %*\r\n'
        text+='exit /b %errorlevel%\r\n'
        (out/filename).write_bytes(text.encode('ascii'))
    launch=('双击“启动服务.exe”：打开本机配置页；已保存过配置时会直接加载模型。\n'
            '再次双击只会重新打开配置页。在页面里点“退出 Augur”即可关闭服务并退出。\n'
            'Double-click 启动服务.exe to open the setup page; a saved configuration starts automatically.\n\n'
            if (out/LAUNCH_EXE).is_file() else '')
    (out/'README.txt').write_text(
        'Augur '+version+' Windows x64 / SM70\n\n'
        + launch
        + ('Includes isolated CPython, locked Python wheels, CUDA/cuBLAS and signed MSVC redistributables.\n'
           'No system Python, Conda or CUDA SDK is needed to launch this package.\n'
           'Run Verify-Package.cmd before first use; this checks bytes/imports, not GPU quality.\n'
           if runtime is not None else
           'Requires Python 3.11+ and dependencies, CUDA 12.8 and MSVC runtime.\n')
        + 'Requires compatible NVIDIA SM70 hardware/driver and Windows x64/UCRT.\n'
        'Start-Augur.cmd opens the same setup page with a console window for diagnostics.\n'
        'See source/docs/AUGUR_WINDOWS_USER_GUIDE.zh-CN.md for setup and persistent-cache disk budgeting.\n'
        'Target/draft model weights, credentials and session data are NOT included.\n'
        'DFlash default: Rust Q8_0 with embedded FP32 gate0.45, fixed B8, runtime scale=1.\n'
        'Decoding default: probability drafts on all 7 B8 rows, sparse p/q verification; shared BF16 head on the FP32 rw4 kernel; draft Q8_0 B8 projections on the direct kernel.\n'
        'Vision default: on_demand. Explicit optional flag: --vision-residency auto.\n'
        'Start-Augur-Engine.cmd --help; see source/docs/AUGUR_WINDOWS_SOURCE_GUIDE.md for model inputs.\n'
        'Start-Monitor.cmd --runs-root RUNS --live-file PRIVATE_LIVE_JSON --engine-port 8294 --port 8295\n'
        'Package launches loopback only. Do not share API keys or browser capability URLs.\n'
        '\nAPI Key / 接入其他 Agent\n'
        '  本发布包不含任何 API Key。首次启动时在本机生成，保存在 %LOCALAPPDATA%\\Augur\\api-key.txt，\n'
        '  升级或换目录后保持不变。默认同时写入当前用户环境变量（无需管理员）：\n'
        '    AUGUR_API_KEY   AUGUR_BASE_URL=http://127.0.0.1:8294/v1   AUGUR_MODEL=quasar-27b\n'
        '  聊天页自动登录，无需输入。其他 Agent：填 Base URL、模型名，API Key 读环境变量 AUGUR_API_KEY，\n'
        '  或在配置页“接入其他 Agent”里复制。新打开的终端/Agent 才能读到环境变量。\n'
        '  事先自己设置了 AUGUR_API_KEY（至少 32 个可见字符）时以你的为准。不会修改 OPENAI_API_KEY。\n'
        '  怀疑泄露：配置页点“更换 API Key”，旧密钥立即失效。不想写环境变量：取消配置页里的勾选。\n'
        '  The package ships no key. One per-user key lives in %LOCALAPPDATA%\\Augur\\api-key.txt and,\n'
        '  by default, in the user variables AUGUR_API_KEY / AUGUR_BASE_URL / AUGUR_MODEL.\n\n'
        'Build/CPU receipts are preserved; portable-build.json resolves binary hashes locally.\n'
        'See THIRD_PARTY and runtime/notices for third-party licensing and inventory.\n'
        'Development-host isolated deployment is not a clean-machine hardware qualification.\n',encoding='utf-8')
    P.verified_build(out)
    files={p.relative_to(out).as_posix():P.sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    result=dict(schema='quasar-windows-binary-distribution-v1',files=files,release_version=version,
        product_ready=False,published=False,weights_included=False,runtime_included=runtime is not None,notices_included=notices is not None,clean_machine_verified=False)
    P.save(out/'windows-distribution.json',result)
    return result


def archive(out, destination):
    out=Path(out).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(out): raise ValueError('ZIP must be outside package')
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for path in sorted(out.rglob('*')):
            if path.is_file(): target.write(path,path.relative_to(out).as_posix())
    return dict(path=str(destination),sha256=P.sha(destination),bytes=destination.stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--zip',type=Path)
    parser.add_argument('--runtime',type=Path,help='verified self-contained Windows runtime bundle')
    parser.add_argument('--notices',type=Path,help='verified Windows source/runtime notice bundle')
    args=parser.parse_args()
    result=package(args.build,args.out,args.runtime,args.notices)
    result=dict(files=len(result['files']),product_ready=False,published=False)
    if args.zip: result['zip']=archive(args.out,args.zip)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
