"""Freeze/build and measure the Augur Windows HTTP profile; no math qualification."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
FEATURE = 'augur-model-qwen35-runtime/qwen35-augur-http'
PYTHON = sys.executable

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(p, v):
    with Path(p).open('x', encoding='utf8') as f: json.dump(v, f, ensure_ascii=False, indent=2)
def run(cmd, log, env, cwd):
    with log.open('wb') as f: code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    if code: raise RuntimeError(f'{code}: {log}')
def source_inputs(root):
    manifest = root/'augur-release-source.json'
    if manifest.is_file():
        receipt = json.loads(manifest.read_bytes())
        files = receipt['files']
        if receipt.get('schema') != 'quasar-release-source-v1' or not isinstance(files, dict) or not files:
            raise ValueError('invalid release source manifest')
        for name, expected in files.items():
            path = root/name
            if not name or ':' in name or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe release source path')
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or sha(path) != expected:
                raise ValueError('release source drift: '+name)
        return files, receipt['git_head']
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    files = {name:sha(root/name) for name in sorted(set(names)) if name and
        (root/name).is_file() and not name.startswith('evidence/')}
    return files, subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip()

def cargo_invocation(cargo, windows=None):
    if windows is None: windows = os.name == 'nt'
    if not windows: return cargo
    vcvars = os.environ.get('AUGUR_VCVARS64', r'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat')
    return 'cmd.exe /d /s /c "call '+subprocess.list2cmdline([vcvars])+' >nul && '+subprocess.list2cmdline(cargo)+'"'

def build(out, target, env):
    if (out/'build-receipt.json').exists(): raise RuntimeError('frozen build already exists')
    inputs, git_head = source_inputs(ROOT)
    files = {}
    for name, expected in sorted(inputs.items()):
        p = ROOT/name
        dest=out/'source'/name; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p,dest); files[name]=sha(dest)
        if files[name] != expected: raise ValueError('source changed while freezing: '+name)
    # Each failed build remains separate; never overwrite GPU evidence.
    save(out/'source-identity.json',dict(git_head=git_head,files=files))
    commands = [
        ['cargo','build','--release','--locked','--example','augur_session_http','--example','augur_session_jsonl','-p','augur-model-qwen35-runtime'],
        ['cargo','test','--release','--locked','--lib','--no-run','-p','augur-backend','-p','augur-model-qwen35','-p','augur-model-qwen35-runtime'],
        ['cargo','test','--release','--locked','--example','augur_session_http','--no-run','-p','augur-model-qwen35-runtime'],
        ['cargo','test','--release','--locked','--example','augur_session_jsonl','--no-run','-p','augur-model-qwen35-runtime'],
    ]
    bins = {}
    for i,cargo in enumerate(commands):
        cargo += ['--features',FEATURE,'--target-dir',str(target),'--message-format=json']
        log=out/f'build-{i}.log'; print(f'build {log}',flush=True)
        run(cargo_invocation(cargo),log,env,out/'source')
        for line in log.read_text(encoding='utf8',errors='replace').splitlines():
            try: v=json.loads(line)
            except ValueError: continue
            if v.get('reason')!='compiler-artifact' or not v.get('executable'): continue
            name=v['target']['name']; key=name+('-tests' if v['profile']['test'] else '')
            dest=out/(key+('.exe' if os.name == 'nt' else '')); shutil.copy2(v['executable'],dest)
            bins[key]=dict(path=str(dest),sha256=sha(dest))
    # Quantization is an engine-delivered Rust CLI, not a Python/llama subprocess.
    # Bind its executable and tests to the same frozen source identity.
    # The double-click entry (启动服务.exe) is a dependency-free Rust stub; rc.exe from the
    # same vcvars environment embeds its icon and version resource.
    steps=[('quantizer-'+mode,'tools/augur-quantize/Cargo.toml','augur-quantize',extra)
           for mode,extra in [('build',[]),('test',['--lib','--no-run'])]]
    if os.name=='nt': steps.append(('launch-build','tools/augur-launch/Cargo.toml','augur-launch',[]))
    for label,manifest,subdir,extra in steps:
        mode=label.rsplit('-',1)[1]
        cargo=['cargo',mode,'--release','--locked','--manifest-path',manifest,
               '--target-dir',str(target/subdir),'--message-format=json',*extra]
        log=out/(label+'.log');run(cargo_invocation(cargo),log,env,out/'source')
        for line in log.read_text(encoding='utf8',errors='replace').splitlines():
            try:v=json.loads(line)
            except ValueError:continue
            if v.get('reason')!='compiler-artifact' or not v.get('executable'):continue
            name=v['target']['name'];key=name+('-tests' if v['profile']['test'] else '')
            dest=out/(key+('.exe' if os.name=='nt' else ''));shutil.copy2(v['executable'],dest)
            bins[key]=dict(path=str(dest),sha256=sha(dest))
    save(out/'build-receipt.json',dict(feature=FEATURE,binaries=bins,source_sha256=sha(out/'source-identity.json')))
def cpu(out, env):
    receipt=json.loads((out/'build-receipt.json').read_bytes())
    for name,b in receipt['binaries'].items():
        if sha(b['path'])!=b['sha256']: raise RuntimeError('binary drift')
        if name.endswith('-tests'): run([b['path'],'--test-threads=1'],out/(name+'-cpu.log'),env,out/'source')
    tests = ['test_augur_api','test_qwen35_api']
    if receipt['feature'].endswith('/qwen35-augur-vision'): tests += ['test_augur_vision','test_augur_product']
    run([PYTHON,'-B','-m','unittest',*tests],out/'python-cpu.log',env,out/'source/scripts')
    save(out/'cpu-complete.json',dict(status='PASS',build_sha256=sha(out/'build-receipt.json')))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['build','cpu'],required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--target-dir',type=Path,default=ROOT/'artifacts/augur-cargo-target')
    ap.add_argument('--feature', choices=[FEATURE,'augur-model-qwen35-runtime/qwen35-augur-vision'],default=FEATURE)
    a=ap.parse_args();globals()['FEATURE']=a.feature;out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['AUGUR_CUDA_GENCODE']='arch=compute_70,code=sm_70';env['PATH']=str(Path(env['CUDA_PATH'])/'bin')+os.pathsep+env['PATH']
    if a.stage=='build':build(out,a.target_dir,env)
    else:cpu(out,env)
    print(a.stage+' PASS '+str(out),flush=True)
if __name__=='__main__':main()
