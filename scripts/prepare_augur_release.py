"""Prepare an isolated source context, not a published or GPU-qualified image."""
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCRIPTS = {
    'audit_augur_sampling_trajectory.py', 'test_augur_sampling_trajectory.py',
    'analyze_augur_sampling_rejections.py', 'test_augur_sampling_rejections.py',
    'analyze_augur_probability_trace.py', 'test_augur_probability_trace.py',
    'augur_launcher.py', 'augur_launch_config.py', 'test_augur_launcher.py',
    'augur_startup.py', 'test_augur_startup.py',
    'check_augur_source.py', 'test_augur_web.cjs', 'augur_bench.py', 'test_augur_bench.py', 'augur_bench_context.py', 'augur_eval.py', 'test_augur_eval.py', 'augur_bench_longctx.py',
    'augur_product.py', 'augur_api.py', 'augur_vision.py', 'qwen35_api.py',
    'augur_reasoning.py', 'augur_dflash_cost_accounting.py', 'run_augur_windows.py',
    'configure_augur_dsh.mjs', 'patch_augur_dsh.mjs', 'augur_monitor.py', 'augur_live_telemetry.py',
    'test_augur_monitor.py', 'test_augur_reasoning.py', 'test_augur_windows.py', 'package_augur_windows.py', 'test_augur_windows_package.py',
    'prepare_augur_windows_runtime.py', 'augur_portable.py', 'test_augur_windows_runtime.py', 'test_augur_safety_docs.py', 'test_augur_repository.py', 'test_augur_prefill_contract.py', 'build_augur_windows.py',
    'prepare_augur_windows_notices.py', 'export_augur_windows_source.py', 'audit_augur_windows_dependencies.py',
    'prepare_augur_release.py', 'prepare_augur_notices.py', 'prepare_augur_source_archive.py',
    'qwen35_chat.py', 'qwen35_tools.py', 'qwen35_persistence.py',
    'test_qwen35_long_history.py',
    'augur_bridge_entry.py', 'package_augur_bridge_update.py', 'test_augur_bridge_update.py',
    'nerkyor_chat.py', 'nerkyor_protocol.py', 'nerkyor_api.py', 'nerkyor_recovery.py',
    'test_nerkyor_chat.py', 'test_nerkyor_api.py', 'test_nerkyor_recovery.py', 'test_nerkyor_no_dflash.py',
    'test_augur_api.py', 'test_augur_vision.py', 'test_augur_product.py', 'test_qwen35_api.py',
    'test_qwen35_chat.py', 'test_qwen35_persistence.py', 'test_qwen35_tools.py'}
REPOSITORY_INPUTS = {
    '.gitignore', '.gitattributes', '.github/workflows/source-checks.yml',
    '.github/ISSUE_TEMPLATE/bug_report.md', '.github/ISSUE_TEMPLATE/feature_request.md',
    '.github/pull_request_template.md',
}
TOP = {'VERSION', 'CHANGELOG.md', 'BUILDING.md', 'CONTRIBUTING.md', 'README.md', 'README.zh-CN.md', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'qwen35-public-source-scope.json', 'LICENSE-APACHE', 'LICENSE-MIT', 'NOTICE', 'SECURITY.md'}
MODEL_CARD_INPUTS = {
    'model-cards/README.md',
    'model-cards/qwen3.8-27b-quasar-nvfp4/runtime-manifest.json',
    'model-cards/qwen3.8-27b-quasar-nvfp4/README.md',
    'model-cards/qwen3.8-27b-quasar-nvfp4/README.zh-CN.md',
    'model-cards/qwen3.8-27b-quasar-nvfp4/artifact-manifest.json',
}
SOURCE_DOCUMENTS = {
    'docs/README.md', 'docs/REPOSITORY_LAYOUT.md', 'docs/RELEASING.md',
    'docs/MODEL_NERKYOR_NO_DFLASH_ENGINE.zh-CN.md',
    'docs/CHAT_CONTINUITY_RELEASE_20261021.md',
    'docs/public/AUGUR_RELEASE_ACCEPTANCE_20261015.json',
    'docs/public/AUGUR_SAMPLING_SPEED_20261015.json',
    'docs/AUGUR_WINDOWS_USER_GUIDE.zh-CN.md',
    'docs/UNSAFE_TRUST_BOUNDARIES.md',
    'docs/AUGUR_SOURCE_RELEASE_README.md',
    'docs/AUGUR_WINDOWS_SOURCE_GUIDE.md',
    'docs/AUGUR_EMBEDDED_GATE045_ENGLISH_ACCEPTANCE_20261015.md',
    'docs/AUGUR_VISION_RESIDENCY_AND_LONG_PREFILL_20261015.md',
    'docs/public/AUGUR_VISION_RESIDENCY_20261015.json',
    'docs/public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json',
    'docs/AUGUR_FIXED_B8_HTTP_LEDGER_20261013.md',
    'docs/public/AUGUR_MODEL_SOURCES.md',
    'docs/public/RUNTIME_DEPENDENCIES.md',
    'docs/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.md',
    'docs/public/AUGUR_ENGLISH_PROPOSAL_CALIBRATION_20261016.json',
    'docs/AUGUR_SHARED_HEAD_RW4_20261016.md',
    'docs/public/AUGUR_SHARED_HEAD_RW4_20261016.json',
    'docs/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.md',
    'docs/public/AUGUR_DRAFT_Q8_B8_DIRECT_20261016.json',
    'docs/AUGUR_GDN_GLUE_B8_20261016.md',
    'docs/public/AUGUR_GDN_GLUE_B8_20261016.json',
    'docs/public/AUGUR_CONTEXT_ACCEPTANCE_20261016.json',
    'docs/public/AUGUR_LONG_CONTEXT_DECODE_20261017.json',
}
DSH_INPUTS = {
    'integrations/dsh/augur-provider.json', 'integrations/dsh/README.md',
    'integrations/dsh/dsh-patch.json', 'integrations/dsh/augur-compaction.json',
    *{'integrations/dsh/reasoning-recovery/'+name for name in ('package.json','index.mjs','single-flight.mjs','test.mjs','single-flight.test.mjs')},
}
BENCH_INPUTS = {'bench/english-thinking-14.json', 'bench/scenarios.json'}
NATIVE_FIXTURES = {
    'crates/augur-model-qwen35-runtime/tests/fixtures/nerkyor-tokenizer-reference.json',
    'crates/augur-model-qwen35/tests/fixtures/nerkyor-w4a16-config.json',
}

MONITOR_ASSETS = {'scripts/augur_monitor_web/'+name for name in ('index.html', 'app.js', 'i18n.js', 'style.css')}
REQUIRED_SOURCE = REPOSITORY_INPUTS | TOP | SOURCE_DOCUMENTS | MODEL_CARD_INPUTS | DSH_INPUTS | MONITOR_ASSETS | BENCH_INPUTS | {
    'scripts/augur_product.py', 'scripts/run_augur_windows.py',
    'scripts/configure_augur_dsh.mjs', 'scripts/patch_augur_dsh.mjs', 'scripts/augur_monitor.py',
    'scripts/augur_live_telemetry.py',
}
NATIVE_NOTICES = ('LICENSE-APACHE', 'LICENSE-MIT', 'NOTICE',
    'crates/augur-cuda-sys/native/llama_attention_vendor/LICENSE',
    'crates/augur-cuda-sys/native/licenses/NINFER-APACHE-2.0.txt',
    'crates/augur-cuda-sys/native/licenses/NINFER-NOTICE.txt',
    'scripts/augur_web/vendor/LUCIDE-LICENSE')
BASES = {
    'rust': ('library/rust', '1.96.0-slim-bookworm'),
    'cuda_devel': ('nvidia/cuda', '12.8.1-devel-ubuntu24.04'),
    'cuda_runtime': ('nvidia/cuda', '12.8.1-runtime-ubuntu24.04')}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    return bool(name and ':' not in name and '\\' not in name and not path.is_absolute()
        and all(part and part not in ('..', '.') for part in name.split('/')))


def selected(name):
    if not safe_name(name):
        return False
    # Admit only these exact repository files, never arbitrary hidden paths.
    if name in REPOSITORY_INPUTS:
        return True
    parts = PurePosixPath(name).parts
    if any(part.startswith('.') or part in ('target', 'node_modules', '__pycache__') for part in parts):
        return False
    if name in TOP | SOURCE_DOCUMENTS | MODEL_CARD_INPUTS | MONITOR_ASSETS | BENCH_INPUTS | NATIVE_FIXTURES:
        return True
    if len(parts) == 2 and parts[0] == 'scripts' and parts[1] in PUBLIC_SCRIPTS:
        return True
    if name in DSH_INPUTS:
        return True
    if name.startswith('scripts/augur_setup_web/'):
        return name in {'scripts/augur_setup_web/'+n for n in ('index.html','app.js','style.css','service.html','service.js','i18n.js')}
    if name.startswith('scripts/augur_web/'):
        return name.endswith(('.html', '.css', '.js', 'LUCIDE-LICENSE')) or name == 'scripts/augur_web/vendor/README.md'
    if name.startswith('container/augur/'):
        return name in ('container/augur/Dockerfile', 'container/augur/entry.py',
                       'container/augur/requirements.in', 'container/augur/README.md')
    if name.startswith('tools/augur-quantize/'):
        return (name in ('tools/augur-quantize/Cargo.toml', 'tools/augur-quantize/Cargo.lock', 'tools/augur-quantize/README.md')
                or (name.startswith('tools/augur-quantize/src/') and name.endswith('.rs'))
                or name in {'tools/augur-quantize/src/dflash/fixtures/'+f for f in ('config.json', 'safetensors-header.json', 'llama-tensor-inventory.json', 'tokenizer-small.json', 'tokenizer-config.json')})
    if name.startswith('tools/augur-launch/'):
        return name in {'tools/augur-launch/'+n for n in ('Cargo.toml', 'Cargo.lock', 'build.rs', 'augur.rc', 'augur.ico', 'make_icon.py', 'src/main.rs')}
    if name.startswith('crates/'):
        return name.endswith(('.rs', '.toml', '.cu', '.cuh', '.h', '.cpp')) or parts[-1].startswith(('LICENSE', 'NINFER-'))
    return False


def image_reference(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._/:-]*@sha256:[a-f0-9]{64}', value):
        raise ValueError('each toolchain/base image must use an explicit SHA256 digest')
    return value


def wheel_lock(wheels):
    rows, inventory, names = [], {}, set()
    for path in sorted(wheels.glob('*.whl')):
        with zipfile.ZipFile(path) as archive:
            metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
            if len(metadata) != 1:
                raise ValueError('one wheel metadata record required')
            value = BytesParser().parsebytes(archive.read(metadata[0]))
        name, version = value['Name'], value['Version']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name or '') or not re.fullmatch(r'[A-Za-z0-9_.+!-]+', version or ''):
            raise ValueError('invalid wheel name/version')
        normalized = re.sub(r'[-_.]+', '-', name.lower())
        if normalized in names:
            raise ValueError('duplicate resolved package')
        names.add(normalized)
        digest = sha(path)
        rows.append(name+'=='+version+' --hash=sha256:'+digest)
        inventory[path.name] = dict(name=name, version=version, sha256=digest)
    if not {'tokenizers', 'jinja2', 'pillow', 'jsonschema'} <= names:
        raise ValueError('required Augur runtime wheels missing')
    return '\n'.join(rows)+'\n', inventory


def copy_locked_wheels(context, destination):
    preparation = validate_context(context)
    destination.mkdir()
    for name, receipt in sorted(preparation['wheels'].items()):
        source = context/'runtime-wheels'/name
        if sha(source) != receipt['sha256']:
            raise ValueError('locked wheel changed before copying')
        shutil.copy2(source, destination/name)
        if sha(source) != receipt['sha256'] or sha(destination/name) != receipt['sha256']:
            raise ValueError('locked wheel changed while copying')
    lock, inventory = wheel_lock(destination)
    if inventory != preparation['wheels'] or lock != (context/'requirements.lock').read_text(encoding='ascii'):
        raise ValueError('reused wheel inventory/lock mismatch')
    return lock, inventory, sha(context/'release-preparation.json')


def prepare(out, download_wheels=False, wheel_context=None):
    if download_wheels and wheel_context is not None:
        raise ValueError('download and reuse are mutually exclusive')
    if wheel_context is not None:
        wheel_context = wheel_context.resolve()
        validate_context(wheel_context)
    out = out.resolve()
    if out.is_relative_to(ROOT):
        raise ValueError('isolated context must be outside the working source tree')
    out.mkdir(exist_ok=False)
    names = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    files = {}
    for name in sorted(set(names)):
        if not selected(name):
            continue
        source = ROOT/name
        if not source.is_file():
            continue
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            raise ValueError('release source symlink/escape: '+name)
        before = sha(source)
        destination = out/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != before or sha(source) != before:
            raise ValueError('source changed during capture: '+name)
        files[name] = before
    missing = REQUIRED_SOURCE - files.keys()
    if missing:
        raise ValueError('required release inputs missing: '+', '.join(sorted(missing)))
    receipt = dict(schema='quasar-release-source-v1',
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
        dirty_source_capture=True, files=files, default_main_model='quasar-27b',
        model_weights_included=False, draft_weights_included=False,
        selected_launch_preset='windows_fixed_b8_v1', active_dflash_in_selected_preset=True,
        product_ready=False, numerical_qualified=False, dflash_active=False)
    (out/'augur-release-source.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf8')
    inventory, wheel_source_sha = {}, None
    if download_wheels:
        wheels = out/'runtime-wheels'
        wheels.mkdir()
        subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'download', '--index-url', 'https://pypi.org/simple',
            '--only-binary=:all:', '--platform', 'manylinux_2_28_x86_64', '--platform', 'manylinux2014_x86_64',
            '--python-version', '312', '--implementation', 'cp', '--abi', 'cp312',
            '-r', str(out/'container/augur/requirements.in'), '--dest', str(wheels)], check=True)
        lock, inventory = wheel_lock(wheels)
        (out/'requirements.lock').write_text(lock, encoding='ascii')
    elif wheel_context is not None:
        lock, inventory, wheel_source_sha = copy_locked_wheels(wheel_context, out/'runtime-wheels')
        (out/'requirements.lock').write_text(lock, encoding='ascii')
    (out/'release-preparation.json').write_text(json.dumps(dict(
        source_manifest_sha256=sha(out/'augur-release-source.json'), wheels=inventory,
        requirements_lock_sha256=sha(out/'requirements.lock') if inventory else None,
        wheel_source_preparation_sha256=wheel_source_sha,
        wheels_locked=bool(inventory), linux_build_verified=False, image_smoke_verified=False,
        published=False, model_rights_verified=False, product_ready=False), indent=2)+'\n', encoding='utf8')
    return receipt


def validate_context(root):
    root = root.resolve()
    preparation = json.loads((root/'release-preparation.json').read_bytes())
    source = json.loads((root/'augur-release-source.json').read_bytes())
    if preparation['source_manifest_sha256'] != sha(root/'augur-release-source.json') or not preparation['wheels_locked']:
        raise ValueError('unbound source or missing wheel lock')
    if source['schema'] != 'quasar-release-source-v1' or source['default_main_model'] != 'quasar-27b':
        raise ValueError('not a QUASAR source context')
    if preparation['requirements_lock_sha256'] != sha(root/'requirements.lock'):
        raise ValueError('runtime requirement lock drift')
    lock, inventory = wheel_lock(root/'runtime-wheels')
    if inventory != preparation['wheels'] or (root/'requirements.lock').read_text(encoding='ascii') != lock:
        raise ValueError('runtime wheel byte/lock drift')
    expected = {'augur-release-source.json', 'release-preparation.json', 'requirements.lock'}
    for name, digest in source['files'].items():
        if not selected(name) or sha(root/name) != digest:
            raise ValueError('unexpected or changed source: '+name)
        expected.add(name)
    expected.update('runtime-wheels/'+name for name in inventory)
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('symlink or escape in context')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError('unexpected/missing files in isolated context')
    return preparation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    wheels = parser.add_mutually_exclusive_group()
    wheels.add_argument('--download-wheels', action='store_true')
    wheels.add_argument('--wheel-context', type=Path, help='reuse the verified complete Linux wheel lock from an earlier context')
    args = parser.parse_args()
    result = prepare(args.out, args.download_wheels, args.wheel_context)
    print(json.dumps(dict(context=str(args.out.resolve()), source_files=len(result['files']),
        main_model=result['default_main_model'], linux_build_verified=False)))


if __name__ == '__main__':
    main()
