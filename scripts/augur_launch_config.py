"""Validated local launcher configuration. No arbitrary environment or shell commands."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

CONTEXT_CHOICES = (8192, 32768, 65536, 131072, 196608, 262144)
LANGUAGES = ('zh-CN', 'en')
LANG = 'zh-CN'  # interface language for launcher messages; set from the saved preference


def L(zh, en):
    """Pick the user-facing message for the current interface language."""
    return en if LANG == 'en' else zh


def load_preferences(data):
    """Interface preferences live apart from config.json, so choosing a language on first run
    never looks like a saved engine configuration (which would autostart)."""
    global LANG
    try:
        value = json.loads((Path(data) / 'preferences.json').read_bytes())
        language = value.get('language') if type(value) is dict else None
    except (OSError, ValueError):
        language = None
    LANG = language if language in LANGUAGES else 'zh-CN'
    return dict(language=LANG, chosen=language in LANGUAGES)


def save_language(data, language):
    global LANG
    if language not in LANGUAGES:
        raise ValueError(L('不支持的语言。', 'Unsupported language.'))
    atomic_json(Path(data) / 'preferences.json', dict(language=language))
    LANG = language
    return dict(language=LANG, chosen=True)
SOURCE = Path(__file__).resolve().parents[1]
MODEL_CARD = SOURCE / 'model-cards/qwen3.8-27b-quasar-nvfp4'
MANIFEST = MODEL_CARD / 'runtime-manifest.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def defaults(package):
    data = Path(package).resolve() / 'user-data'
    return dict(schema=1, model_dir='', draft_source_dir='', draft_quantized_dir='',
                quant_output_dir=str(data / 'models/dflash-q8-fp32-gate045'),
                context_tokens=262144, default_output_tokens=262144 * 95 // 100, port=8294,
                vision_residency='on_demand', cache_enabled=False,
                cache_dir=str(data / 'prefix-cache'), export_env=True)


def normalize(value, package):
    if type(value) is dict and 'export_env' not in value:
        value = dict(value, export_env=True)  # configurations saved before the switch existed
    if type(value) is not dict or set(value) != set(defaults(package)):
        raise ValueError(L('配置字段不完整或含不支持的字段。', 'Settings are incomplete or contain unsupported fields.'))
    value = dict(value)
    if type(value['schema']) is not int or value['schema'] != 1:
        raise ValueError(L('不支持的配置版本。', 'Unsupported settings version.'))
    for name in ('model_dir', 'draft_source_dir', 'draft_quantized_dir', 'quant_output_dir', 'cache_dir'):
        path = value[name]
        if type(path) is not str or len(path) > 32700 or any(ord(c) < 32 for c in path):
            raise ValueError(L('无效路径：', 'Invalid path: ') + name)
        if path and (not Path(path).is_absolute() or path.startswith(('\\\\', '//'))):
            raise ValueError(L('请选择本机绝对路径，不支持网络共享路径：', 'Use a local absolute path; network shares are not supported: ') + name)
        value[name] = str(Path(path).resolve()) if path else ''
    n = value['context_tokens']
    if type(n) is not int or n not in CONTEXT_CHOICES:
        raise ValueError(L('上下文只能选择 8K、32K、64K、128K、192K 或 256K。', 'Context must be 8K, 32K, 64K, 128K, 192K or 256K.'))
    tokens = value['default_output_tokens']
    if type(tokens) is not int or not 1 <= tokens <= n * 95 // 100:
        raise ValueError(L('默认输出上限必须在 1 到所选上下文的 95% 之间。', 'The default output limit must be between 1 and 95% of the selected context.'))
    if type(value['port']) is not int or not 1024 <= value['port'] <= 65535:
        raise ValueError(L('API 端口必须在 1024～65535 之间。', 'The API port must be between 1024 and 65535.'))
    if value['vision_residency'] not in ('on_demand', 'auto'):
        raise ValueError(L('无效的 Vision 驻留策略。', 'Invalid vision residency.'))
    if type(value['cache_enabled']) is not bool or type(value['export_env']) is not bool:
        raise ValueError(L('开关必须为布尔值。', 'Switches must be true or false.'))
    if value['cache_enabled'] and not value['cache_dir']:
        raise ValueError(L('请选择缓存目录。', 'Choose a cache folder.'))
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.augur-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def target_preflight(config, full_hash=True):
    root = Path(config['model_dir'])
    if not config['model_dir'] or not root.is_dir():
        raise ValueError(L('请选择 QUASAR 主模型目录。', 'Choose the QUASAR target model folder.'))
    manifest = json.loads(MANIFEST.read_bytes())
    expected = {'config.json': manifest['checkpoint']['config_sha256'],
                'model.safetensors.index.json': manifest['checkpoint']['index_sha256'],
                'tokenizer.json': manifest['tokenizer_sha256'],
                'tokenizer_config.json': manifest['tokenizer_config_sha256'],
                'generation_config.json': manifest['generation_config_sha256'],
                'chat_template.jinja': manifest['chat_template_sha256']}
    for name, digest in (expected | manifest['checkpoint']['shards']).items():
        path = root / name
        if not path.is_file():
            raise ValueError(L('缺少主模型文件：', 'Missing target model file: ') + name)
        if (full_hash or name in expected) and sha(path) != digest:
            raise ValueError(L('主模型文件身份不匹配：', 'Target model file does not match: ') + name)
    return dict(manifest=str(MANIFEST), manifest_sha256=sha(MANIFEST))


def draft_environment(config):
    folder = Path(config['draft_quantized_dir'])
    if not config['draft_quantized_dir'] or not folder.is_dir():
        raise ValueError(L('请先量化 DFlash，或选择已有 Q8_0＋FP32 gate0.45 目录。', 'Quantize DFlash first, or choose an existing Q8_0 + FP32 gate0.45 folder.'))
    admitted = json.loads((MODEL_CARD / 'artifact-manifest.json').read_bytes())['draft']
    weights, metadata = folder / 'model-q8_0.gguf', folder / 'config.json'
    for path, digest in ((weights, admitted['sha256']), (metadata, admitted['config_sha256'])):
        if not path.is_file() or sha(path) != digest:
            raise ValueError(L('DFlash 文件不是当前支持的 Q8_0＋FP32 gate0.45：', 'This DFlash file is not the supported Q8_0 + FP32 gate0.45 build: ') + path.name)
    return dict(AUGUR_DRAFT_PATH=str(weights), AUGUR_DRAFT_SHA256=admitted['sha256'],
                AUGUR_DRAFT_CONFIG=str(metadata), AUGUR_DRAFT_CONFIG_SHA256=admitted['config_sha256'])


def quantization_source(config):
    target_preflight(config, full_hash=False)
    folder = Path(config['draft_source_dir'])
    if not config['draft_source_dir'] or not folder.is_dir():
        raise ValueError(L('请选择原始 DFlash BF16 目录。', 'Choose the original DFlash BF16 folder.'))
    expected = json.loads((MODEL_CARD / 'artifact-manifest.json').read_bytes())['draft']['source_safetensors_sha256']
    if not (folder / 'config.json').is_file() or not (folder / 'model.safetensors').is_file():
        raise ValueError(L('DFlash 原始目录需要 config.json 和 model.safetensors。', 'The original DFlash folder needs config.json and model.safetensors.'))
    if sha(folder / 'model.safetensors') != expected:
        raise ValueError(L('原始 DFlash 权重 SHA256 不匹配当前支持的版本。', 'The original DFlash weights do not match the supported SHA256.'))
    return expected


CACHE_SCOPES = ('chat', 'agent', 'bench')
_CACHE_SCOPE_OK = set('abcdefghijklmnopqrstuvwxyz0123456789-_')


def cache_scope(value):
    """A cache scope is one path segment: who may reuse a persisted prefix. Two runs that must not
    see each other's cache use different scopes instead of deleting the shared directory."""
    if not isinstance(value, str) or not 1 <= len(value) <= 40 or any(c not in _CACHE_SCOPE_OK for c in value.lower()):
        raise ValueError(L('缓存作用域只能包含字母、数字、短横线和下划线。',
                           'A cache scope may contain only letters, digits, dash and underscore.'))
    return value.lower()


def prefix_cache_root(cache_dir, context_tokens, scope='chat'):
    """<cache>/context-<N>/<scope>. The native side appends its build-identity digest below this,
    so the scope is the only part an operator controls and the only part that separates runs."""
    return Path(cache_dir) / ('context-' + str(context_tokens)) / cache_scope(scope)


def writable_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.augur-write-check-', dir=path)
    os.close(fd)
    Path(name).unlink()
    return shutil.disk_usage(path).free
