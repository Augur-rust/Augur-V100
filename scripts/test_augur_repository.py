"""Public repository navigation, model identity and renamed prefill wiring."""
import ast
import json
from pathlib import Path
import re
import tomllib
import unittest
import build_augur_windows as B
import prepare_augur_release as R

MODEL = 'qwen3.8-27b-quasar-nvfp4'
CARD = 'model-cards/' + MODEL
DOCUMENTS = (
    'README.md', 'README.zh-CN.md', 'BUILDING.md', 'CONTRIBUTING.md',
    'docs/README.md', 'docs/REPOSITORY_LAYOUT.md', 'docs/RELEASING.md',
    'CHANGELOG.md', 'model-cards/README.md',
    '.github/ISSUE_TEMPLATE/bug_report.md', '.github/ISSUE_TEMPLATE/feature_request.md',
    '.github/pull_request_template.md',
    CARD + '/README.md', CARD + '/README.zh-CN.md',
)


def read(relative):
    return (R.ROOT / relative).read_text(encoding='utf-8')


def constants(relative):
    values = {}
    for node in ast.parse(read(relative)).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value.value
    return values


class AugurRepository(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(read(CARD + '/artifact-manifest.json'))
        self.evidence = json.loads(read('docs/public/AUGUR_ENGLISH_THINKING_GATE045_20261015.json'))

    def test_current_entry_points_are_required_release_inputs(self):
        for relative in (*DOCUMENTS, CARD + '/artifact-manifest.json'):
            with self.subTest(path=relative):
                self.assertTrue(R.selected(relative))
                self.assertIn(relative, R.REQUIRED_SOURCE)
                self.assertTrue((R.ROOT / relative).is_file())
        self.assertTrue(R.selected('scripts/build_augur_windows.py'))
        self.assertEqual(B.ROOT, R.ROOT)

    def test_repository_dotfiles_are_exact_allowlist_only(self):
        for relative in R.REPOSITORY_INPUTS:
            self.assertTrue(R.selected(relative), relative)
            self.assertIn(relative, R.REQUIRED_SOURCE)
            self.assertTrue((R.ROOT / relative).is_file())
        for relative in ('.git/config', '.env', '.env.local', '.github/private.json',
                         '.github/workflows/private.yml', '.codex/auth.json',
                         '../.gitignore', 'scripts/.env', '.github/../.env'):
            self.assertFalse(R.selected(relative), relative)

    def test_exactly_one_supported_target_and_no_weight_export(self):
        directories = sorted(p.name for p in (R.ROOT / 'model-cards').iterdir() if p.is_dir())
        self.assertEqual(directories, [MODEL])
        self.assertEqual(self.manifest['model_id'], MODEL)
        self.assertEqual(self.manifest['role'], 'documentation_only_not_runtime_admission')
        self.assertEqual(self.manifest['schema_version'], 1)
        self.assertFalse(self.manifest['distribution']['weights_included'])
        self.assertTrue(self.manifest['distribution']['runtime_admission_manifest_included'])
        for relative in (CARD + '/model.safetensors', CARD + '/private.json',
                         CARD + '/model-q8_0.gguf', 'model-cards/another-model/README.md',
                         'docs/public/QWEN_MODEL_SOURCES.md'):
            self.assertFalse(R.selected(relative), relative)

    def test_documentation_links_resolve_inside_public_source(self):
        for relative in DOCUMENTS:
            source = R.ROOT / relative
            for target in re.findall(r'\]\(([^)]+)\)', read(relative)):
                if '://' in target or target.startswith('#'):
                    continue
                path = (source.parent / target.split('#', 1)[0]).resolve()
                with self.subTest(document=relative, link=target):
                    self.assertTrue(path.is_relative_to(R.ROOT.resolve()))
                    self.assertTrue(path.is_file())
                    self.assertTrue(R.selected(path.relative_to(R.ROOT).as_posix()))

    def test_model_card_matches_product_bounds_and_profiles(self):
        api = constants('scripts/augur_api.py')
        profile = json.loads(read('integrations/dsh/augur-provider.json'))['models'][0]
        runtime = self.manifest['runtime']
        self.assertEqual(self.manifest['product_api_alias'], profile['id'])
        self.assertEqual(runtime['capacity_tokens'], api['CAPACITY'])
        self.assertEqual(runtime['max_completion_tokens'], api['MAX_OUTPUT'])
        self.assertEqual(runtime['text_profile'], api['AUGUR_PROFILE'])
        self.assertEqual(runtime['dflash_profile'], api['DFLASH_PROFILE'])
        self.assertEqual(runtime['vision_default'], 'on_demand')
        self.assertEqual(runtime['selected_dflash_mode'], 'fixed_b8')
        for language in ('README.md', 'README.zh-CN.md'):
            text = read(CARD + '/' + language)
            self.assertIn(f"{api['CAPACITY']:,}", text)
            self.assertIn(f"{api['MAX_OUTPUT']:,}", text)
            example = json.loads(re.findall(r'```json\n(.*?)\n```', text, re.S)[0])
            self.assertEqual(example['model'], profile['id'])
            self.assertLessEqual(example['max_completion_tokens'], api['MAX_OUTPUT'])
            self.assertEqual(example['thinking_language'], 'en')

    def test_draft_identity_and_measurements_match_recorded_evidence(self):
        self.assertEqual(self.manifest['draft']['sha256'], self.evidence['draft_gguf_sha256'])
        self.assertEqual(self.manifest['draft']['selector_gate']['sha256'], self.evidence['gate_sha256'])
        self.assertEqual(self.manifest['evaluation']['evaluation_build_receipt_sha256'],
                         self.evidence['evaluation_build_receipt_sha256'])
        self.assertEqual(self.manifest['runtime']['tested_hardware'], self.evidence['hardware'])
        ids = {'en_conditional_probability', 'en_nonadjacent_combinations', 'en_pattern_waiting_time',
               'holdout_python_histogram', 'holdout_javascript_zip_longest', 'zh_control', 'tool_lookup', 'tool_result'}
        for language in ('README.md', 'README.zh-CN.md'):
            text = read(CARD + '/' + language)
            for row in self.evidence['summary']:
                if row['id'] not in ids:
                    continue
                sample = row['embedded']
                expected = (f"| {sample['native_tps']:.2f} | {sample['http_tps']:.2f} | "
                            f"{sample['accepted_over_proposed']:.2%} | {sample['accepted_over_checked']:.2%} |")
                self.assertIn(expected, text, (language, row['id']))

    def test_target_metadata_has_no_private_paths_or_keys(self):
        self.assertEqual(self.manifest['architecture']['num_hidden_layers'],
                         self.manifest['architecture']['gdn_layers'] + self.manifest['architecture']['full_attention_layers'])
        self.assertEqual(len(self.manifest['target']['shards']), 5)
        for digest in self.manifest['target']['shards'].values():
            self.assertRegex(digest, r'^[0-9a-f]{64}$')
        text = read(CARD + '/artifact-manifest.json')
        self.assertIsNone(re.search(r'[A-Za-z]:[\\/]|Bearer |BEGIN .*PRIVATE KEY', text))

    def test_cuda_header_ffi_and_build_inputs_share_new_names(self):
        header = 'augur_qwen35_augur_prefill.cuh'
        native = read('crates/augur-cuda-sys/native/' + header)
        self.assertIn(header, read('crates/augur-cuda-sys/build.rs'))
        self.assertIn('#include "' + header + '"', read('crates/augur-cuda-sys/native/augur_qwen35_prefill_cublas_sm70.cu'))
        ffi = read('crates/augur-cuda-sys/src/qwen35_qpn.rs')
        for suffix in ('prefill_preflight', 'prefill_submit', 'prefill_group_submit', 'silu_down_submit'):
            symbol = 'augur_cuda_qwen35_augur_' + suffix + '_v1'
            self.assertIn(symbol + '(', native)
            self.assertIn(symbol + '(', ffi)
        runtime = tomllib.loads(read('crates/augur-model-qwen35-runtime/Cargo.toml'))
        self.assertIn('qwen35-augur-http', runtime['features']['qwen35-augur-vision'])
        self.assertEqual(B.FEATURE, 'augur-model-qwen35-runtime/qwen35-augur-http')


if __name__ == '__main__':
    unittest.main()
