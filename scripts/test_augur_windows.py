"""Selected Windows launcher identity/policy tests; no model or GPU needed."""
import contextlib
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import run_augur_windows as W
import augur_reasoning as R

class WindowsPreset(unittest.TestCase):
    def test_gate_size_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            gate=Path(d)/'gate.f32'
            gate.write_bytes(b'short')
            with self.assertRaisesRegex(ValueError,'256 x 5120'): W.gate_identity(gate)
            payload=b'\0'*(256*5120*4);gate.write_bytes(payload)
            with self.assertRaisesRegex(ValueError,'SHA256'): W.gate_identity(gate)
            with patch.object(R,'ENGLISH_GATE_SHA256',hashlib.sha256(payload).hexdigest()):
                self.assertEqual(W.gate_identity(gate)['path'],str(gate.resolve()))

    def test_missing_or_wrong_gate_cannot_clear_environment_or_spawn(self):
        with patch.dict(os.environ,{'AUGUR_OLD_EXPERIMENT':'keep'},clear=True):
            with patch.object(W,'gate_identity',side_effect=ValueError('bad gate')):
                with self.assertRaises(ValueError): W.configure('missing')
            self.assertEqual(os.environ['AUGUR_OLD_EXPERIMENT'],'keep')

    def test_clean_fixed_b8_fp32_global_gate_without_double_scale(self):
        gate=dict(path='external-gate',sha256=R.ENGLISH_GATE_SHA256)
        with patch.dict(os.environ,{'AUGUR_OLD_EXPERIMENT':'1','OTHER':'keep'},clear=True),patch.object(W,'gate_identity',return_value=gate):
            W.configure('gate')
            self.assertNotIn('AUGUR_OLD_EXPERIMENT',os.environ)
            self.assertEqual(os.environ['OTHER'],'keep')
            self.assertEqual(os.environ['AUGUR_DFLASH_SELECTOR_SCALE'],'1')
            self.assertEqual(os.environ['AUGUR_DFLASH_SELECTOR_GATE_ACCUMULATION'],'fp32')
            self.assertEqual(os.environ['AUGUR_DFLASH_GATE'],'fixed_b8')
            self.assertEqual(os.environ['AUGUR_DFLASH_TOPK_TILED'],'1')
            self.assertEqual(os.environ['AUGUR_SAMPLING_TOPK'],'tiled')
            self.assertEqual(os.environ['AUGUR_DFLASH_SELECTOR_PATH'],'greedy')
            self.assertEqual(os.environ['AUGUR_DFLASH_PROBABILITY_PREFIX'],'7')
            self.assertEqual(os.environ['AUGUR_DFLASH_PROBABILITY_TEMPERATURES'],'0.875,0.875,0.875,0.75,0.75,0.75,0.75')
            self.assertEqual(os.environ['AUGUR_DFLASH_PROBABILITY_PAIR_SCALES'],'1,1.5,1.5,1.5,1.5,1.5,1.5')
            self.assertEqual(os.environ['AUGUR_DFLASH_VERIFICATION'],'token')
            self.assertEqual(os.environ['AUGUR_SHARED_HEAD_KERNEL'],'rw4')
            self.assertEqual(os.environ['AUGUR_DFLASH_Q8_KERNEL'],'direct')
            self.assertEqual(os.environ['AUGUR_GDN_INPUT_KERNEL'],'fused')
            self.assertNotIn('AUGUR_DFLASH_PROBABILITY_POS234_TEMPERATURE',os.environ)
            self.assertNotIn('AUGUR_DFLASH_HTTP_POSITION',os.environ)
            self.assertNotIn('AUGUR_DFLASH_COUPLED_SAMPLE',os.environ)
            self.assertEqual(R.selector_experiment(os.environ,True),R.GLOBAL_GATE_POLICY)
            self.assertIsNone(R.selector_request(R.ENGLISH_GATE_POLICY,R.policy(True,'zh'),False))
            self.assertIsNone(R.selector_request(R.ENGLISH_GATE_POLICY,R.policy(True,'en'),True))

    def test_embedded_gate_is_global_default_without_external_file_or_double_scale(self):
        with patch.dict(os.environ,{'AUGUR_OLD':'1'},clear=True):
            W.configure()
            self.assertEqual(R.selector_experiment(os.environ,True),R.GLOBAL_GATE_POLICY)
            self.assertNotIn('AUGUR_DFLASH_SELECTOR_GATE_F32',os.environ)
            self.assertNotIn('AUGUR_DFLASH_SELECTOR_GATE_F32_SHA256',os.environ)
            self.assertEqual(os.environ['AUGUR_DFLASH_SELECTOR_SCALE'],'1')
            for enabled,lang in [(False,'en'),(True,'en'),(True,'zh')]:
                self.assertEqual(R.selector_request(R.GLOBAL_GATE_POLICY,R.policy(enabled,lang),False),R.GLOBAL_GATE_POLICY)
                self.assertIsNone(R.selector_request(R.GLOBAL_GATE_POLICY,R.policy(enabled,lang),True))
            os.environ['AUGUR_DFLASH_SELECTOR_GATE_F32']='conflicting'
            with self.assertRaises(Exception):R.selector_experiment(os.environ,True)

    def test_argument_boundaries_and_observer_teardown(self):
        product=SimpleNamespace(main=Mock(side_effect=RuntimeError('stop')))
        tracker=SimpleNamespace(closed=Mock())
        telemetry=SimpleNamespace(install=Mock(return_value=tracker))
        with patch.dict('sys.modules',augur_product=product,augur_live_telemetry=telemetry),patch.object(W,'configure'):
            with self.assertRaisesRegex(RuntimeError,'stop'):
                W.main(['--selector-gate-file','gate with space.f32','--live-file','live.json','--build','build with space'])
        product.main.assert_called_once_with(['--build','build with space'])
        tracker.closed.set.assert_called_once_with()

    def test_sampling_seed_is_explicit_validated_and_survives_environment_reset(self):
        with patch.dict(os.environ,{'AUGUR_SAMPLING_SEED':'999'},clear=True):
            W.configure(sampling_seed=2)
            self.assertEqual(os.environ['AUGUR_SAMPLING_SEED'],'2')
            for value in (-1,1<<64,True,1.0):
                with self.assertRaises(ValueError):W.configure(sampling_seed=value)
                self.assertEqual(os.environ['AUGUR_SAMPLING_SEED'],'2')
            W.configure()
            self.assertEqual(os.environ['AUGUR_SAMPLING_SEED'],'0')
        product=SimpleNamespace(main=Mock(return_value=0))
        with patch.dict('sys.modules',augur_product=product),patch.object(W,'configure') as configure:
            W.main(['--sampling-seed','2','--build','build'])
        configure.assert_called_once_with(None,'on_demand',2)
        product.main.assert_called_once_with(['--build','build'])

    def test_vision_residency_defaults_and_validation(self):
        with patch.dict(os.environ, {"AUGUR_OLD": "preserve"}, clear=True):
            with self.assertRaises(ValueError): W.configure(vision_residency="forever")
            self.assertEqual(os.environ["AUGUR_OLD"], "preserve")
            W.configure()
            self.assertEqual(os.environ["AUGUR_VISION_RESIDENCY"], "on_demand")
            W.configure(vision_residency="auto")
            self.assertEqual(os.environ["AUGUR_VISION_RESIDENCY"], "auto")
            W.configure()
            self.assertEqual(os.environ["AUGUR_VISION_RESIDENCY"], "on_demand")

    def test_help_does_not_require_gate_or_load_weights(self):
        product=SimpleNamespace(main=Mock())
        with patch.dict('sys.modules',augur_product=product),patch.object(W,'configure') as configure,contextlib.redirect_stdout(io.StringIO()):
            W.main(['--help'])
        configure.assert_not_called()
        product.main.assert_called_once_with(['--help'])

if __name__=='__main__': unittest.main()
