"""CPU-only guard against accidentally resurrecting retired Augur prefill."""
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]
class RetirementTests(unittest.TestCase):
 def test_removed_runtime_and_native_lanes(self):
  for name in [
   'crates/augur-model-qwen35-runtime/src/checkpoint_target_cuda/source_carry.rs',
   'crates/augur-model-qwen35-runtime/src/checkpoint_target_cuda/wide_prefill.rs',
   'crates/augur-backend/src/cuda_qwen35_prefill_cublas/qpn_wide.rs',
   'crates/augur-cuda-sys/native/augur_qwen35_ffn_carry.cuh',
   'crates/augur-cuda-sys/native/augur_qwen35_gdn_carry.cuh',
   'crates/augur-cuda-sys/native/augur_qwen35_full_carry.cuh',
   'crates/augur-cuda-sys/native/augur_qwen35_kv_source_repair.cuh']:
   with self.subTest(path=name):self.assertFalse((ROOT/name).exists())
 def test_adapter_shares_common_contraction_and_no_compensation(self):
  native=(ROOT/'crates/augur-cuda-sys/native/augur_qwen35_augur_prefill.cuh').read_text(encoding='utf8')
  self.assertIn('llama_f16_gemm_postscale(',native)
  for old in ['cublasDgemm','__dadd_rn','__dmul_rn','split_limbs','ffn_carry','source_carry']:
   self.assertNotIn(old,native)
  common=(ROOT/'crates/augur-cuda-sys/native/augur_qwen35_prefill_cublas_sm70.cu').read_text(encoding='utf8')
  self.assertIn('return llama_f16_gemm_postscale(stream,plan,',common)
 def test_profile_is_explicitly_speed_first(self):
  text=(ROOT/'crates/augur-backend/src/cuda_qwen35_prefill_cublas/augur.rs').read_text(encoding='utf8')
  self.assertIn('quasar_original_qpn_f16_f32_speed_first_v1',text)
if __name__=='__main__':unittest.main()
