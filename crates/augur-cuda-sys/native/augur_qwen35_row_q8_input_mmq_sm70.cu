// Separate specialization and kernel names: no process-global layout switch.
// Same isolated fast-math policy as the pinned wire-layout MMQ translation unit.
#define AUGUR_QWEN35_ROW_Q8_INPUT 1
#define augur_qwen35_launch_llama_mmq_sm70 augur_qwen35_launch_row_q8_input_mmq_sm70
#define mul_mat_q augur_qwen35_row_q8_input_mul_mat_q
#define mul_mat_q_stream_k_fixup augur_qwen35_row_q8_input_stream_k_fixup
#include "augur_qwen35_prefill_llama_mmq_sm70.cu"
