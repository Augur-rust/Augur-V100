// SPDX-License-Identifier: Apache-2.0
// Private cross-TU descriptor for the grouped direct-B8 QPN projection launch.
// Built only by the validated norm-group program in the cuBLAS plan TU and
// consumed only by augur_qpn_project_group_b8_enqueue in the QPN TU, so the
// projection tiles keep the QPN TU's compile flags (bit-identical to project).
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
struct augur_qpn_group_member {
    uint64_t weight, output, mirror; // mirror == 0: no second store
    int32_t rows, out_cols, first_tile;
    float scale;
};
struct augur_qpn_group {
    augur_qpn_group_member m[4];
    int32_t count, cols, tokens, tiles;
};
extern "C" cudaError_t augur_qpn_project_group_b8_enqueue(
    cudaStream_t stream, const augur_qpn_group* group, uint64_t staged_f16);
