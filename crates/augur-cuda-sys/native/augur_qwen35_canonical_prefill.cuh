// Included inside the full-attention implementation, after the V4 kernels.
// Scores may be generated in parallel, but each row still visits the exact
// V3/V4 split interval in increasing key order. No tree/softmax replacement.
template<unsigned Q>
__global__ void augur_qwen35_canonical_consume_scores_v1(
    const float* scores, const unsigned char* values, float* workspace,
    unsigned base, unsigned rows, unsigned key_begin, unsigned key_count,
    unsigned score_stride) {
    constexpr unsigned PAR = Q < 4 ? Q : 4, STATES = Q / PAR;
    const unsigned split = blockIdx.x % AUGUR_QWEN35_FULL_SPLITS_V2;
    const unsigned head = blockIdx.x / AUGUR_QWEN35_FULL_SPLITS_V2;
    const unsigned row0 = blockIdx.y * Q;
    const unsigned union_first = (uint64_t(base + row0 + 1) * split) / AUGUR_QWEN35_FULL_SPLITS_V2;
    const unsigned union_last = (uint64_t(base + min(row0 + Q, rows)) * (split + 1))
        / AUGUR_QWEN35_FULL_SPLITS_V2;
    // Tile zero initializes every partial, including empty early intervals.
    // Subsequent nonintersecting tiles must leave that initialized state alone.
    if (key_begin != 0 && (union_last <= key_begin || union_first >= key_begin + key_count)) return;
    const unsigned group = threadIdx.x / 32, local_row = group / 6;
    const unsigned qhead = head * 6 + group % 6, lane = threadIdx.x % 32;
    unsigned first[STATES], last[STATES], row[STATES];
    float out[STATES][8] = {}, maximum[STATES], sum[STATES];
    #pragma unroll
    for (unsigned s = 0; s < STATES; ++s) {
        row[s] = row0 + local_row + s * PAR;
        first[s] = (uint64_t(base + row[s] + 1) * split) / AUGUR_QWEN35_FULL_SPLITS_V2;
        last[s] = (uint64_t(base + row[s] + 1) * (split + 1)) / AUGUR_QWEN35_FULL_SPLITS_V2;
        maximum[s] = -INFINITY; sum[s] = 0;
        if (key_begin != 0 && row[s] < rows) {
            const size_t p = size_t(row[s]) * AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2
                + (qhead * AUGUR_QWEN35_FULL_SPLITS_V2 + split) * 258;
            maximum[s] = workspace[p]; sum[s] = workspace[p + 1];
            #pragma unroll
            for (unsigned i = 0; i < 8; ++i) out[s][i] = workspace[p + 2 + lane + i * 32];
        }
    }
    const unsigned begin_tile = max(union_first, key_begin);
    const unsigned end_tile = min(union_last, key_begin + key_count);
    __shared__ float value_tile[8 * 256];
    for (unsigned begin = begin_tile; begin < end_tile; begin += 8) {
        const unsigned count = min(8U, end_tile - begin);
        for (unsigned index = threadIdx.x; index < count * 256; index += blockDim.x)
            value_tile[index] = augur_qwen35_kv8_decode(values, begin + index / 256, head, index % 256, 0);
        __syncthreads();
        #pragma unroll
        for (unsigned s = 0; s < STATES; ++s) {
            if (row[s] < rows) {
                for (unsigned t = 0; t < count; ++t) {
                    const unsigned position = begin + t;
                    if (position < first[s] || position >= last[s]) continue;
                    float next = 0, previous = 0, current = 0;
                    if (lane == 0) {
                        const float scaled = scores[(size_t(row[s]) * 24 + qhead) * score_stride
                            + position - key_begin];
                        next = fmaxf(maximum[s], scaled);
                        previous = sum[s] == 0 ? 0 : expf(maximum[s] - next);
                        current = expf(scaled - next);
                    }
                    next = __shfl_sync(0xffffffff, next, 0);
                    previous = __shfl_sync(0xffffffff, previous, 0);
                    current = __shfl_sync(0xffffffff, current, 0);
                    sum[s] = sum[s] * previous + current;
                    #pragma unroll
                    for (unsigned i = 0; i < 8; ++i)
                        out[s][i] = out[s][i] * previous + current * value_tile[t * 256 + lane + i * 32];
                    maximum[s] = next;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned s = 0; s < STATES; ++s) if (row[s] < rows) {
        const size_t p = size_t(row[s]) * AUGUR_QWEN35_FULL_WORKSPACE_ELEMENTS_V2
            + (qhead * AUGUR_QWEN35_FULL_SPLITS_V2 + split) * 258;
        if (lane == 0) { workspace[p] = maximum[s]; workspace[p + 1] = sum[s]; }
        #pragma unroll
        for (unsigned i = 0; i < 8; ++i) workspace[p + 2 + lane + i * 32] = out[s][i];
    }
}
