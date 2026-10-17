// Included after the existing tiled FP64 pack/score kernels. Reuse their
// bounded row128/key8192 storage; never materialize the whole context matrix.
bool canonical_scores_metadata_valid(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t address, uint64_t bytes, uint32_t policy, bool inspect_device) {
    if (plan == nullptr || plan->handle == nullptr || plan->owner != stream
        || plan->ordinal != ordinal || policy > AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_SCORE_DGEMM_V1
        || augur_cuda_qwen35_canonical_prefill_preflight_v1(ordinal, stream, args).status != AUGUR_CUDA_STATUS_OK)
        return false;
    const Range scratch{address, bytes, alignof(double)};
    if (bytes != dgemm_score_workspace_bytes(args->tokens, args->capacity) || !range_valid(scratch))
        return false;
    const Range views[] = {
        {args->query_f32, args->query_len_bytes, 4}, {args->gate_f32, args->gate_len_bytes, 4},
        {args->key_f32, args->key_len_bytes, 4}, {args->value_f32, args->value_len_bytes, 4},
        {args->key_cache_q8_0, args->key_cache_len_bytes, 2},
        {args->value_cache_q8_0, args->value_cache_len_bytes, 2},
        {args->attention_f32, args->attention_len_bytes, 4}, {args->gated_f32, args->gated_len_bytes, 4},
        {args->workspace_f32, args->workspace_len_bytes, 4},
        {plan->snapshot.decoded_weight_f16, plan->snapshot.decoded_weight_len_bytes, 16},
        {plan->snapshot.decoded_input_f16, plan->snapshot.decoded_input_len_bytes, 16},
    };
    for (const auto view : views)
        if (ranges_overlap(view, scratch) || (inspect_device && !device_range(view, ordinal))) return false;
    return !inspect_device || device_range(scratch, ordinal);
}

augur_cuda_result_v1 submit_canonical_scores(int32_t ordinal, augur_cuda_stream_t stream,
    augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_canonical_prefill_args_v1* args,
    uint64_t address, uint64_t bytes, uint32_t policy) {
    if (!canonical_scores_metadata_valid(ordinal, stream, plan, args, address, bytes, policy, false))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT, 0);
    const auto prepared = augur_cuda_qwen35_canonical_prefill_prepare_v1(ordinal, stream, args);
    if (prepared.status != AUGUR_CUDA_STATUS_OK) return prepared;
    auto* scores = reinterpret_cast<float*>(static_cast<uintptr_t>(address));
    const uint32_t rows = args->tokens, stride = std::min(8192U, args->capacity);
    float* state_padding = scores + uint64_t(rows) * kFullQueryHeads * stride;
    auto* q = reinterpret_cast<double*>(state_padding + uint64_t(rows) * kFullQueryHeads * 2U);
    double* k = q + uint64_t(rows) * 6U * kFullHeadDim;
    double* products = k + uint64_t(stride) * kFullHeadDim;
    const double alpha = 1.0, beta = 0.0;
    const uint32_t positions = args->position_base + rows;
    for (uint32_t begin = 0; begin < positions; begin += stride) {
        const uint32_t count = std::min(stride, positions - begin);
        if (policy == AUGUR_CUDA_QWEN35_CANONICAL_PREFILL_SCORE_DGEMM_V1) {
            for (uint32_t head = 0; head < kFullKvHeads; ++head) {
                prefill_kv8_dgemm_pack_kernel<<<(std::max(rows * 6U, count) * kFullHeadDim + 255U) / 256U,
                    256U, 0, stream->stream>>>(
                    reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
                    reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
                    q, k, 0, rows, head, begin, count, args->capacity);
                auto error = cudaGetLastError();
                if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
                // FP64, not an FP16/TF32 approximation. The reduction tree is
                // nevertheless different and must pass the current oracle.
                const auto status = cublasDgemm(plan->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    count, rows * 6U, kFullHeadDim, &alpha, k, kFullHeadDim,
                    q, kFullHeadDim, &beta, products, stride);
                if (status != CUBLAS_STATUS_SUCCESS) return blas_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, status);
                const uint64_t elements = uint64_t(rows) * 6U * count;
                prefill_kv8_dgemm_store_kernel<<<(elements + 255U) / 256U, 256U, 0, stream->stream>>>(
                    products, scores, rows, head, count, stride);
                error = cudaGetLastError();
                if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
            }
        } else {
            const dim3 grid((count + 31U) / 32U, kFullKvHeads, rows);
            prefill_kv8_tiled_scores_kernel<true><<<grid, 256U, 0, stream->stream>>>(
                reinterpret_cast<const float*>(static_cast<uintptr_t>(args->query_f32)),
                reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->key_cache_q8_0)),
                scores, 0, args->position_base, args->capacity, begin, count, stride);
            const auto error = cudaGetLastError();
            if (error != cudaSuccess) return cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE, error);
        }
        const auto consumed = augur_cuda_qwen35_canonical_prefill_consume_scores_v1(
            ordinal, stream, args, address, begin, count, stride);
        if (consumed.status != AUGUR_CUDA_STATUS_OK) return consumed;
    }
    return augur_cuda_qwen35_canonical_prefill_finish_v1(ordinal, stream, args);
}
