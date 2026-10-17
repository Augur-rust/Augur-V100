// Replay-only scalar reuse. Canonical FP32 K-order and state rounding are unchanged.
__global__ void prefill_linear_fold_scalars_kernel(const float* a, const float* ar, const float* br,
    const float* bias, const float* sa, float* scalars, unsigned rows, float eps) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rows * kLinearValueHeads) return;
    const unsigned t = index / kLinearValueHeads, head = index % kLinearValueHeads;
    const float* k = a + t*kLinearQkvElements + kLinearKeyHeads*kLinearHeadDim
        + (head%kLinearKeyHeads)*kLinearHeadDim;
    float sum = 0;
    for (unsigned c=0; c<kLinearHeadDim; ++c) sum += k[c]*k[c];
    scalars[3*index] = rsqrtf(fmaxf(sum,eps*eps));
    scalars[3*index+1] = expf(prefill_softplus(ar[index]+bias[head])*sa[head]);
    scalars[3*index+2] = prefill_sigmoid(br[index]);
}

// One 128-channel head per CTA. Only the 16 K-head CTAs compute scalars.
// Each K norm is reused by its three value heads; their gate values stay distinct.
__global__ void prefill_linear_conv_fold_scalars_kernel(const float* qkv, const float* weights,
    float* history, float* activated, const float* ar, const float* br,
    const float* bias, const float* sa, float* scalars, unsigned rows, float eps) {
    const unsigned channel = blockIdx.x*kLinearHeadDim+threadIdx.x;
    if (channel >= kLinearQkvElements) return;
    const unsigned hb = channel*3, wb = channel*4;
    float h0=history[hb], h1=history[hb+1], h2=history[hb+2];
    const float w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2], w3=weights[wb+3];
    const bool key_block = blockIdx.x >= kLinearKeyHeads && blockIdx.x < 2*kLinearKeyHeads;
    __shared__ float key[kLinearHeadDim];
    for (unsigned t=0; t<rows; ++t) {
        const unsigned index=t*kLinearQkvElements+channel;
        const float current=qkv[index];
        const float value=prefill_silu(h0*w0+h1*w1+h2*w2+current*w3);
        activated[index]=value;
        h0=h1; h1=h2; h2=current;
        if (key_block) {
            key[threadIdx.x]=value;
            __syncthreads();
            if (threadIdx.x==0) {
                float sum=0;
                for (unsigned c=0; c<kLinearHeadDim; ++c) sum+=key[c]*key[c];
                const float scale=rsqrtf(fmaxf(sum,eps*eps));
                for (unsigned group=0; group<kLinearValueHeads/kLinearKeyHeads; ++group) {
                    const unsigned head=blockIdx.x-kLinearKeyHeads+group*kLinearKeyHeads;
                    const unsigned gate=t*kLinearValueHeads+head;
                    scalars[3*gate]=scale;
                    scalars[3*gate+1]=expf(prefill_softplus(ar[gate]+bias[head])*sa[head]);
                    scalars[3*gate+2]=prefill_sigmoid(br[gate]);
                }
            }
            __syncthreads();
        }
    }
    history[hb]=h0; history[hb+1]=h1; history[hb+2]=h2;
}

__global__ __launch_bounds__(128,2) void prefill_linear_shared_fold_state_kernel(
    const float* a, const float* scalars, float* s, unsigned rows) {
    const unsigned head=blockIdx.x, lane=threadIdx.x, row=blockIdx.y*4+threadIdx.y;
    if (head>=kLinearValueHeads || row>=kLinearHeadDim) return;
    float* dst=s+(head*kLinearHeadDim+row)*kLinearHeadDim;
    float values[4];
    #pragma unroll
    for(unsigned i=0;i<4;++i) values[i]=dst[i*32+lane];
    for(unsigned t=0;t<rows;++t) {
        const float* k=a+t*kLinearQkvElements+kLinearKeyHeads*kLinearHeadDim+(head%kLinearKeyHeads)*kLinearHeadDim;
        const float* v=a+t*kLinearQkvElements+2*kLinearKeyHeads*kLinearHeadDim+head*kLinearHeadDim;
        const float* scalar=scalars+3*(t*kLinearValueHeads+head);
        const float scale=scalar[0], decay=scalar[1], beta=scalar[2];
        float proj=0,keys[4];
        #pragma unroll
        for(unsigned i=0;i<4;++i) { keys[i]=k[i*32+lane]*scale; values[i]*=decay; proj+=values[i]*keys[i]; }
        proj=__shfl_sync(0xffffffff,prefill_warp_sum(proj),0);
        const float delta=(v[row]-proj)*beta;
        #pragma unroll
        for(unsigned i=0;i<4;++i) values[i]+=keys[i]*delta;
    }
    #pragma unroll
    for(unsigned i=0;i<4;++i) dst[i*32+lane]=values[i];
}

// Direct-checkpoint candidate: identical arithmetic, separate immutable initial state.
__global__ void prefill_checkpoint_conv(const float* qkv,
    const float* weights, float* history, const float* source_history, float* activated, uint32_t tokens) {
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kLinearQkvElements) return;
    const uint32_t history_base = channel * (kLinearKernelSize - 1U);
    const uint32_t weight_base = channel * kLinearKernelSize;
    float h0 = source_history[history_base];
    float h1 = source_history[history_base + 1U];
    float h2 = source_history[history_base + 2U];
    const float w0 = weights[weight_base];
    const float w1 = weights[weight_base + 1U];
    const float w2 = weights[weight_base + 2U];
    const float w3 = weights[weight_base + 3U];
    for (uint32_t token = 0; token < tokens; ++token) {
        const uint64_t index = static_cast<uint64_t>(token)
            * kLinearQkvElements + channel;
        const float current = qkv[index];
        activated[index] = prefill_silu(
            h0 * w0 + h1 * w1 + h2 * w2 + current * w3);
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    history[history_base] = h0;
    history[history_base + 1U] = h1;
    history[history_base + 2U] = h2;
}

__global__ void prefill_checkpoint_conv_scalars(const float* qkv, const float* weights,
    float* history, const float* source_history, float* activated, const float* ar, const float* br,
    const float* bias, const float* sa, float* scalars, unsigned rows, float eps) {
    const unsigned channel = blockIdx.x*kLinearHeadDim+threadIdx.x;
    if (channel >= kLinearQkvElements) return;
    const unsigned hb = channel*3, wb = channel*4;
    float h0=source_history[hb], h1=source_history[hb+1], h2=source_history[hb+2];
    const float w0=weights[wb], w1=weights[wb+1], w2=weights[wb+2], w3=weights[wb+3];
    const bool key_block = blockIdx.x >= kLinearKeyHeads && blockIdx.x < 2*kLinearKeyHeads;
    __shared__ float key[kLinearHeadDim];
    for (unsigned t=0; t<rows; ++t) {
        const unsigned index=t*kLinearQkvElements+channel;
        const float current=qkv[index];
        const float value=prefill_silu(h0*w0+h1*w1+h2*w2+current*w3);
        activated[index]=value;
        h0=h1; h1=h2; h2=current;
        if (key_block) {
            key[threadIdx.x]=value;
            __syncthreads();
            if (threadIdx.x==0) {
                float sum=0;
                for (unsigned c=0; c<kLinearHeadDim; ++c) sum+=key[c]*key[c];
                const float scale=rsqrtf(fmaxf(sum,eps*eps));
                for (unsigned group=0; group<kLinearValueHeads/kLinearKeyHeads; ++group) {
                    const unsigned head=blockIdx.x-kLinearKeyHeads+group*kLinearKeyHeads;
                    const unsigned gate=t*kLinearValueHeads+head;
                    scalars[3*gate]=scale;
                    scalars[3*gate+1]=expf(prefill_softplus(ar[gate]+bias[head])*sa[head]);
                    scalars[3*gate+2]=prefill_sigmoid(br[gate]);
                }
            }
            __syncthreads();
        }
    }
    history[hb]=h0; history[hb+1]=h1; history[hb+2]=h2;
}

__global__ __launch_bounds__(128,2) void prefill_checkpoint_state(
    const float* a, const float* scalars, float* s, const float* source_state, unsigned rows) {
    const unsigned head=blockIdx.x, lane=threadIdx.x, row=blockIdx.y*4+threadIdx.y;
    if (head>=kLinearValueHeads || row>=kLinearHeadDim) return;
    float* dst=s+(head*kLinearHeadDim+row)*kLinearHeadDim;
    const float* src=source_state+(head*kLinearHeadDim+row)*kLinearHeadDim;
    float values[4];
    #pragma unroll
    for(unsigned i=0;i<4;++i) values[i]=src[i*32+lane];
    for(unsigned t=0;t<rows;++t) {
        const float* k=a+t*kLinearQkvElements+kLinearKeyHeads*kLinearHeadDim+(head%kLinearKeyHeads)*kLinearHeadDim;
        const float* v=a+t*kLinearQkvElements+2*kLinearKeyHeads*kLinearHeadDim+head*kLinearHeadDim;
        const float* scalar=scalars+3*(t*kLinearValueHeads+head);
        const float scale=scalar[0], decay=scalar[1], beta=scalar[2];
        float proj=0,keys[4];
        #pragma unroll
        for(unsigned i=0;i<4;++i) { keys[i]=k[i*32+lane]*scale; values[i]*=decay; proj+=values[i]*keys[i]; }
        proj=__shfl_sync(0xffffffff,prefill_warp_sum(proj),0);
        const float delta=(v[row]-proj)*beta;
        #pragma unroll
        for(unsigned i=0;i<4;++i) values[i]+=keys[i]*delta;
    }
    #pragma unroll
    for(unsigned i=0;i<4;++i) dst[i*32+lane]=values[i];
}
