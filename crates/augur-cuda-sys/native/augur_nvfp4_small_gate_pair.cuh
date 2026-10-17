// Independent launch-schedule experiment for TWO real 48x5120 gates.
// Every row retains the small-gate SplitK16 F32-pair/F64-fold contract.
// Include after the original exact decode helpers. Not padded QPN weights.
template<bool FixedK>
__global__ __launch_bounds__(512, 1)
void nvfp4_sm70_small_gate_pair_concat_kernel(
    const uint8_t* __restrict__ wa, const uint8_t* __restrict__ wb,
    const float* __restrict__ input, float* __restrict__ ya, float* __restrict__ yb,
    uint32_t cols, float scale_a, float scale_b
) {
    __shared__ double partials[16][32];
    const bool second = blockIdx.x >= 48U;
    const uint32_t row = second ? blockIdx.x-48U : blockIdx.x;
    const uint32_t split = threadIdx.x / 32U, lane = threadIdx.x & 31U;
    const uint32_t k = FixedK ? 5120U : cols;
    input += static_cast<uint64_t>(blockIdx.y) * k;
    float* output = (second ? yb : ya) + static_cast<uint64_t>(blockIdx.y)*48U;
    const float tensor_scale = second ? scale_b : scale_a;
    const uint32_t blocks_per_row = k / 64U, quotient = blocks_per_row / 16U;
    const uint32_t begin = split * quotient;
    const uint32_t end = split == 15U ? blocks_per_row : begin + quotient;
    const uint8_t* row_source = (second ? wb : wa) + static_cast<uint64_t>(row)*blocks_per_row*36U;
    double partial = 0.0;
    #pragma unroll
    for (uint32_t block = begin; block < end; ++block) {
        const uint8_t* source = row_source + static_cast<uint64_t>(block)*36U;
        const uint32_t sub_block = lane / 8U, sub_lane = lane & 7U;
        const uint32_t input_base = block*64U + sub_block*16U;
        const uint8_t packed = source[4U+lane];
        float scale = lane < 4U ? ue4m3_scale_to_f32(source[lane]) : 0.0f;
        scale = __shfl_sync(0xffffffffU,scale,sub_block);
        const float low = e2m1_value(packed&15U), high = e2m1_value(packed>>4U);
        const float pair = low*input[input_base+sub_lane] + high*input[input_base+8U+sub_lane];
        partial += static_cast<double>(pair)*static_cast<double>(scale);
    }
    partials[split][lane] = partial;
    __syncthreads();
    if (split != 0) return;
    partial = partials[0][lane];
    #pragma unroll
    for (int s=1;s<16;++s) partial += partials[s][lane];
    #pragma unroll
    for (uint32_t offset=16U;offset!=0;offset>>=1U)
        partial += __shfl_down_sync(0xffffffffU,partial,offset);
    if (lane==0) output[row] = static_cast<float>(partial*static_cast<double>(tensor_scale));
}

template<bool FixedK, bool Prefetch=false>
__global__ __launch_bounds__(512, 1)
void nvfp4_sm70_small_gate_pair_shared_input_kernel(
    const uint8_t* __restrict__ wa, const uint8_t* __restrict__ wb,
    const float* __restrict__ input, float* __restrict__ ya, float* __restrict__ yb,
    uint32_t cols, float scale_a, float scale_b
) {
    static_assert(!Prefetch || FixedK,"five-block prefetch is shape-fixed");
    __shared__ double partials[2][16][32];
    const uint32_t row=blockIdx.x,split=threadIdx.x/32U,lane=threadIdx.x&31U;
    const uint32_t k=FixedK?5120U:cols;
    input += static_cast<uint64_t>(blockIdx.y)*k;
    ya += static_cast<uint64_t>(blockIdx.y)*48U;yb += static_cast<uint64_t>(blockIdx.y)*48U;
    const uint32_t blocks_per_row=k/64U,quotient=blocks_per_row/16U;
    const uint32_t begin=split*quotient;
    wa += static_cast<uint64_t>(row)*blocks_per_row*36U;wb += static_cast<uint64_t>(row)*blocks_per_row*36U;
    const uint32_t sub_block=lane/8U,sub_lane=lane&7U;
    double pa=0.0,pb=0.0;
    if constexpr(Prefetch) {
        uint8_t qa[5],qb[5];float sa[5],sb[5],xlo[5],xhi[5];
        // Spend registers to expose independent packed/input loads before math.
        #pragma unroll
        for(int i=0;i<5;++i) {
            const uint32_t block=begin+i;const uint8_t* a=wa+block*36U;const uint8_t* b=wb+block*36U;
            qa[i]=a[4U+lane];qb[i]=b[4U+lane];
            sa[i]=lane<4U?ue4m3_scale_to_f32(a[lane]):0.0f;
            sb[i]=lane<4U?ue4m3_scale_to_f32(b[lane]):0.0f;
            const uint32_t ix=block*64U+sub_block*16U+sub_lane;
            xlo[i]=input[ix];xhi[i]=input[ix+8U];
        }
        #pragma unroll
        for(int i=0;i<5;++i) {
            const float a=__shfl_sync(0xffffffffU,sa[i],sub_block),b=__shfl_sync(0xffffffffU,sb[i],sub_block);
            const float alo=e2m1_value(qa[i]&15U),ahi=e2m1_value(qa[i]>>4U);
            const float blo=e2m1_value(qb[i]&15U),bhi=e2m1_value(qb[i]>>4U);
            const float pair_a=alo*xlo[i]+ahi*xhi[i],pair_b=blo*xlo[i]+bhi*xhi[i];
            pa += static_cast<double>(pair_a)*static_cast<double>(a);
            pb += static_cast<double>(pair_b)*static_cast<double>(b);
        }
    } else {
        const uint32_t end=split==15U?blocks_per_row:begin+quotient;
        #pragma unroll
        for(uint32_t block=begin;block<end;++block) {
            const uint8_t* a=wa+static_cast<uint64_t>(block)*36U;
            const uint8_t* b=wb+static_cast<uint64_t>(block)*36U;
            const uint8_t qa=a[4U+lane],qb=b[4U+lane];
            float sa=lane<4U?ue4m3_scale_to_f32(a[lane]):0.0f,sb=lane<4U?ue4m3_scale_to_f32(b[lane]):0.0f;
            sa=__shfl_sync(0xffffffffU,sa,sub_block);sb=__shfl_sync(0xffffffffU,sb,sub_block);
            const uint32_t ix=block*64U+sub_block*16U+sub_lane;
            const float xlo=input[ix],xhi=input[ix+8U];
            const float alo=e2m1_value(qa&15U),ahi=e2m1_value(qa>>4U);
            const float blo=e2m1_value(qb&15U),bhi=e2m1_value(qb>>4U);
            const float pair_a=alo*xlo+ahi*xhi,pair_b=blo*xlo+bhi*xhi;
            pa += static_cast<double>(pair_a)*static_cast<double>(sa);
            pb += static_cast<double>(pair_b)*static_cast<double>(sb);
        }
    }
    partials[0][split][lane]=pa;partials[1][split][lane]=pb;
    __syncthreads();if(split!=0)return;
    pa=partials[0][0][lane];pb=partials[1][0][lane];
    #pragma unroll
    for(int s=1;s<16;++s) {pa+=partials[0][s][lane];pb+=partials[1][s][lane];}
    #pragma unroll
    for(uint32_t offset=16U;offset!=0;offset>>=1U) {
        pa+=__shfl_down_sync(0xffffffffU,pa,offset);pb+=__shfl_down_sync(0xffffffffU,pb,offset);
    }
    if(lane==0) {ya[row]=static_cast<float>(pa*static_cast<double>(scale_a));yb[row]=static_cast<float>(pb*static_cast<double>(scale_b));}
}
