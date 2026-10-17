// Augur B8 DFlash Q8_0 projection (original kernel, not llama-derived).
// Reads the ORIGINAL 34-byte Q8_0 blocks in place (no repack, no workspace) and
// the FP32 activations directly (no Q8_1 activation quantization). Each warp
// owns RW rows; per 512-column step a lane owns 16 codes of each row. Codes are
// recovered from five aligned 32-bit loads with funnel shifts and converted to
// FP32 exactly with PRMT+FADD. Per 16-code half block the products are summed
// in FP32 and scaled once by the block's FP16 scale: w = d*q exactly as Q8_0
// defines it. KS warps split K; their partials are merged in fixed order.
namespace augur_dflash_q8 {
__device__ __forceinline__ float s8f(uint32_t v,int byte) {
    // (2^23 + (b xor 0x80)) - (2^23 + 128) == signed byte b, exact in FP32.
    const uint32_t u=__byte_perm(v^0x80808080u,0x4B000000u,0x7650u|unsigned(byte));
    return __uint_as_float(u)-8388736.0f;
}
template<int RW,int KS>
__global__ void __launch_bounds__(256) b8_direct(const uint8_t* __restrict__ w,
    const float* __restrict__ x,float* __restrict__ y,unsigned rows,unsigned cols) {
    constexpr int G=8/KS;
    __shared__ float red[KS>1?KS:1][G][RW][8];
    const unsigned warp=threadIdx.x>>5,lane=threadIdx.x&31,g=warp/KS,ks=warp%KS;
    const unsigned row0=(blockIdx.x*G+g)*RW,row_bytes=cols/32*34,steps=cols/512;
    const unsigned blk=lane>>1,h=lane&1,inner=34*blk+2+16*h,shift=(inner&3)*8;
    float acc[RW][8]={};
    for(unsigned st=ks;st<steps;st+=KS) {
        const unsigned k=st*512+lane*16;
        uint32_t c[RW][4];float d[RW];
        #pragma unroll
        for(int r=0;r<RW;++r) {
            const uint8_t* base=w+size_t(row0+r)*row_bytes+st*544;
            const uint32_t* p=reinterpret_cast<const uint32_t*>(base+(inner&~3u));
            uint32_t v[5];
            #pragma unroll
            for(int i=0;i<5;++i)v[i]=__ldg(p+i);
            #pragma unroll
            for(int i=0;i<4;++i)c[r][i]=__funnelshift_r(v[i],v[i+1],shift);
            d[r]=__half2float(__ldg(reinterpret_cast<const __half*>(base+34*blk)));
        }
        float t[RW][8]={};
        #pragma unroll
        for(int q=0;q<4;++q) {
            float4 xv[8];
            #pragma unroll
            for(int j=0;j<8;++j)xv[j]=__ldg(reinterpret_cast<const float4*>(x+size_t(j)*cols+k+q*4));
            #pragma unroll
            for(int r=0;r<RW;++r) {
                const uint32_t v=c[r][q];
                const float w0=s8f(v,0),w1=s8f(v,1),w2=s8f(v,2),w3=s8f(v,3);
                #pragma unroll
                for(int j=0;j<8;++j) {
                    t[r][j]=__fmaf_rn(w0,xv[j].x,t[r][j]);t[r][j]=__fmaf_rn(w1,xv[j].y,t[r][j]);
                    t[r][j]=__fmaf_rn(w2,xv[j].z,t[r][j]);t[r][j]=__fmaf_rn(w3,xv[j].w,t[r][j]);
                }
            }
        }
        #pragma unroll
        for(int r=0;r<RW;++r) {
            #pragma unroll
            for(int j=0;j<8;++j)acc[r][j]=__fmaf_rn(d[r],t[r][j],acc[r][j]);
        }
    }
    #pragma unroll
    for(int r=0;r<RW;++r) {
        #pragma unroll
        for(int j=0;j<8;++j) {
            float v=acc[r][j];
            for(int s=16;s;s>>=1)v=__fadd_rn(v,__shfl_xor_sync(0xffffffff,v,s));
            acc[r][j]=v;
        }
    }
    if constexpr(KS==1) {
        if(lane<8) {
            #pragma unroll
            for(int r=0;r<RW;++r)y[size_t(lane)*rows+row0+r]=acc[r][lane];
        }
    } else {
        if(lane==0) {
            #pragma unroll
            for(int r=0;r<RW;++r) {
                #pragma unroll
                for(int j=0;j<8;++j)red[ks][g][r][j]=acc[r][j];
            }
        }
        __syncthreads();
        for(unsigned e=threadIdx.x;e<unsigned(G*RW*8);e+=256) {
            const unsigned gg=e/(RW*8),r=(e/8)%RW,j=e%8;
            float v=0.0f;
            #pragma unroll
            for(int s=0;s<KS;++s)v=__fadd_rn(v,red[s][gg][r][j]);
            y[size_t(j)*rows+(blockIdx.x*G+gg)*RW+r]=v;
        }
    }
}
// Exact tiling: every admitted draft shape has rows%64==0 and cols%512==0, so no
// thread reads past its rows and no row guard is needed (checked by admission).
inline bool b8_direct_shape(unsigned rows,unsigned cols) {return rows%64U==0U && cols%512U==0U;}
// KS=4 for short matrices (more CTAs to fill 80 SMs) and the long-K down projection.
inline cudaError_t launch(const uint8_t* w,const float* x,float* y,unsigned rows,unsigned cols,cudaStream_t s) {
    if(!b8_direct_shape(rows,cols))return cudaErrorInvalidValue;
    if(rows<=1024U || cols>=17408U) b8_direct<8,4><<<rows/16,256,0,s>>>(w,x,y,rows,cols);
    else b8_direct<8,2><<<rows/32,256,0,s>>>(w,x,y,rows,cols);
    return cudaGetLastError();
}
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_b8_direct_preflight_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    return args && args->tokens==8U && augur_dflash_q8::b8_direct_shape(args->rows,args->cols)
        && args->input_f32%16U==0U
        && llama_projection_metadata_valid(ordinal,stream,plan,args,true,&footprint,true,true)
        ? result(AUGUR_CUDA_STATUS_OK,0) : result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
}
extern "C" augur_cuda_result_v1 AUGUR_CUDA_CALL augur_cuda_dflash_q8_b8_direct_submit_v1(
    int32_t ordinal, augur_cuda_stream_t stream, augur_cuda_qwen35_prefill_cublas_plan_v1* plan,
    const augur_cuda_qwen35_prefill_f32_projection_args_v1* args) {
    Footprint footprint{};
    if(!args || args->tokens!=8U || !augur_dflash_q8::b8_direct_shape(args->rows,args->cols)
        || args->input_f32%16U!=0U
        || !llama_projection_metadata_valid(ordinal,stream,plan,args,false,&footprint,true,true))
        return result(AUGUR_CUDA_STATUS_INVALID_ARGUMENT,0);
    const cudaError_t selected=cudaSetDevice(ordinal);
    if(selected!=cudaSuccess)return cuda_result(AUGUR_CUDA_STATUS_RUNTIME_FAILURE,selected);
    const cudaError_t e=augur_dflash_q8::launch(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(args->weight)),
        reinterpret_cast<const float*>(static_cast<uintptr_t>(args->input_f32)),
        reinterpret_cast<float*>(static_cast<uintptr_t>(args->output_f32)),
        args->rows,args->cols,stream->stream);
    return e==cudaSuccess?result(AUGUR_CUDA_STATUS_OK,0):cuda_result(AUGUR_CUDA_STATUS_SUBMISSION_FAILURE,e);
}
