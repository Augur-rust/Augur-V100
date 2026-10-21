// Research: original F32 pair expression over losslessly permuted QPN bytes.
// Compiled in the original NVFP4 TU (not the --fmad=false prefill TU).
// QPN group/lane permutation follows augur_qwen35_nvfp4_qpn_layout.cuh.
// No F16 activation staging, tensor-core contraction, requantization or atomics.
namespace qpn_source_pair {
__device__ __forceinline__ uint32_t logical_row(uint32_t tile,uint32_t lane) {
    return tile*32U+((lane>>2U)&3U)*8U+(lane&3U)+((lane&16U)?4U:0U);
}
__device__ __forceinline__ double contract(const uint8_t* weight,const float* input,
    uint32_t rows,uint32_t cols,uint32_t tile,uint32_t lane,uint32_t split,uint32_t splits) {
    const uint32_t groups=cols/16U,quotient=groups/splits;
    const uint32_t begin=split*quotient,end=split+1U==splits?groups:begin+quotient;
    const uint8_t* scales=weight+uint64_t(rows)*cols/2U;
    double acc[8]={};
    for(uint32_t g=begin;g<end;++g) {
        const uint64_t at=(uint64_t(tile)*groups+g)*32U+lane;
        const uint2 codes=__ldg(reinterpret_cast<const uint2*>(weight+at*8U));
        const float scale=ue4m3_scale_to_f32(scales[at]);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const uint32_t ordered=uint32_t(j)/2U+(uint32_t(j)&1U)*4U;
            const float low=e2m1_value((codes.x>>(ordered*4U))&15U);
            const float high=e2m1_value((codes.y>>(ordered*4U))&15U);
            const float pair=low*input[g*16U+j]+high*input[g*16U+8U+j];
            acc[j]+=static_cast<double>(pair)*static_cast<double>(scale);
        }
    }
    // Explicit fixed F64 tree, distinct from the scalar lane partition.
    #pragma unroll
    for(int offset=4;offset>0;offset>>=1)
        for(int j=0;j<offset;++j)acc[j]+=acc[j+offset];
    return acc[0];
}
template<int Splits>
__global__ __launch_bounds__(Splits*32,1) void cta_kernel(const uint8_t* weight,const float* input,
    float* output,uint32_t rows,uint32_t cols,float tensor_scale) {
    __shared__ double partial[Splits][32];
    const uint32_t lane=threadIdx.x&31U,split=threadIdx.x/32U,tile=blockIdx.x;
    partial[split][lane]=contract(weight,input,rows,cols,tile,lane,split,Splits);
    __syncthreads();
    if(split==0) {
        double v=partial[0][lane];
        #pragma unroll
        for(int i=1;i<Splits;++i)v+=partial[i][lane];
        output[logical_row(tile,lane)]=static_cast<float>(v*static_cast<double>(tensor_scale));
    }
}
__global__ __launch_bounds__(128,1) void parts_kernel(const uint8_t* weight,const float* input,
    double* partial,uint32_t rows,uint32_t cols,uint32_t splits) {
    const uint32_t lane=threadIdx.x&31U,tile=blockIdx.x*4U+threadIdx.x/32U;
    const uint32_t row=logical_row(tile,lane);
    if(row<rows)partial[uint64_t(blockIdx.y)*rows+row]=contract(weight,input,rows,cols,tile,lane,blockIdx.y,splits);
}
__global__ void merge_kernel(const double* partial,float* output,uint32_t rows,uint32_t splits,float scale) {
    const uint32_t row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)return;
    double value=partial[row];
    for(uint32_t s=1;s<splits;++s)value+=partial[uint64_t(s)*rows+row];
    output[row]=static_cast<float>(value*static_cast<double>(scale));
}
// Preserve every source lane's group order and its warp reduction tree.
// Four warps read four interleaved group streams, 32 output rows per CTA.
// The layout changes; the F32 pair and F64 expression order do not.
__global__ __launch_bounds__(128,1) void source_exact_kernel(
    const uint8_t* weight,const float* input,float* output,
    uint32_t rows,uint32_t cols,float tensor_scale) {
    __shared__ double parts[4][8][32];
    const uint32_t lane=threadIdx.x&31U,sub=threadIdx.x/32U,tile=blockIdx.x;
    const uint32_t groups=cols/16U;
    const uint8_t* scales=weight+uint64_t(rows)*cols/2U;
    double acc[8]={};
    for(uint32_t g=sub;g<groups;g+=4U) {
        const uint64_t at=(uint64_t(tile)*groups+g)*32U+lane;
        const uint2 codes=__ldg(reinterpret_cast<const uint2*>(weight+at*8U));
        const float scale=ue4m3_scale_to_f32(scales[at]);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const uint32_t ordered=uint32_t(j)/2U+(uint32_t(j)&1U)*4U;
            const float low=e2m1_value((codes.x>>(ordered*4U))&15U);
            const float high=e2m1_value((codes.y>>(ordered*4U))&15U);
            const float pair=low*input[g*16U+j]+high*input[g*16U+8U+j];
            acc[j]+=static_cast<double>(pair)*static_cast<double>(scale);
        }
    }
    #pragma unroll
    for(int j=0;j<8;++j)parts[sub][j][lane]=acc[j];
    __syncthreads();
    if(sub==0U) {
        // Source shuffle offsets 16,8,4,2,1, in that order.
        #pragma unroll
        for(int j=0;j<8;++j)
            acc[j]=(parts[0][j][lane]+parts[2][j][lane])
                  +(parts[1][j][lane]+parts[3][j][lane]);
        #pragma unroll
        for(int offset=4;offset>0;offset>>=1)
            for(int j=0;j<offset;++j)acc[j]+=acc[j+offset];
        output[logical_row(tile,lane)]=static_cast<float>(acc[0]*static_cast<double>(tensor_scale));
    }
}
} // namespace qpn_source_pair
namespace qpn_source_pair {
// Exact E2M1 without an int->float convert: the three magnitude bits placed at
// float bit 22 give m/2 * 2^-126 (denormal for m<2), and the 2^126 rescale is
// exact without FTZ. Values, including -0 for nibble 8, equal e2m1_value.
__device__ __forceinline__ float e2m1_bits(uint32_t word,uint32_t k) {
    const uint32_t b=((word>>(4U*k))&7U)<<22U|((word>>(4U*k+3U))&1U)<<31U;
    return __uint_as_float(b)*0x1p126f;
}
// Same chains and the same final tree as source_exact_kernel: chain
// (row, j, sub) still sums groups sub, sub+4, ... in K order with the same F32
// pair and F64 multiply-add, then the identical shared-memory/source tree.
// Only the issue changes: U groups' codes/scales are loaded before use, the
// input is read as float4, and the eight j chains are split over 8/JS warps.
template<int JS,int U>
__global__ __launch_bounds__(128*(8/JS),1) void source_prefetch_kernel(
    const uint8_t* weight,const float* input,float* output,
    uint32_t rows,uint32_t cols,float tensor_scale) {
    static_assert(JS==4||JS==8,"chains per thread");
    __shared__ double parts[4][8][32];
    const uint32_t lane=threadIdx.x&31U,warp=threadIdx.x/32U,tile=blockIdx.x;
    const uint32_t sub=warp&3U,jb=warp>>2U;
    const uint32_t groups=cols/16U;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const uint2* codes_at=reinterpret_cast<const uint2*>(weight)+base;
    const uint8_t* scales_at=weight+uint64_t(rows)*cols/2U+base;
    const float4* x4=reinterpret_cast<const float4*>(input);
    double acc[JS]={};
    for(uint32_t g0=sub;g0<groups;g0+=4U*U) {
        uint2 codes[U];uint8_t sc[U];
        #pragma unroll
        for(int u=0;u<U;++u) {
            const uint32_t g=g0+4U*u;
            if(g<groups){codes[u]=__ldg(codes_at+uint64_t(g)*32U);sc[u]=__ldg(scales_at+uint64_t(g)*32U);}
        }
        #pragma unroll
        for(int u=0;u<U;++u) {
            const uint32_t g=g0+4U*u;
            if(g>=groups)break;
            const float scale=ue4m3_scale_to_f32(sc[u]);
            float xl[JS],xh[JS];
            #pragma unroll
            for(int q=0;q<JS/4;++q) {
                const float4 lo=__ldg(x4+g*4U+jb*(JS/4)+q),hi=__ldg(x4+g*4U+2U+jb*(JS/4)+q);
                xl[4*q]=lo.x;xl[4*q+1]=lo.y;xl[4*q+2]=lo.z;xl[4*q+3]=lo.w;
                xh[4*q]=hi.x;xh[4*q+1]=hi.y;xh[4*q+2]=hi.z;xh[4*q+3]=hi.w;
            }
            #pragma unroll
            for(int i=0;i<JS;++i) {
                const uint32_t j=jb*JS+uint32_t(i);
                const uint32_t ordered=j/2U+(j&1U)*4U;
                const float low=e2m1_bits(codes[u].x,ordered);
                const float high=e2m1_bits(codes[u].y,ordered);
                const float pair=low*xl[i]+high*xh[i];
                acc[i]+=static_cast<double>(pair)*static_cast<double>(scale);
            }
        }
    }
    #pragma unroll
    for(int i=0;i<JS;++i)parts[sub][jb*JS+i][lane]=acc[i];
    __syncthreads();
    if(warp==0U) {
        double a[8];
        #pragma unroll
        for(int j=0;j<8;++j)
            a[j]=(parts[0][j][lane]+parts[2][j][lane])+(parts[1][j][lane]+parts[3][j][lane]);
        #pragma unroll
        for(int offset=4;offset>0;offset>>=1)
            for(int j=0;j<offset;++j)a[j]+=a[j+offset];
        output[logical_row(tile,lane)]=static_cast<float>(a[0]*static_cast<double>(tensor_scale));
    }
}
} // namespace qpn_source_pair
namespace qpn_source_pair {
// Source F32 pair expression and F64 multiply-add per term, but the F64 sum is
// RE-ASSOCIATED: warp w takes groups w, w+W, ... and each lane keeps even/odd
// j in two chains, then a fixed W-way fold. Each F32 pair times a UE4M3 scale
// is exact in F64, so only F64 rounding of the running sum can differ from
// source order. This is a distinct identity from source_exact/prefetch and
// must be qualified on its own full-state evidence, not inherit theirs.
// Exact E2M1 -> F32 for the eight nibbles of one word. Every E2M1 value is an
// FP16 with a zero low byte, so one PRMT over a two-register byte table yields
// two halves; the sign bits are moved to 15/31 by a second PRMT. half->float
// is exact, so v[k] == e2m1_value(nibble k) bit for bit (including -0).
__device__ __forceinline__ void e2m1x8(uint32_t w,float (&v)[8]) {
    const uint32_t lo=0x3E3C3800U,hi=0x46444240U,w4=w<<4;
    const uint32_t me=w4&0x70707070U,mo=w&0x70707070U;
    uint32_t h02=__byte_perm(lo,hi,me),h46=__byte_perm(lo,hi,me>>16);
    uint32_t h13=__byte_perm(lo,hi,mo),h57=__byte_perm(lo,hi,mo>>16);
    h02|=__byte_perm(w4,0,0x1404)&0x80008000U;h46|=__byte_perm(w4,0,0x3424)&0x80008000U;
    h13|=__byte_perm(w,0,0x1404)&0x80008000U;h57|=__byte_perm(w,0,0x3424)&0x80008000U;
    const __half2 a=*reinterpret_cast<const __half2*>(&h02),b=*reinterpret_cast<const __half2*>(&h46);
    const __half2 c=*reinterpret_cast<const __half2*>(&h13),d=*reinterpret_cast<const __half2*>(&h57);
    v[0]=__low2float(a);v[2]=__high2float(a);v[4]=__low2float(b);v[6]=__high2float(b);
    v[1]=__low2float(c);v[3]=__high2float(c);v[5]=__low2float(d);v[7]=__high2float(d);
}
// Three 16-warp CTAs per SM caps registers at 40 without spilling and keeps
// more loads in flight (bench: about 12% faster than one CTA per SM).
template<int W>
__global__ __launch_bounds__(W*32,W==16?3:1) void source_reassoc_kernel(
    const uint8_t* weight,const float* input,float* output,
    uint32_t rows,uint32_t cols,float tensor_scale) {
    __shared__ double parts[W][32];
    const uint32_t lane=threadIdx.x&31U,warp=threadIdx.x/32U,tile=blockIdx.x;
    const uint32_t groups=cols/16U;
    const uint64_t base=uint64_t(tile)*groups*32U+lane;
    const uint2* codes_at=reinterpret_cast<const uint2*>(weight)+base;
    const uint8_t* scales_at=weight+uint64_t(rows)*cols/2U+base;
    const float4* x4=reinterpret_cast<const float4*>(input);
    double even=0.0,odd=0.0;
    for(uint32_t g=warp;g<groups;g+=uint32_t(W)) {
        const uint2 codes=__ldg(codes_at+uint64_t(g)*32U);
        const double scale=static_cast<double>(ue4m3_scale_to_f32(__ldg(scales_at+uint64_t(g)*32U)));
        const float4 l0=__ldg(x4+g*4U),l1=__ldg(x4+g*4U+1U),h0=__ldg(x4+g*4U+2U),h1=__ldg(x4+g*4U+3U);
        const float xl[8]={l0.x,l0.y,l0.z,l0.w,l1.x,l1.y,l1.z,l1.w};
        const float xh[8]={h0.x,h0.y,h0.z,h0.w,h1.x,h1.y,h1.z,h1.w};
        float low[8],high[8];e2m1x8(codes.x,low);e2m1x8(codes.y,high);
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int ordered=j/2+(j&1)*4;
            const float pair=low[ordered]*xl[j]+high[ordered]*xh[j];
            if(j&1)odd+=static_cast<double>(pair)*scale;else even+=static_cast<double>(pair)*scale;
        }
    }
    parts[warp][lane]=even+odd;
    __syncthreads();
    if(warp==0U) {
        double a=parts[0][lane];
        #pragma unroll
        for(int w=1;w<W;++w)a+=parts[w][lane];
        output[logical_row(tile,lane)]=static_cast<float>(a*static_cast<double>(tensor_scale));
    }
}
} // namespace qpn_source_pair
