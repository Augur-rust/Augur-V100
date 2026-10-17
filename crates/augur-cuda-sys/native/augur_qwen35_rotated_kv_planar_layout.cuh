// Research format2: an exact permutation of each272-byte Q8_0 head.
// No quantization, arithmetic, capacity, or per-position extent change.
#pragma once
#include <cuda_fp16.h>

static __device__ __forceinline__ uint4 augur_planar_q8_decode8(
        const unsigned char* source, int col) {
    const half2 scale=__half2half2(*reinterpret_cast<const half*>(source+256+2*(col/16)));
    const uint2 words=*reinterpret_cast<const uint2*>(source+2*col);
    unsigned decoded[4];
    #pragma unroll
    for(int j=0;j<4;++j){
        const unsigned bits=__byte_perm(j<2?words.x:words.y,0x64646464u,
            (j%2)?0x4342u:0x4140u)^0x00800080u;
        const __half2_raw encoded{static_cast<unsigned short>(bits),static_cast<unsigned short>(bits>>16)};
        const __half2_raw offset{0x6480u,0x6480u};
        const half2 pair=__hmul2(scale,__hsub2_rn(half2(encoded),half2(offset)));
        decoded[j]=unsigned(__half_as_ushort(__low2half(pair)))|
            (unsigned(__half_as_ushort(__high2half(pair)))<<16);
    }
    return make_uint4(decoded[0],decoded[1],decoded[2],decoded[3]);
}
