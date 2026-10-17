// SPDX-License-Identifier: Apache-2.0
// Time tiles read already-materialized QKV, not another tile's updated history.
// Only tile zero reads/writes history, after saving its three initial samples.
// Thus there is no cross-CTA history race and no extra copy/workspace/launch.
template<unsigned Tile> __global__ void augur_prefill_conv_time_tiles(
    const float* qkv, const float* weights, float* history, float* activated, uint32_t tokens) {
    static_assert(Tile >= 3, "history is confined to tile zero");
    const uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= kLinearQkvElements) return;
    const uint32_t first = blockIdx.y * Tile;
    if (first >= tokens) return;
    const uint32_t hb = channel * 3U, wb = channel * 4U;
    float h0 = first == 0 ? history[hb] : qkv[uint64_t(first-3U)*kLinearQkvElements+channel];
    float h1 = first == 0 ? history[hb+1] : qkv[uint64_t(first-2U)*kLinearQkvElements+channel];
    float h2 = first == 0 ? history[hb+2] : qkv[uint64_t(first-1U)*kLinearQkvElements+channel];
    const float w0=weights[wb],w1=weights[wb+1],w2=weights[wb+2],w3=weights[wb+3];
    const uint32_t last = min(first+Tile,tokens);
    for (uint32_t token=first;token<last;++token) {
        const uint64_t index=uint64_t(token)*kLinearQkvElements+channel;
        const float current=qkv[index];
        activated[index]=prefill_silu(h0*w0+h1*w1+h2*w2+current*w3);
        h0=h1;h1=h2;h2=current;
    }
    if (first == 0) {
        if (tokens >= 3) {
            history[hb]=qkv[uint64_t(tokens-3U)*kLinearQkvElements+channel];
            history[hb+1]=qkv[uint64_t(tokens-2U)*kLinearQkvElements+channel];
            history[hb+2]=qkv[uint64_t(tokens-1U)*kLinearQkvElements+channel];
        } else { history[hb]=h0;history[hb+1]=h1;history[hb+2]=h2; }
    }
}
