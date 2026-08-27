#include "common.h"

// Fused MoE routing: softmax over n_experts logits -> top-k selection ->
// renormalized weights, one simdgroup per token row, everything in registers.
// Port of the CUDA topk-moe kernel (ggml-cuda/topk-moe.cu): top-k is k rounds
// of simdgroup argmax with -INF knockout — no sort, no threadgroup memory, no
// barriers. Replaces the SOFT_MAX -> ARGSORT -> VIEW -> GET_ROWS -> SUM_ROWS
// -> CLAMP -> DIV chain (6 dispatches, incl. a 512-thread bitonic sort) with
// one dispatch.
//
// Outputs match the unfused graph's two live tensors:
//   ids     : i32, first n_expert_used entries of each argsort-shaped row
//   weights : f32, [n_expert_used, n_rows], renormalized when with_norm
// Ties break toward the lower expert id, matching the CUDA kernel.

#define TOPK_MOE_NSG   4
#define TOPK_MOE_K_MAX 16

template<short NE>
kernel void kernel_topk_moe_f32(
        constant ggml_metal_kargs_topk_moe & args,
        device const char * logits,
        device       char * ids,
        device       char * weights,
        uint  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short EPT = NE/32; // experts per thread (lane)

    const int row = (int) tgpig*TOPK_MOE_NSG + sgitg;
    if (row >= args.n_rows) {
        return;
    }

    device const float * x = (device const float *)(logits + (int64_t) row*args.nb01);

    // load this lane's logits; expert id = slot*32 + lane (CUDA layout);
    // NaN -> -INF so knockout produces unique ids
    float wt[EPT];
    #pragma unroll
    for (short i = 0; i < EPT; ++i) {
        const float v = x[i*32 + tiisg];
        wt[i] = isnan(v) ? -INFINITY : v;
    }

    // softmax over all NE, register-resident
    float mx = wt[0];
    #pragma unroll
    for (short i = 1; i < EPT; ++i) {
        mx = max(mx, wt[i]);
    }
    mx = simd_max(mx);

    float sum = 0.0f;
    #pragma unroll
    for (short i = 0; i < EPT; ++i) {
        wt[i] = exp(wt[i] - mx);
        sum += wt[i];
    }
    sum = simd_sum(sum);

    const float inv_sum = 1.0f/sum;
    #pragma unroll
    for (short i = 0; i < EPT; ++i) {
        wt[i] *= inv_sum;
    }

    // top-k: k rounds of simdgroup argmax + knockout
    const short k = (short) args.n_expert_used;

    float sel_w [TOPK_MOE_K_MAX];
    int   sel_id[TOPK_MOE_K_MAX];

    float wt_sum = 0.0f;

    for (short kk = 0; kk < k; ++kk) {
        float best   = wt[0];
        short best_i = 0;
        #pragma unroll
        for (short i = 1; i < EPT; ++i) {
            if (wt[i] > best) {
                best   = wt[i];
                best_i = i;
            }
        }
        int best_id = best_i*32 + tiisg;

        // 5-step butterfly argmax across the simdgroup; ties -> lower expert id
        #pragma unroll
        for (short off = 16; off > 0; off >>= 1) {
            const float ov = simd_shuffle_xor(best,    (ushort) off);
            const int   oe = simd_shuffle_xor(best_id, (ushort) off);
            if (ov > best || (ov == best && oe < best_id)) {
                best    = ov;
                best_id = oe;
            }
        }

        // owner lane knocks the winner out; unrolled select avoids a dynamic
        // register index
        if ((best_id & 31) == tiisg) {
            const short slot = best_id >> 5;
            #pragma unroll
            for (short i = 0; i < EPT; ++i) {
                wt[i] = (i == slot) ? -INFINITY : wt[i];
            }
        }

        // best/best_id are uniform across the simdgroup after the butterfly
        sel_w [kk] = best;
        sel_id[kk] = best_id;
        wt_sum    += best;
    }

    float inv_den = 1.0f;
    if (args.with_norm) {
        inv_den = 1.0f/max(wt_sum, args.clamp_min); // clamp(x, min, INF) == max
    }

    device int32_t * ids_row = (device int32_t *)(ids     + (int64_t) row*args.nbi1);
    device float   * w_row   = (device float   *)(weights + (int64_t) row*args.nbw1);

    for (short kk = tiisg; kk < k; kk += 32) {
        ids_row[kk] = sel_id[kk];
        w_row  [kk] = sel_w[kk]*inv_den*args.scale;
    }
}

typedef decltype(kernel_topk_moe_f32<32>) kernel_topk_moe_f32_t;

template [[host_name("kernel_topk_moe_f32_ne32")]]  kernel kernel_topk_moe_f32_t kernel_topk_moe_f32<32>;
template [[host_name("kernel_topk_moe_f32_ne64")]]  kernel kernel_topk_moe_f32_t kernel_topk_moe_f32<64>;
template [[host_name("kernel_topk_moe_f32_ne128")]] kernel kernel_topk_moe_f32_t kernel_topk_moe_f32<128>;
template [[host_name("kernel_topk_moe_f32_ne256")]] kernel kernel_topk_moe_f32_t kernel_topk_moe_f32<256>;
template [[host_name("kernel_topk_moe_f32_ne512")]] kernel kernel_topk_moe_f32_t kernel_topk_moe_f32<512>;
