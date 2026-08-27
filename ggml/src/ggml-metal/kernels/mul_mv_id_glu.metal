#include "common.h"
#include "dequantize.h"

// Fused MoE expert FFN front half at decode (n_tokens == 1):
//   gate = mul_mat_id(gate_exps, x, ids)
//   up   = mul_mat_id(up_exps,   x, ids)
//   out  = swiglu_split(gate, up)        // silu(gate) * up
// collapsed into ONE dispatch. Metal port of the CUDA mmvq has_fusion path
// (ggml-cuda/mmvq.cu): both dots accumulate against the same activation
// registers, the GLU is applied in-register at the write. Replaces two
// mul_mv_id dispatches + one GLU dispatch (and their barriers) per MoE layer.
//
// Layout mirrors kernel_mul_mv_q4_K_f32_impl: one expert slot per tgpig.z,
// NR0 output rows per simdgroup, 32 lanes split the k dimension, simd_sum
// epilogue, no threadgroup memory, no barriers.
//
// The silu formula matches kernel_swiglu (unary.metal) exactly for parity:
//   silu = x0 / (1.0f + exp(-x0)); dst = silu * x1

#define MMID_GLU_NSG 4
#define MMID_GLU_NR0 1

kernel void kernel_mul_mv_id_q4_K_swiglu_f32(
        constant ggml_metal_kargs_mul_mv_id_glu & args,
        device const char * src_gate,
        device const char * src_up,
        device const char * src1,
        device const char * ids,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = MMID_GLU_NSG;
    constexpr short nr0 = MMID_GLU_NR0;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const int islot = tgpig.z; // expert slot 0..n_expert_used-1
    const int itok  = tgpig.y; // token 0..ne21-1

    const int32_t i02 = ((device const int32_t *) (ids + (int64_t)itok*args.nbi1))[islot];

    const int first_row = (tgpig.x * NSG + sgitg) * nr0;

    device float * dst_f32 = (device float *) dst + ((int64_t)itok*args.nei0 + islot)*args.ne01;

    if (i02 < 0 || i02 >= args.ne02) {
        // masked/invalid slot: keep the output defined
        for (int row = 0; row < nr0 && first_row + row < args.ne01; ++row) {
            if (tiisg == 0) {
                dst_f32[first_row + row] = 0.0f;
            }
        }
        return;
    }

    const short ix = tiisg/8;  // 0...3
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const uint64_t offs_w = (uint64_t)first_row*args.nb01 + (uint64_t)i02*args.nb02;

    device const block_q4_K * xg = (device const block_q4_K *) (src_gate + offs_w);
    device const block_q4_K * xu = (device const block_q4_K *) (src_up   + offs_w);
    device const float      * y  = (device const float      *) (src1 + (int64_t)itok*args.nb12);

    float yl[16];
    float yh[16];

    float sumf_g[nr0] = {0.f};
    float sumf_u[nr0] = {0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // gate matrix rows
        {
            device const uint16_t * sc = (device const uint16_t *)xg[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)xg[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &xg[ib].d;

            for (short row = 0; row < nr0; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                }

                sumf_g[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                        (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                               dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.nb01/2;
                sc += args.nb01/2;
                dh += args.nb01/2;
            }
        }

        // up matrix rows (same yl/yh/sumy registers)
        {
            device const uint16_t * sc = (device const uint16_t *)xu[ib].scales + iq;
            device const uint16_t * q1 = (device const uint16_t *)xu[ib].qs + 16 * iq + 4 * ir;
            device const half     * dh = &xu[ib].d;

            for (short row = 0; row < nr0; row++) {
                sc16[0] = sc[0] & kmask1;
                sc16[1] = sc[2] & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                device const uint16_t * q2 = q1 + 32;

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
                }

                sumf_u[row] += dh[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                        (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                               dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.nb01/2;
                sc += args.nb01/2;
                dh += args.nb01/2;
            }
        }

        y4 += 4 * QK_K;
    }

    for (int row = 0; row < nr0 && first_row + row < args.ne01; ++row) {
        const float g = simd_sum(sumf_g[row]);
        const float u = simd_sum(sumf_u[row]);
        if (tiisg == 0) {
            const float silu = g / (1.0f + exp(-g));
            dst_f32[first_row + row] = silu*u;
        }
    }
}

// Fused MoE expert FFN back half at decode (n_tokens == 1):
//   experts = mul_mat_id(down_exps, act, ids)      // [ne01, n_used]
//   experts = mul(experts, weights)                // route weights [1, n_used]
//   out     = sum over the n_used expert slices    // view + add chain
// collapsed into ONE dispatch: each simdgroup owns NR0 output rows and loops
// the expert slots in-kernel, accumulating w[e] * dot(down[ids[e]][row], act[:,e])
// in registers. The [ne01, n_used] intermediate never touches device memory.
// Left-to-right slot order matches the unfused add chain.

// q5_1 half-block dot (copy of mul_mv.metal's block_q_n_dot_y for q5_1 — that
// helper is file-local there). yl carries the pre-scaled shift factors.
static inline float mmid_reduce_q5_1_dot(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = *((device const uint32_t *)qb_curr->qh);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

kernel void kernel_mul_mv_id_q5_1_reduce_f32(
        constant ggml_metal_kargs_mul_mv_id_reduce & args,
        device const char * src0,
        device const char * act,
        device const char * ids,
        device const char * wts,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = MMID_GLU_NSG;
    constexpr short NR0 = MMID_GLU_NR0;
    constexpr short NQ  = 16;

    const int r0   = (tgpig.x*NSG + sgitg)*NR0;
    const int itok = tgpig.y; // token 0..ne21-1

    const int nb = args.ne00/QK5_1;

    const short ix = tiisg/2;     // 16 blocks in flight
    const short il = (tiisg%2)*8; // half-block offset

    device const int32_t * ids_i32 = (device const int32_t *) (ids + (int64_t)itok*args.nbi1);
    device const float   * wts_f32 = (device const float   *) wts + (int64_t)itok*args.n_used;

    float sumf[NR0] = {0.f};

    float yl[16];

    for (int e = 0; e < args.n_used; ++e) {
        const int32_t i02 = ids_i32[e];
        if (i02 < 0 || i02 >= args.ne02) {
            continue;
        }

        const float w = wts_f32[e];

        device const block_q5_1 * ax[NR0];
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            ax[row] = (device const block_q5_1 *) (src0 + (uint64_t)(r0 + row)*args.nb01 + (uint64_t)i02*args.nb02);
        }

        device const float * y  = (device const float *) act + ((int64_t)itok*args.n_used + e)*args.ne00;
        device const float * yb = y + ix*QK5_1 + il;

        float acc[NR0] = {0.f};

        for (int ib = ix; ib < nb; ib += NQ) {
            float sumy[2] = { 0.f, 0.f };

            FOR_UNROLL (short i = 0; i < 8; i += 2) {
                sumy[0]  += yb[i +  0] + yb[i +  1];
                yl[i + 0] = yb[i +  0];
                yl[i + 1] = yb[i +  1]/256.f;

                sumy[1]  += yb[i + 16] + yb[i + 17];
                yl[i + 8] = yb[i + 16]/16.f;
                yl[i + 9] = yb[i + 17]/4096.f;
            }

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                acc[row] += mmid_reduce_q5_1_dot(ax[row] + ib, sumy[0] + sumy[1], yl, il);
            }

            yb += QK5_1 * NQ;
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            sumf[row] += w*acc[row];
        }
    }

    device float * dst_f32 = (device float *) dst + (int64_t)itok*args.ne01;

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}
