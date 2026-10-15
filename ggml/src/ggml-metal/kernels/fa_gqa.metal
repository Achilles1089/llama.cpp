#include "common.h"

// GQA flash-decoding on the simdgroup-matrix units, for few query rows (decode and speculative verify) with F16 K/V.
//
// Measured on an M1 Ultra (Qwen3.8-27B: 24 query heads on 4 KV heads, head 256): the vec kernel's cost is per
// (query row x head x key) - a 7-row verify cost 5.4x a single-row decode at 45k context - and neither its
// several-queries-per-threadgroup mode nor packing the grouped heads into its rows changed that. The half8x8 kernel
// (fa_common.metal) does the math on the matrix units but runs one threadgroup per (8 rows, head) over the WHOLE
// cache, so a decode launches only a handful of threadgroups.
//
// This kernel does the half8x8 kernel's math - Q*K^T and P*V on 8x8 simdgroup matrices, the online softmax - with:
//   - packed rows: a threadgroup's Q rows are the G = n_head/n_head_kv heads of one KV head at each query position
//     (row R = head R%G, position R/G), so one read of a K/V block serves every head of the group;
//   - Q = 8, 16 or 24 rows per threadgroup: every K and V 8x8 tile loaded from device memory is used by all Q/8 row
//     tiles, so a 7-row verify of a 6-head group (42 rows) reads its KV head twice, not six times. The output O stays
//     in registers across the cache (rescaled by a diagonal matrix of the softmax corrections), which keeps the
//     threadgroup memory small enough for 24 rows;
//   - a key split: NWG threadgroups share the cache of a (row tile, KV head), and kernel_flash_attn_ext_vec_reduce
//     combines their partial results (unnormalized O, S, M per row, in the vec kernel's layout).
// The host takes it only without sinks, ALiBi, softcap or a per-head mask, and with n_kv a multiple of C.

constant bool    FC_fa_gqa_has_mask [[function_constant(FC_FLASH_ATTN_EXT_GQA + 0)]];
constant int32_t FC_fa_gqa_ns10     [[function_constant(FC_FLASH_ATTN_EXT_GQA + 1)]];
constant int32_t FC_fa_gqa_ns20     [[function_constant(FC_FLASH_ATTN_EXT_GQA + 2)]];
constant int32_t FC_fa_gqa_nwg      [[function_constant(FC_FLASH_ATTN_EXT_GQA + 3)]];

template<short DK, short DV, short Q, short C, short NSG>
kernel void kernel_flash_attn_ext_gqa(
        constant ggml_metal_kargs_flash_attn_ext & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    typedef simdgroup_half8x8  q8x8_t;
    typedef simdgroup_half8x8  k8x8_t;
    typedef simdgroup_half8x8  v8x8_t;
    typedef simdgroup_float8x8 qk8x8_t;
    typedef simdgroup_float8x8 s8x8_t;
    typedef simdgroup_float8x8 o8x8_t;

    static_assert(Q % 8 == 0 && (Q/NSG)*NSG == Q, "Q rows: 8x8 row tiles, an equal share per simdgroup");
    static_assert((C/8) % NSG == 0 && (DV/8) % NSG == 0 && DK % 16 == 0 && C % 16 == 0, "tile shares");

#define NS10 (FC_fa_gqa_ns10)
#define NS20 (FC_fa_gqa_ns20)
#define NWG  (FC_fa_gqa_nwg)

    const int iwg  = tgpig[2]%NWG;
    const int iq3  = tgpig[2]/NWG;
    const int ikv2 = tgpig[1];
    const int iq1  = tgpig[0]*Q;                      // first packed row of this tile
    const int ikv3 = iq3/(args.ne03/args.ne_12_3);

    const int G  = args.ne02/args.ne_12_2;            // query heads per KV head
    const int NR = G*args.ne01;                       // packed rows of this KV head

    constexpr short RT  = Q/8;          // row tiles
    constexpr short NQ  = Q/NSG;        // softmax rows per simdgroup
    constexpr short KT  = (C/8)/NSG;    // key tiles per simdgroup (Q*K^T)
    constexpr short NO  = (DV/8)/NSG;   // output column tiles per simdgroup (P*V)
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NW  = N_SIMDWIDTH;

    // threadgroup memory: [ sq: Q x DK half | ss: Q x C float | sm: Q x C half ] (reused at the end as so: Q x DV/2
    // float, O leaving in two column halves), then sd: RT diagonal 8x8 float matrices
    threadgroup half   * sq  = (threadgroup half   *) shmem_f16;
    threadgroup half4  * sq4 = (threadgroup half4  *) shmem_f16;
    threadgroup float  * ss  = (threadgroup float  *) (shmem_f16 + Q*DK);
    threadgroup float2 * ss2 = (threadgroup float2 *) (shmem_f16 + Q*DK);
    threadgroup half2  * sm2 = (threadgroup half2  *) (shmem_f16 + Q*DK + 2*Q*C);
    threadgroup float  * so  = (threadgroup float  *) shmem_f16;
    threadgroup float4 * so4 = (threadgroup float4 *) shmem_f16;

    constexpr int SMEM_A = (Q*DK + 2*Q*C + Q*C) > Q*DV ? (Q*DK + 2*Q*C + Q*C) : Q*DV; // in halfs
    threadgroup float  * sd  = (threadgroup float  *) (shmem_f16 + SMEM_A);

    k += ikv2*args.nb12 + ikv3*args.nb13;
    v += ikv2*args.nb22 + ikv3*args.nb23;

    // the mask rows of this simdgroup's softmax rows (rows past the group read position 0 - never stored)
    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j  = jj*NSG + sgitg;
        const int   R  = iq1 + j;
        const int   qp = R < NR ? R/G : 0;

        pm[jj] = (device const half *) (mask + qp*args.nb31 + (iq3%args.ne33)*args.nb33);
    }

    // the packed query rows
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        const int   R = iq1 + j;

        if (R < NR) {
            const int h  = ikv2*G + R%G;
            const int qp = R/G;

            device const float4 * q4 = (device const float4 *) (q + qp*args.nb01 + h*args.nb02 + iq3*args.nb03);

            for (short i = tiisg; i < DK4; i += NW) {
                sq4[j*DK4 + i] = (half4) q4[i];
            }
        } else {
            for (short i = tiisg; i < DK4; i += NW) {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // the diagonal matrices start at zero; the softmax writes only their diagonals
    for (short i = sgitg*NW + tiisg; i < RT*64; i += NSG*NW) {
        sd[i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        S[jj] = 0.0f;
        M[jj] = -FLT_MAX/2;
    }

    o8x8_t lo[RT][NO];
    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
        FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
            lo[rt][ii] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    // loop over this split's blocks of the KV cache
    for (int ic0 = iwg; ; ic0 += NWG) {
        const int ic = ic0*C;
        if (ic >= args.ne11) {
            break;
        }

        if (FC_fa_gqa_has_mask) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                sm2[j*(C/2) + tiisg] = ((device const half2 *) (pm[jj] + ic))[tiisg];
            }
        }

        // S = Q*K^T: each simdgroup takes KT key tiles for every row tile - one K tile load serves all Q rows
        FOR_UNROLL (short kt = 0; kt < KT; ++kt) {
            const short cc = kt*NSG + sgitg;   // key tile

            device const half * pk = (device const half *) (k + (ic + 8*cc)*args.nb11);

            qk8x8_t mqk[RT];
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                mqk[rt] = make_filled_simdgroup_matrix<float, 8>(0.0f);
            }

            #pragma unroll (MIN(DK/16, 4))
            for (short i = 0; i < DK/16; ++i) {
                k8x8_t mk[2];

                simdgroup_barrier(mem_flags::mem_none);

                simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    q8x8_t mq[2];

                    simdgroup_load(mq[0], sq + rt*8*DK + 0*8 + 16*i, DK);
                    simdgroup_load(mq[1], sq + rt*8*DK + 1*8 + 16*i, DK);

                    simdgroup_multiply_accumulate(mqk[rt], mq[0], mk[0], mqk[rt]);
                    simdgroup_multiply_accumulate(mqk[rt], mq[1], mk[1], mqk[rt]);
                }
            }

            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                simdgroup_store(mqk[rt], ss + rt*8*C + 8*cc, C, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: P into ss, the correction of each row onto its diagonal
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            const float m = M[jj];

            float2 s2 = ss2[j*(C/2) + tiisg]*args.scale;

            if (FC_fa_gqa_has_mask) {
                s2 += float2(sm2[j*(C/2) + tiisg]);
            }

            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

            const float  ms  = exp(m  - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);

            S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

            ss2[j*(C/2) + tiisg] = vs2;

            if (tiisg == 0) {
                sd[(j/8)*64 + (j%8)*9] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O + P*V: each simdgroup owns NO column tiles of every row tile - one V tile load serves all rows
        {
            FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                s8x8_t md;
                simdgroup_load(md, sd + rt*64, 8, 0, false);

                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    simdgroup_multiply(lo[rt][ii], md, lo[rt][ii]);
                }
            }

            device const half * pv = (device const half *) (v + ic*args.nb21) + 8*sgitg;

            FOR_UNROLL (short cc = 0; cc < C/16; ++cc) {
                s8x8_t vs[RT][2];

                FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                    simdgroup_load(vs[rt][0], ss + rt*8*C + 16*cc + 0, C, 0, false);
                    simdgroup_load(vs[rt][1], ss + rt*8*C + 16*cc + 8, C, 0, false);
                }

                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    v8x8_t mv[2];

                    simdgroup_load(mv[0], pv + 8*NSG*ii + 0*8*NS20, NS20, 0, false);
                    simdgroup_load(mv[1], pv + 8*NSG*ii + 1*8*NS20, NS20, 0, false);

                    FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
                        simdgroup_multiply_accumulate(lo[rt][ii], vs[rt][0], mv[0], lo[rt][ii]);
                        simdgroup_multiply_accumulate(lo[rt][ii], vs[rt][1], mv[1], lo[rt][ii]);
                    }
                }

                pv += 16*NS20;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O to threadgroup memory (over sq/ss/sm, all done) in two column halves, then the partials in the vec kernel's
    // layout (kernel_flash_attn_ext_vec_reduce combines the splits)
    const int64_t nrows = (int64_t) args.ne3*args.ne2*args.ne1;

    device float4 * dst4 = (device float4 *) dst;
    device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

    static_assert(NO % 2 == 0, "O leaves in two halves of the column tiles");

    FOR_UNROLL (short half_o = 0; half_o < 2; ++half_o) {
        // column tiles ii in [half_o*NO/2, (half_o + 1)*NO/2) hold O columns [half_o*DV/2, (half_o + 1)*DV/2)
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short rt = 0; rt < RT; ++rt) {
            FOR_UNROLL (short i2 = 0; i2 < NO/2; ++i2) {
                simdgroup_store(lo[rt][half_o*(NO/2) + i2], so + rt*8*(DV/2) + 8*(NSG*i2 + sgitg), DV/2, 0, false);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;
            const int   R = iq1 + j;

            if (R < NR) {
                const int h  = ikv2*G + R%G;
                const int qp = R/G;

                const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + h + (int64_t) qp*args.ne1;

                for (short i = tiisg; i < DV4/2; i += NW) {
                    dst4[rid*DV4*NWG + NWG*(half_o*(DV4/2) + i) + iwg] = so4[j*(DV4/2) + i];
                }

                if (half_o == 0 && tiisg == 0) {
                    dst1[rid*(2*NWG) + 2*iwg + 0] = S[jj];
                    dst1[rid*(2*NWG) + 2*iwg + 1] = M[jj];
                }
            }
        }
    }

#undef NS10
#undef NS20
#undef NWG
}

typedef decltype(kernel_flash_attn_ext_gqa<256, 256, 8, 64, 4>) flash_attn_ext_gqa_t;

template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_q8" )]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<256, 256,  8, 64, 4>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_q16")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<256, 256, 16, 64, 8>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256_q24")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<256, 256, 24, 64, 8>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk128_dv128_q8" )]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<128, 128,  8, 64, 4>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk128_dv128_q16")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<128, 128, 16, 64, 8>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk128_dv128_q24")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<128, 128, 24, 64, 8>;
