#include "common.h"

// GQA flash-decoding on the simdgroup-matrix units, for few query rows (decode and speculative verify) with F16 K/V.
//
// Measured on an M1 Ultra (Qwen3.8-27B: 24 query heads on 4 KV heads, head 256): the vec kernel's cost is per
// (query row x head x key) - a 7-row verify cost 5.4x a single-row decode at 45k context - and neither its
// several-queries-per-threadgroup mode nor packing the grouped heads into its rows changed that. The half8x8 kernel
// (fa_common.metal) does the math on the matrix units but runs one threadgroup per (8 rows, head) over the WHOLE
// cache, so a decode launches only a handful of threadgroups.
//
// This is that kernel's math - Q*K^T and P*V on 8x8 simdgroup matrices, the online softmax - unchanged, with:
//   - packed rows: a threadgroup's Q rows are the G = n_head/n_head_kv heads of one KV head at each query position
//     (row R = head R%G, position R/G), so one read of a K/V block serves every head of the group;
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
    // the F16 types of fa_f16.metal (FA_TYPES)
    typedef half               q_t;
    typedef half4              q4_t;
    typedef simdgroup_half8x8  q8x8_t;
    typedef half               k_t;
    typedef simdgroup_half8x8  k8x8_t;
    typedef half               v_t;
    typedef simdgroup_half8x8  v8x8_t;
    typedef float              qk_t;
    typedef simdgroup_float8x8 qk8x8_t;
    typedef float              s_t;
    typedef float2             s2_t;
    typedef simdgroup_float8x8 s8x8_t;
    typedef float              o_t;
    typedef float4             o4_t;
    typedef simdgroup_float8x8 o8x8_t;

    static_assert(DK % 16 == 0 && DV > 64, "head sizes of the packed kernel");

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

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DV4  = DV/4;

    constexpr short PV   = PAD2(DV, 64);
    constexpr short PV4  = PV/4;
    constexpr short PV8  = PV/8;

    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NQ  = Q/NSG;
    constexpr short SH  = 2*C; // shared memory per simdgroup (s_t == float)

    constexpr short TS = 2*SH;
    constexpr short T  = DK + 2*PV; // shared memory size per query in (half)

    threadgroup q_t  * sq  = (threadgroup q_t  *) (shmem_f16 + 0*T); // holds the query data
    threadgroup q4_t * sq4 = (threadgroup q4_t *) (shmem_f16 + 0*T); // same as above but in q4_t
    threadgroup o_t  * so  = (threadgroup o_t  *) (shmem_f16 + 0*T + Q*DK); // the result for all queries in 8x8 matrices (the O matrix from the paper)
    threadgroup o4_t * so4 = (threadgroup o4_t *) (shmem_f16 + 0*T + Q*DK);
    threadgroup s_t  * ss  = (threadgroup s_t  *) (shmem_f16 + Q*T); // scratch buffer for attention, mask and diagonal matrix
    threadgroup s2_t * ss2 = (threadgroup s2_t *) (shmem_f16 + Q*T); // same as above but in s2_t

    // mask storage in shared mem
    threadgroup half2 * sm2 = (threadgroup half2 *) (shmem_f16 + Q*T + 2*C);

    k += ikv2*args.nb12 + ikv3*args.nb13;
    v += ikv2*args.nb22 + ikv3*args.nb23;

    // per-row mask rows (rows past the group read position 0 - their results are never stored)
    device const half * pm[NQ];

    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j  = jj*NSG + sgitg;
        const int   R  = iq1 + j;
        const int   qp = R < NR ? R/G : 0;

        pm[jj] = (device const half *) (mask + qp*args.nb31 + (iq3%args.ne33)*args.nb33);
    }

    // load the packed query rows to shared memory
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        const int   R = iq1 + j;

        if (R < NR) {
            const int h  = ikv2*G + R%G;
            const int qp = R/G;

            device const float4 * q4 = (device const float4 *) (q + qp*args.nb01 + h*args.nb02 + iq3*args.nb03);

            for (short i = tiisg; i < DK4; i += NW) {
                sq4[j*DK4 + i] = (q4_t) q4[i];
            }
        } else {
            for (short i = tiisg; i < DK4; i += NW) {
                sq4[j*DK4 + i] = 0;
            }
        }
    }

    // zero out
    FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;

        for (short i = tiisg; i < DV4; i += NW) {
            so4[j*PV4 + i] = 0;
        }

        for (short i = tiisg; i < SH; i += NW) {
            ss[j*SH + i] = 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ] = { [0 ... NQ-1] = 0.0f };
    float M[NQ] = { [0 ... NQ-1] = -FLT_MAX/2 };

    // loop over this split's blocks of the KV cache
    for (int ic0 = iwg; ; ic0 += NWG) {
        const int ic = ic0*C;
        if (ic >= args.ne11) {
            break;
        }

        // read the mask into shared mem
        if (FC_fa_gqa_has_mask) {
            FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
                const short j = jj*NSG + sgitg;

                sm2[j*SH + tiisg] = ((device const half2 *) (pm[jj] + ic))[tiisg];
            }
        }

        // Q*K^T, read directly from global memory
        {
            device      const k_t * pk = (device const k_t *) (k + ic*args.nb11);
            threadgroup const q_t * pq = sq;
            threadgroup       s_t * ps = ss;

            pk += sgitg*(8*NS10);
            ps += sgitg*(8*1);

            static_assert((C/8) % NSG == 0, "");

            constexpr short NC = (C/8)/NSG;

            FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>((qk_t) 0.0f);

                k8x8_t mk[2];
                q8x8_t mq[2];

                // note: too much unroll can tank the performance for large heads
                #pragma unroll (MIN(DK8/2, 4*NSG))
                for (short i = 0; i < DK8/2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
                    simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);

                    simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
                    simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);

                    simdgroup_barrier(mem_flags::mem_none);

                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }

                simdgroup_store(mqk, ps, SH, 0, false);

                pk += 8*(NSG*NS10);
                ps += 8*(NSG);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
            const short j = jj*NSG + sgitg;

            const float m = M[jj];

            // scale and apply the mask
            float2 s2 = ss2[j*SH/2 + tiisg]*args.scale;

            if (FC_fa_gqa_has_mask) {
                s2 += s2_t(sm2[j*SH + tiisg]);
            }

            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));

            const float  ms  = exp(m  - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);

            S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);

            // the P matrix from the paper (Q rows, C columns)
            ss2[j*SH/2 + tiisg] = vs2;

            if (DV4 % NW == 0) {
                FOR_UNROLL (short ii = 0; ii < DV4/NW; ++ii) {
                    const short i = ii*NW + tiisg;

                    so4[j*PV4 + i] *= ms;
                }
            } else {
                for (short i = tiisg; i < DV4; i += NW) {
                    so4[j*PV4 + i] *= ms;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V, read directly from global memory
        {
            static_assert(PV8 % NSG == 0, "");

            constexpr short NO = PV8/NSG;

            o8x8_t lo[NO];

            {
                auto sot = so + 8*sgitg;

                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);

                    sot += 8*NSG;
                }
            }

            {
                device const v_t * pv = (device const v_t *) (v + ic*args.nb21);

                pv += 8*sgitg;

                constexpr short NC = (C/8)/2;

                FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
                    s8x8_t vs[2];

                    simdgroup_load(vs[0], ss + 16*cc + 0, SH, 0, false);
                    simdgroup_load(vs[1], ss + 16*cc + 8, SH, 0, false);

                    FOR_UNROLL (short ii = 0; ii < NO/2; ++ii) {
                        v8x8_t mv[4];

                        simdgroup_load(mv[0], pv + 0*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[1], pv + 8*NSG + 16*ii*NSG + 0*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[2], pv + 0*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);
                        simdgroup_load(mv[3], pv + 8*NSG + 16*ii*NSG + 1*8*NS20, NS20, 0, false);

                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[0], mv[0], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[0], mv[1], lo[2*ii + 1]);
                        simdgroup_multiply_accumulate(lo[2*ii + 0], vs[1], mv[2], lo[2*ii + 0]);
                        simdgroup_multiply_accumulate(lo[2*ii + 1], vs[1], mv[3], lo[2*ii + 1]);
                    }

                    pv  += 2*8*NS20;
                }
            }

            {
                auto sot = so + 8*sgitg;

                FOR_UNROLL (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);

                    sot += 8*NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // the partials, in the vec kernel's layout (kernel_flash_attn_ext_vec_reduce combines the splits)
    const int64_t nrows = (int64_t) args.ne3*args.ne2*args.ne1;

    device float4 * dst4 = (device float4 *) dst;
    device float  * dst1 = (device float  *) dst + nrows*DV*NWG;

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj*NSG + sgitg;
        const int   R = iq1 + j;
        if (R >= NR) {
            break;
        }

        const int h  = ikv2*G + R%G;
        const int qp = R/G;

        const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + h + (int64_t) qp*args.ne1;

        for (short i = tiisg; i < DV4; i += NW) {
            dst4[rid*DV4*NWG + NWG*i + iwg] = (float4) so4[j*PV4 + i];
        }

        if (tiisg == 0) {
            dst1[rid*(2*NWG) + 2*iwg + 0] = S[jj];
            dst1[rid*(2*NWG) + 2*iwg + 1] = M[jj];
        }
    }

#undef NS10
#undef NS20
#undef NWG
}

typedef decltype(kernel_flash_attn_ext_gqa<256, 256, 8, 64, 4>) flash_attn_ext_gqa_t;

template [[host_name("kernel_flash_attn_ext_gqa_f16_dk256_dv256")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<256, 256, 8, 64, 4>;
template [[host_name("kernel_flash_attn_ext_gqa_f16_dk128_dv128")]] kernel flash_attn_ext_gqa_t kernel_flash_attn_ext_gqa<128, 128, 8, 64, 4>;
