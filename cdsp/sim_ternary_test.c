// Verification and benchmark of Interleaved-Group Ternary Q2 HVX kernel on hexagon-sim
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline unsigned f16_scale_to_f32_bits_full(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x38000000u; // full s (for ternary {-s, 0, +s})
    return (h < 0x0400) ? 0 : f;
}

static inline float f16r(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    unsigned f = (e == 0) ? (s << 31) : (e == 31) ? ((s << 31) | 0x7f800000u | (m << 13))
                                                  : ((s << 31) | ((e + 112) << 23) | (m << 13));
    float r; memcpy(&r, &f, 4); return r;
}

static inline float hvx_reduce_sum32(HVX_Vector v) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
    float r; memcpy(&r, &v, 4); return r;
}

#define HVX_MAX_K 17408
static float hvx_xalign[HVX_MAX_K] __attribute__((aligned(128)));

static const uint32_t hvx_bitmask_arr[32] __attribute__((aligned(128))) = {
    1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
    1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
    1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
    1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
};

// Interleaved-Group Ternary Q2 kernel:
// Each group g of 128 weights has 8 uint32_t words (32 bytes):
//   gb[0..3] = sgn_bits (1 -> +s, 0 -> -s)
//   gb[4..7] = nz_bits  (1 -> non-zero, 0 -> 0)
__attribute__((noinline))
void kernel_ternary_q2_interleaved(int out_dim, int in_dim, int prow,
                                   const uint32_t* bits_u32,
                                   const unsigned short* scales, float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    HVX_Vector vMask = *(const HVX_Vector*)hvx_bitmask_arr;
    HVX_Vector vZero = Q6_V_vzero();

    for (int i = 0; i < out_dim; i += 4) {
        const unsigned short* sg0 = scales + (unsigned)(i + 0) * (unsigned)ng;
        const unsigned short* sg1 = scales + (unsigned)(i + 1) * (unsigned)ng;
        const unsigned short* sg2 = scales + (unsigned)(i + 2) * (unsigned)ng;
        const unsigned short* sg3 = scales + (unsigned)(i + 3) * (unsigned)ng;

        const uint32_t* bp0 = bits_u32 + (unsigned)(i + 0) * (unsigned)prow_u32;
        const uint32_t* bp1 = bits_u32 + (unsigned)(i + 1) * (unsigned)prow_u32;
        const uint32_t* bp2 = bits_u32 + (unsigned)(i + 2) * (unsigned)prow_u32;
        const uint32_t* bp3 = bits_u32 + (unsigned)(i + 3) * (unsigned)prow_u32;

        HVX_Vector vAcc0 = Q6_V_vzero(), vAcc1 = Q6_V_vzero();
        HVX_Vector vAcc2 = Q6_V_vzero(), vAcc3 = Q6_V_vzero();

        for (int g = 0; g < ng; g++) {
            const uint32_t* gb0 = bp0 + (unsigned)g * 8u;
            const uint32_t* gb1 = bp1 + (unsigned)g * 8u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 8u;
            const uint32_t* gb3 = bp3 + (unsigned)g * 8u;

            const float* gx = hvx_xalign + (unsigned)g * 128u;

            HVX_Vector vGAcc0 = Q6_V_vzero(), vGAcc1 = Q6_V_vzero();
            HVX_Vector vGAcc2 = Q6_V_vzero(), vGAcc3 = Q6_V_vzero();

            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);
                HVX_Vector vNegX = Q6_Vsf_vfneg_Vsf(vX);

                HVX_Vector vSign0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c]), vMask), vMask), vX, vNegX);
                HVX_Vector vSign1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c]), vMask), vMask), vX, vNegX);
                HVX_Vector vSign2 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb2[c]), vMask), vMask), vX, vNegX);
                HVX_Vector vSign3 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb3[c]), vMask), vMask), vX, vNegX);

                HVX_Vector vTerm0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c + 4]), vMask), vMask), vSign0, vZero);
                HVX_Vector vTerm1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c + 4]), vMask), vMask), vSign1, vZero);
                HVX_Vector vTerm2 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb2[c + 4]), vMask), vMask), vSign2, vZero);
                HVX_Vector vTerm3 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb3[c + 4]), vMask), vMask), vSign3, vZero);

                vGAcc0 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc0, vTerm0);
                vGAcc1 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc1, vTerm1);
                vGAcc2 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc2, vTerm2);
                vGAcc3 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc3, vTerm3);
            }

            unsigned sbits0 = f16_scale_to_f32_bits_full(sg0[g]);
            unsigned sbits1 = f16_scale_to_f32_bits_full(sg1[g]);
            unsigned sbits2 = f16_scale_to_f32_bits_full(sg2[g]);
            unsigned sbits3 = f16_scale_to_f32_bits_full(sg3[g]);

            vAcc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc0, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc0), Q6_V_vsplat_R((int)sbits0)));
            vAcc1 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc1, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc1), Q6_V_vsplat_R((int)sbits1)));
            vAcc2 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc2, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc2), Q6_V_vsplat_R((int)sbits2)));
            vAcc3 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc3, Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(vGAcc3), Q6_V_vsplat_R((int)sbits3)));
        }
        y[i + 0] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc0));
        y[i + 1] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc1));
        y[i + 2] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc2));
        y[i + 3] = hvx_reduce_sum32(Q6_Vsf_equals_Vqf32(vAcc3));
    }
}

static unsigned rng_state = 0x12345678u;
static unsigned rnd(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

#define ROWS 64
#define COLS 5120
static uint32_t q2bits[ROWS * (COLS / 16)] __attribute__((aligned(128)));
static unsigned short ssc[ROWS * (COLS / 128)] __attribute__((aligned(128)));
static float y_hvx[ROWS], y_ref[ROWS];

int main(void) {
    for (int j = 0; j < COLS; j++) hvx_xalign[j] = ((float)(rnd() % 2000) / 1000.0f - 1.0f);
    int ng = COLS / 128;
    for (int i = 0; i < ROWS; i++) {
        for (int g = 0; g < ng; g++) {
            ssc[i * ng + g] = (unsigned short)(0x3800 + (rnd() % 0x800));
            for (int w = 0; w < 8; w++) {
                q2bits[(i * ng + g) * 8 + w] = (rnd() << 16) ^ rnd();
            }
        }
    }
    int prow = 2 * (COLS / 8);
    for (int i = 0; i < ROWS; i++) {
        double acc = 0;
        for (int g = 0; g < ng; g++) {
            double s = (double)f16r(ssc[i * ng + g]);
            const uint32_t* gb = &q2bits[(i * ng + g) * 8];
            for (int c = 0; c < 4; c++) {
                uint32_t sgn_w = gb[c];
                uint32_t nz_w = gb[c + 4];
                for (int b = 0; b < 32; b++) {
                    int sgn = (sgn_w >> b) & 1;
                    int nz = (nz_w >> b) & 1;
                    double w = nz ? (sgn ? s : -s) : 0.0;
                    acc += w * (double)hvx_xalign[g * 128 + c * 32 + b];
                }
            }
        }
        y_ref[i] = (float)acc;
    }

    kernel_ternary_q2_interleaved(ROWS, COLS, prow, q2bits, ssc, y_hvx);

    double max_err = 0;
    int fails = 0;
    for (int i = 0; i < ROWS; i++) {
        double e = y_hvx[i] - y_ref[i]; if (e < 0) e = -e;
        if (e > max_err) max_err = e;
        if (e > 0.01) fails++;
    }
    printf("Interleaved Ternary Q2 on Hexagon v79: ROWS=%d COLS=%d fails=%d max_abs_err=%.6f\n",
           ROWS, COLS, fails, max_err);
    printf(fails == 0 ? "INTERLEAVED TERNARY Q2 PASS!\n" : "INTERLEAVED TERNARY Q2 FAIL!\n");
    return fails == 0 ? 0 : 1;
}
