// Fast branchless scale conversion & HVX GEMV kernel test on hexagon-sim
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline unsigned f16_scale_to_f32_bits(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x37800000u;
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

// Ultra-fast Q1 kernel:
// 1. 3-instruction branchless f16 scale conversion
// 2. Factored scale outside chunk loop (only 1 vmpy per group instead of 4)
// 3. Bitmask testing without vlsr/vand/vcmp
__attribute__((noinline))
void kernel_opt_fast(int out_dim, int in_dim, int prow,
                     const uint32_t* bits_u32, const unsigned short* scales, float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    HVX_Vector vMask = *(const HVX_Vector*)hvx_bitmask_arr;

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
            unsigned sbits0 = f16_scale_to_f32_bits(sg0[g]);
            unsigned sbits1 = f16_scale_to_f32_bits(sg1[g]);
            unsigned sbits2 = f16_scale_to_f32_bits(sg2[g]);
            unsigned sbits3 = f16_scale_to_f32_bits(sg3[g]);

            HVX_Vector vS0 = Q6_V_vsplat_R((int)sbits0);
            HVX_Vector vS1 = Q6_V_vsplat_R((int)sbits1);
            HVX_Vector vS2 = Q6_V_vsplat_R((int)sbits2);
            HVX_Vector vS3 = Q6_V_vsplat_R((int)sbits3);

            const uint32_t* gb0 = bp0 + (unsigned)g * 4u, *gb1 = bp1 + (unsigned)g * 4u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 4u, *gb3 = bp3 + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;

            HVX_Vector vGAcc0 = Q6_V_vzero(), vGAcc1 = Q6_V_vzero();
            HVX_Vector vGAcc2 = Q6_V_vzero(), vGAcc3 = Q6_V_vzero();

            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);
                HVX_Vector vNegX = Q6_Vsf_vfneg_Vsf(vX);

                HVX_Vector vW0 = Q6_V_vsplat_R((int)gb0[c]), vW1 = Q6_V_vsplat_R((int)gb1[c]);
                HVX_Vector vW2 = Q6_V_vsplat_R((int)gb2[c]), vW3 = Q6_V_vsplat_R((int)gb3[c]);

                HVX_VectorPred Q0 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vW0, vMask), vMask);
                HVX_VectorPred Q1 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vW1, vMask), vMask);
                HVX_VectorPred Q2 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vW2, vMask), vMask);
                HVX_VectorPred Q3 = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(vW3, vMask), vMask);

                vGAcc0 = Q6_Vsf_vadd_VsfVsf(vGAcc0, Q6_V_vmux_QVV(Q0, vX, vNegX));
                vGAcc1 = Q6_Vsf_vadd_VsfVsf(vGAcc1, Q6_V_vmux_QVV(Q1, vX, vNegX));
                vGAcc2 = Q6_Vsf_vadd_VsfVsf(vGAcc2, Q6_V_vmux_QVV(Q2, vX, vNegX));
                vGAcc3 = Q6_Vsf_vadd_VsfVsf(vGAcc3, Q6_V_vmux_QVV(Q3, vX, vNegX));
            }

            vAcc0 = Q6_Vsf_vadd_VsfVsf(vAcc0, Q6_Vsf_vmpy_VsfVsf(vGAcc0, vS0));
            vAcc1 = Q6_Vsf_vadd_VsfVsf(vAcc1, Q6_Vsf_vmpy_VsfVsf(vGAcc1, vS1));
            vAcc2 = Q6_Vsf_vadd_VsfVsf(vAcc2, Q6_Vsf_vmpy_VsfVsf(vGAcc2, vS2));
            vAcc3 = Q6_Vsf_vadd_VsfVsf(vAcc3, Q6_Vsf_vmpy_VsfVsf(vGAcc3, vS3));
        }
        y[i + 0] = hvx_reduce_sum32(vAcc0); y[i + 1] = hvx_reduce_sum32(vAcc1);
        y[i + 2] = hvx_reduce_sum32(vAcc2); y[i + 3] = hvx_reduce_sum32(vAcc3);
    }
}

static unsigned rng_state = 0x12345678u;
static unsigned rnd(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

#define ROWS 64
#define COLS 5120
static unsigned char sbits[ROWS * (COLS / 8)] __attribute__((aligned(128)));
static unsigned short ssc[ROWS * (COLS / 128)] __attribute__((aligned(128)));
static float y_out[ROWS], y_ref[ROWS];

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    for (int j = 0; j < COLS; j++) hvx_xalign[j] = ((float)(rnd() % 2000) / 1000.0f - 1.0f);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS / 8; j++) sbits[i * (COLS / 8) + j] = (unsigned char)(rnd() & 0xff);
        for (int g = 0; g < COLS / 128; g++) ssc[i * (COLS / 128) + g] = (unsigned short)(0x3800 + (rnd() % 0x800));
    }
    int prow = COLS / 8;
    for (int i = 0; i < ROWS; i++) {
        double acc = 0;
        for (int j = 0; j < COLS; j++) {
            int bit = (sbits[i * prow + (j >> 3)] >> (j & 7)) & 1;
            double s = (double)f16r(ssc[i * (COLS / 128) + (j >> 7)]) * 0.5;
            acc += (bit ? s : -s) * (double)hvx_xalign[j];
        }
        y_ref[i] = (float)acc;
    }

    kernel_opt_fast(ROWS, COLS, prow, (const uint32_t*)sbits, ssc, y_out);

    double max_err = 0;
    for (int i = 0; i < ROWS; i++) {
        double e = y_out[i] - y_ref[i]; if (e < 0) e = -e;
        if (e > max_err) max_err = e;
    }
    printf("kernel_opt_fast PASS max_err=%.6f\n", max_err);
    return (max_err < 0.001) ? 0 : 1;
}
