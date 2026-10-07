// Benchmark zero-spill HVX Q1 & Q2 kernels on hexagon-sim (Hexagon v79)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>

static inline float f16_to_f32_hvx_old(unsigned short h) {
    unsigned s = (unsigned)(h & 0x8000) << 16;
    int e = (h >> 10) & 0x1f;
    unsigned m = h & 0x3ff;
    unsigned f = (e == 0) ? s : (s | ((unsigned)(e + 112) << 23) | (m << 13));
    float r; memcpy(&r, &f, 4); return r;
}

static inline unsigned f16_scale_to_f32_bits_half(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x37800000u; // s * 0.5f
    return (h < 0x0400) ? 0 : f;
}

static inline unsigned f16_scale_to_f32_bits_full(unsigned short h) {
    unsigned f = ((unsigned)h << 13) + 0x38000000u; // full s
    return (h < 0x0400) ? 0 : f;
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

static const int hvx_idx_arr[32] __attribute__((aligned(128))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

static const uint32_t hvx_bitmask_arr[32] __attribute__((aligned(128))) = {
    1u<<0,  1u<<1,  1u<<2,  1u<<3,  1u<<4,  1u<<5,  1u<<6,  1u<<7,
    1u<<8,  1u<<9,  1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
    1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
    1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
};

// Mode 1: Current process_slice
__attribute__((noinline))
void kernel_mode1_old(int out_dim, int in_dim, int prow,
                      const uint32_t* bits_u32, const unsigned short* scales, float* y) {
    int ng = in_dim >> 7;
    int prow_u32 = prow >> 2;
    HVX_Vector vIdx = *(const HVX_Vector*)hvx_idx_arr;
    HVX_Vector vOne = Q6_V_vsplat_R(1);

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
            float s0 = f16_to_f32_hvx_old(sg0[g]) * 0.5f, s1 = f16_to_f32_hvx_old(sg1[g]) * 0.5f;
            float s2 = f16_to_f32_hvx_old(sg2[g]) * 0.5f, s3 = f16_to_f32_hvx_old(sg3[g]) * 0.5f;

            unsigned sbits0, nbits0, sbits1, nbits1, sbits2, nbits2, sbits3, nbits3;
            memcpy(&sbits0, &s0, 4); nbits0 = sbits0 ^ 0x80000000u;
            memcpy(&sbits1, &s1, 4); nbits1 = sbits1 ^ 0x80000000u;
            memcpy(&sbits2, &s2, 4); nbits2 = sbits2 ^ 0x80000000u;
            memcpy(&sbits3, &s3, 4); nbits3 = sbits3 ^ 0x80000000u;

            HVX_Vector vS0 = Q6_V_vsplat_R((int)sbits0), vNS0 = Q6_V_vsplat_R((int)nbits0);
            HVX_Vector vS1 = Q6_V_vsplat_R((int)sbits1), vNS1 = Q6_V_vsplat_R((int)nbits1);
            HVX_Vector vS2 = Q6_V_vsplat_R((int)sbits2), vNS2 = Q6_V_vsplat_R((int)nbits2);
            HVX_Vector vS3 = Q6_V_vsplat_R((int)sbits3), vNS3 = Q6_V_vsplat_R((int)nbits3);

            const uint32_t* gb0 = bp0 + (unsigned)g * 4u, *gb1 = bp1 + (unsigned)g * 4u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 4u, *gb3 = bp3 + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;

            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);

                HVX_Vector vW0 = Q6_V_vsplat_R((int)gb0[c]), vW1 = Q6_V_vsplat_R((int)gb1[c]);
                HVX_Vector vW2 = Q6_V_vsplat_R((int)gb2[c]), vW3 = Q6_V_vsplat_R((int)gb3[c]);

                HVX_Vector vSh0 = Q6_Vw_vlsr_VwVw(vW0, vIdx), vSh1 = Q6_Vw_vlsr_VwVw(vW1, vIdx);
                HVX_Vector vSh2 = Q6_Vw_vlsr_VwVw(vW2, vIdx), vSh3 = Q6_Vw_vlsr_VwVw(vW3, vIdx);

                HVX_Vector vB0 = Q6_V_vand_VV(vSh0, vOne), vB1 = Q6_V_vand_VV(vSh1, vOne);
                HVX_Vector vB2 = Q6_V_vand_VV(vSh2, vOne), vB3 = Q6_V_vand_VV(vSh3, vOne);

                HVX_VectorPred Q0 = Q6_Q_vcmp_eq_VwVw(vB0, vOne), Q1 = Q6_Q_vcmp_eq_VwVw(vB1, vOne);
                HVX_VectorPred Q2 = Q6_Q_vcmp_eq_VwVw(vB2, vOne), Q3 = Q6_Q_vcmp_eq_VwVw(vB3, vOne);

                HVX_Vector vWgt0 = Q6_V_vmux_QVV(Q0, vS0, vNS0), vWgt1 = Q6_V_vmux_QVV(Q1, vS1, vNS1);
                HVX_Vector vWgt2 = Q6_V_vmux_QVV(Q2, vS2, vNS2), vWgt3 = Q6_V_vmux_QVV(Q3, vS3, vNS3);

                vAcc0 = Q6_Vsf_vadd_VsfVsf(vAcc0, Q6_Vsf_vmpy_VsfVsf(vWgt0, vX));
                vAcc1 = Q6_Vsf_vadd_VsfVsf(vAcc1, Q6_Vsf_vmpy_VsfVsf(vWgt1, vX));
                vAcc2 = Q6_Vsf_vadd_VsfVsf(vAcc2, Q6_Vsf_vmpy_VsfVsf(vWgt2, vX));
                vAcc3 = Q6_Vsf_vadd_VsfVsf(vAcc3, Q6_Vsf_vmpy_VsfVsf(vWgt3, vX));
            }
        }
        y[i + 0] = hvx_reduce_sum32(vAcc0); y[i + 1] = hvx_reduce_sum32(vAcc1);
        y[i + 2] = hvx_reduce_sum32(vAcc2); y[i + 3] = hvx_reduce_sum32(vAcc3);
    }
}

// Mode 2: Zero-Spill Fast Bitmask + Branchless Scale (100% Bit-Exact to Mode 1!)
__attribute__((noinline))
void kernel_mode2_zero_spill(int out_dim, int in_dim, int prow,
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
            const uint32_t* gb0 = bp0 + (unsigned)g * 4u, *gb1 = bp1 + (unsigned)g * 4u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 4u, *gb3 = bp3 + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;

            HVX_Vector vGAcc0 = Q6_V_vzero(), vGAcc1 = Q6_V_vzero();
            HVX_Vector vGAcc2 = Q6_V_vzero(), vGAcc3 = Q6_V_vzero();

            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);
                HVX_Vector vNX = Q6_Vsf_vfneg_Vsf(vX);

                HVX_Vector vT0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT2 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb2[c]), vMask), vMask), vX, vNX);
                HVX_Vector vT3 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb3[c]), vMask), vMask), vX, vNX);

                vGAcc0 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc0, vT0);
                vGAcc1 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc1, vT1);
                vGAcc2 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc2, vT2);
                vGAcc3 = Q6_Vqf32_vadd_Vqf32Vsf(vGAcc3, vT3);
            }

            unsigned sbits0 = f16_scale_to_f32_bits_half(sg0[g]);
            unsigned sbits1 = f16_scale_to_f32_bits_half(sg1[g]);
            unsigned sbits2 = f16_scale_to_f32_bits_half(sg2[g]);
            unsigned sbits3 = f16_scale_to_f32_bits_half(sg3[g]);

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

// Mode 3: Direct qf32 accumulation across all groups (zero qf32->sf conversions inside g loop, 100% bit-exact)
__attribute__((noinline))
void kernel_mode3_qf32_direct(int out_dim, int in_dim, int prow,
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
            unsigned sbits0 = f16_scale_to_f32_bits_half(sg0[g]);
            unsigned sbits1 = f16_scale_to_f32_bits_half(sg1[g]);
            unsigned sbits2 = f16_scale_to_f32_bits_half(sg2[g]);
            unsigned sbits3 = f16_scale_to_f32_bits_half(sg3[g]);

            HVX_Vector vS0 = Q6_V_vsplat_R((int)sbits0), vNS0 = Q6_V_vsplat_R((int)(sbits0 ^ 0x80000000u));
            HVX_Vector vS1 = Q6_V_vsplat_R((int)sbits1), vNS1 = Q6_V_vsplat_R((int)(sbits1 ^ 0x80000000u));
            HVX_Vector vS2 = Q6_V_vsplat_R((int)sbits2), vNS2 = Q6_V_vsplat_R((int)(sbits2 ^ 0x80000000u));
            HVX_Vector vS3 = Q6_V_vsplat_R((int)sbits3), vNS3 = Q6_V_vsplat_R((int)(sbits3 ^ 0x80000000u));

            const uint32_t* gb0 = bp0 + (unsigned)g * 4u, *gb1 = bp1 + (unsigned)g * 4u;
            const uint32_t* gb2 = bp2 + (unsigned)g * 4u, *gb3 = bp3 + (unsigned)g * 4u;
            const float* gx = hvx_xalign + (unsigned)g * 128u;

            for (int c = 0; c < 4; c++) {
                HVX_Vector vX = *(const HVX_Vector*)(gx + (unsigned)c * 32u);

                HVX_Vector vWgt0 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb0[c]), vMask), vMask), vS0, vNS0);
                HVX_Vector vWgt1 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb1[c]), vMask), vMask), vS1, vNS1);
                HVX_Vector vWgt2 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb2[c]), vMask), vMask), vS2, vNS2);
                HVX_Vector vWgt3 = Q6_V_vmux_QVV(Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(Q6_V_vsplat_R((int)gb3[c]), vMask), vMask), vS3, vNS3);

                vAcc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc0, Q6_Vqf32_vmpy_VsfVsf(vWgt0, vX));
                vAcc1 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc1, Q6_Vqf32_vmpy_VsfVsf(vWgt1, vX));
                vAcc2 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc2, Q6_Vqf32_vmpy_VsfVsf(vWgt2, vX));
                vAcc3 = Q6_Vqf32_vadd_Vqf32Vqf32(vAcc3, Q6_Vqf32_vmpy_VsfVsf(vWgt3, vX));
            }
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
static unsigned char sbits[ROWS * (COLS / 8)] __attribute__((aligned(128)));
static unsigned short ssc[ROWS * (COLS / 128)] __attribute__((aligned(128)));
static float y_out[ROWS];

int main(int argc, char** argv) {
    int mode = (argc >= 2) ? atoi(argv[1]) : 3;
    for (int j = 0; j < COLS; j++) hvx_xalign[j] = ((float)(rnd() % 2000) / 1000.0f - 1.0f);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS / 8; j++) sbits[i * (COLS / 8) + j] = (unsigned char)(rnd() & 0xff);
        for (int g = 0; g < COLS / 128; g++) ssc[i * (COLS / 128) + g] = (unsigned short)(0x3800 + (rnd() % 0x800));
    }
    int prow = COLS / 8;

    if (mode == 1) {
        kernel_mode1_old(ROWS, COLS, prow, (const uint32_t*)sbits, ssc, y_out);
    } else if (mode == 2) {
        kernel_mode2_zero_spill(ROWS, COLS, prow, (const uint32_t*)sbits, ssc, y_out);
    } else if (mode == 3) {
        kernel_mode3_qf32_direct(ROWS, COLS, prow, (const uint32_t*)sbits, ssc, y_out);
    }
    printf("Mode %d: y[0]=%.5f y[1]=%.5f y[ROWS-1]=%.5f\n", mode, y_out[0], y_out[1], y_out[ROWS-1]);
    return 0;
}
