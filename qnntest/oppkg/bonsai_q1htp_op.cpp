// Bonsai Q1MatMul HTP Op Implementation with 4-way HVX Vector Acceleration & Register Reductions
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"

#ifdef __hexagon__
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <string.h>

#pragma weak HAP_power_request
extern "C" int HAP_power_request(int clock, int bus, int latency);

static inline float f16_to_f32_hvx(unsigned short h) {
    unsigned s = (unsigned)(h & 0x8000) << 16;
    int e = (h >> 10) & 0x1f;
    unsigned m = h & 0x3ff;
    unsigned f = (e == 0) ? s : (s | (unsigned)(e + 112) << 23 | (m << 13));
    float r;
    memcpy(&r, &f, 4);
    return r;
}

static inline unsigned short f32_to_f16_hvx(float f) {
    unsigned u;
    memcpy(&u, &f, 4);
    unsigned s = (u >> 16) & 0x8000;
    int e = (int)((u >> 23) & 0xff) - 112;
    unsigned m = u & 0x7fffff;
    if (e <= 0) return (unsigned short)s;
    if (e >= 31) return (unsigned short)(s | 0x7bff);
    return (unsigned short)(s | ((unsigned)e << 10) | (m >> 13));
}

static inline float hvx_reduce_sum32(HVX_Vector v) {
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vsf_vadd_VsfVsf(v, Q6_V_vror_VR(v, 4));
    float r;
    memcpy(&r, &v, 4);
    return r;
}

#define HVX_MAX_K 17408
static float hvx_xalign[HVX_MAX_K] __attribute__((aligned(128)));
static const int hvx_idx_arr[32] __attribute__((aligned(128))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};
#endif

BEGIN_PKG_OP_DEFINITION(PKG_BonsaiQ1Htp);

template <typename TensorType>
GraphStatus q1MatMulImpl(TensorType& out_0,
                         const TensorType& in_0,
                         const TensorType& in_1,
                         const TensorType& in_2) {
  // rank-4 SHWC: x[1,1,1,K], bits[1,1,M,prow], scales[1,1,M,ng], y[1,1,1,M]
  auto [bx, hx, wx, dx] = in_0.dims();
  auto [bb, hb, wb, db] = in_1.dims();
  auto [bs, hs, ws, ds] = in_2.dims();
  if (bx != 1 || hx != 1 || wx != 1) return GraphStatus::ErrorFatal;
  if (bb != 1 || hb != 1 || bs != 1 || hs != 1) return GraphStatus::ErrorFatal;
  Idx K = dx;
  Idx M = wb;
  if (K <= 0 || (K & 127) != 0) return GraphStatus::ErrorFatal;
  if (db != (Idx)(K / 8)) return GraphStatus::ErrorFatal;
  if (ws != M || ds != (Idx)(K / 128)) return GraphStatus::ErrorFatal;

  size_t odims[4] = {1, 1, 1, (size_t)M};
  out_0.set_dims(odims);

#ifdef __hexagon__
  if (HAP_power_request) {
    HAP_power_request(100, 100, 100);
  }

  const uint16_t* x_raw = (const uint16_t*)in_0.raw_data_const();
  const uint8_t* bits_raw = (const uint8_t*)in_1.raw_data_const();
  const uint16_t* scales_raw = (const uint16_t*)in_2.raw_data_const();
  uint16_t* out_raw = (uint16_t*)out_0.raw_data();

  if (x_raw && bits_raw && scales_raw && out_raw && K <= HVX_MAX_K) {
    for (Idx j = 0; j < K; j++) {
      hvx_xalign[j] = f16_to_f32_hvx(x_raw[j]);
    }
    Idx ng = K / 128;
    Idx prow_u32 = (K / 8) / 4; // words per row
    const uint32_t* bits_u32 = (const uint32_t*)bits_raw;
    HVX_Vector vIdx = *(HVX_Vector*)hvx_idx_arr;
    HVX_Vector vOne = Q6_V_vsplat_R(1);

    Idx i = 0;
    for (; i + 3 < M; i += 4) {
      const uint16_t* sg0 = scales_raw + (i + 0) * ng;
      const uint16_t* sg1 = scales_raw + (i + 1) * ng;
      const uint16_t* sg2 = scales_raw + (i + 2) * ng;
      const uint16_t* sg3 = scales_raw + (i + 3) * ng;

      const uint32_t* bp0 = bits_u32 + (i + 0) * prow_u32;
      const uint32_t* bp1 = bits_u32 + (i + 1) * prow_u32;
      const uint32_t* bp2 = bits_u32 + (i + 2) * prow_u32;
      const uint32_t* bp3 = bits_u32 + (i + 3) * prow_u32;

      HVX_Vector vAcc0 = Q6_V_vzero();
      HVX_Vector vAcc1 = Q6_V_vzero();
      HVX_Vector vAcc2 = Q6_V_vzero();
      HVX_Vector vAcc3 = Q6_V_vzero();

      for (Idx g = 0; g < ng; g++) {
        float s0 = f16_to_f32_hvx(sg0[g]) * 0.5f;
        float s1 = f16_to_f32_hvx(sg1[g]) * 0.5f;
        float s2 = f16_to_f32_hvx(sg2[g]) * 0.5f;
        float s3 = f16_to_f32_hvx(sg3[g]) * 0.5f;

        unsigned sbits0, nbits0, sbits1, nbits1, sbits2, nbits2, sbits3, nbits3;
        memcpy(&sbits0, &s0, 4); nbits0 = sbits0 ^ 0x80000000u;
        memcpy(&sbits1, &s1, 4); nbits1 = sbits1 ^ 0x80000000u;
        memcpy(&sbits2, &s2, 4); nbits2 = sbits2 ^ 0x80000000u;
        memcpy(&sbits3, &s3, 4); nbits3 = sbits3 ^ 0x80000000u;

        HVX_Vector vS0 = Q6_V_vsplat_R((int)sbits0);
        HVX_Vector vNS0 = Q6_V_vsplat_R((int)nbits0);
        HVX_Vector vS1 = Q6_V_vsplat_R((int)sbits1);
        HVX_Vector vNS1 = Q6_V_vsplat_R((int)nbits1);
        HVX_Vector vS2 = Q6_V_vsplat_R((int)sbits2);
        HVX_Vector vNS2 = Q6_V_vsplat_R((int)nbits2);
        HVX_Vector vS3 = Q6_V_vsplat_R((int)sbits3);
        HVX_Vector vNS3 = Q6_V_vsplat_R((int)nbits3);

        const uint32_t* gb0 = bp0 + g * 4u;
        const uint32_t* gb1 = bp1 + g * 4u;
        const uint32_t* gb2 = bp2 + g * 4u;
        const uint32_t* gb3 = bp3 + g * 4u;
        const float* gx = hvx_xalign + g * 128u;

        for (int c = 0; c < 4; c++) {
          HVX_Vector vX = *(HVX_Vector*)(gx + c * 32u);

          uint32_t w0 = gb0[c];
          uint32_t w1 = gb1[c];
          uint32_t w2 = gb2[c];
          uint32_t w3 = gb3[c];

          HVX_Vector vW0 = Q6_V_vsplat_R((int)w0);
          HVX_Vector vW1 = Q6_V_vsplat_R((int)w1);
          HVX_Vector vW2 = Q6_V_vsplat_R((int)w2);
          HVX_Vector vW3 = Q6_V_vsplat_R((int)w3);

          HVX_Vector vSh0 = Q6_Vw_vlsr_VwVw(vW0, vIdx);
          HVX_Vector vSh1 = Q6_Vw_vlsr_VwVw(vW1, vIdx);
          HVX_Vector vSh2 = Q6_Vw_vlsr_VwVw(vW2, vIdx);
          HVX_Vector vSh3 = Q6_Vw_vlsr_VwVw(vW3, vIdx);

          HVX_Vector vB0 = Q6_V_vand_VV(vSh0, vOne);
          HVX_Vector vB1 = Q6_V_vand_VV(vSh1, vOne);
          HVX_Vector vB2 = Q6_V_vand_VV(vSh2, vOne);
          HVX_Vector vB3 = Q6_V_vand_VV(vSh3, vOne);

          HVX_VectorPred Q0 = Q6_Q_vcmp_eq_VwVw(vB0, vOne);
          HVX_VectorPred Q1 = Q6_Q_vcmp_eq_VwVw(vB1, vOne);
          HVX_VectorPred Q2 = Q6_Q_vcmp_eq_VwVw(vB2, vOne);
          HVX_VectorPred Q3 = Q6_Q_vcmp_eq_VwVw(vB3, vOne);

          HVX_Vector vWgt0 = Q6_V_vmux_QVV(Q0, vS0, vNS0);
          HVX_Vector vWgt1 = Q6_V_vmux_QVV(Q1, vS1, vNS1);
          HVX_Vector vWgt2 = Q6_V_vmux_QVV(Q2, vS2, vNS2);
          HVX_Vector vWgt3 = Q6_V_vmux_QVV(Q3, vS3, vNS3);

          HVX_Vector vMul0 = Q6_Vsf_vmpy_VsfVsf(vWgt0, vX);
          HVX_Vector vMul1 = Q6_Vsf_vmpy_VsfVsf(vWgt1, vX);
          HVX_Vector vMul2 = Q6_Vsf_vmpy_VsfVsf(vWgt2, vX);
          HVX_Vector vMul3 = Q6_Vsf_vmpy_VsfVsf(vWgt3, vX);

          vAcc0 = Q6_Vsf_vadd_VsfVsf(vAcc0, vMul0);
          vAcc1 = Q6_Vsf_vadd_VsfVsf(vAcc1, vMul1);
          vAcc2 = Q6_Vsf_vadd_VsfVsf(vAcc2, vMul2);
          vAcc3 = Q6_Vsf_vadd_VsfVsf(vAcc3, vMul3);
        }
      }
      out_raw[i + 0] = f32_to_f16_hvx(hvx_reduce_sum32(vAcc0));
      out_raw[i + 1] = f32_to_f16_hvx(hvx_reduce_sum32(vAcc1));
      out_raw[i + 2] = f32_to_f16_hvx(hvx_reduce_sum32(vAcc2));
      out_raw[i + 3] = f32_to_f16_hvx(hvx_reduce_sum32(vAcc3));
    }

    // Handle tail if any (M % 4)
    for (; i < M; i++) {
      const uint16_t* sg = scales_raw + i * ng;
      const uint32_t* bp = bits_u32 + i * prow_u32;
      HVX_Vector vAcc = Q6_V_vzero();
      for (Idx g = 0; g < ng; g++) {
        float s = f16_to_f32_hvx(sg[g]) * 0.5f;
        unsigned sbits, nbits;
        memcpy(&sbits, &s, 4); nbits = sbits ^ 0x80000000u;
        HVX_Vector vS = Q6_V_vsplat_R((int)sbits);
        HVX_Vector vNS = Q6_V_vsplat_R((int)nbits);
        const uint32_t* gb = bp + g * 4u;
        const float* gx = hvx_xalign + g * 128u;
        for (int c = 0; c < 4; c++) {
          HVX_Vector vX = *(HVX_Vector*)(gx + c * 32u);
          uint32_t w = gb[c];
          HVX_Vector vW = Q6_V_vsplat_R((int)w);
          HVX_Vector vSh = Q6_Vw_vlsr_VwVw(vW, vIdx);
          HVX_Vector vB = Q6_V_vand_VV(vSh, vOne);
          HVX_VectorPred Q = Q6_Q_vcmp_eq_VwVw(vB, vOne);
          HVX_Vector vWgt = Q6_V_vmux_QVV(Q, vS, vNS);
          vAcc = Q6_Vsf_vadd_VsfVsf(vAcc, Q6_Vsf_vmpy_VsfVsf(vWgt, vX));
        }
      }
      out_raw[i] = f32_to_f16_hvx(hvx_reduce_sum32(vAcc));
    }
    return GraphStatus::Success;
  }
#endif

  // Fallback / ARM64 prepare path
  for (Idx i = 0; i < M; i++) {
    float acc = 0.0f;
    for (Idx g = 0; g < (Idx)(K / 128); g++) {
      float s = (float)in_2(0, 0, i, g) * 0.5f;
      float gsum = 0.0f;
      for (Idx j = 0; j < 128; j++) {
        Idx jj = g * 128 + j;
        unsigned byte = (unsigned)in_1(0, 0, i, jj >> 3);
        int bit = (byte >> (jj & 7)) & 1;
        float xv = in_0(0, 0, 0, jj);
        gsum += bit ? xv : -xv;
      }
      acc += s * gsum;
    }
    out_0(0, 0, 0, i) = acc;
  }
  return GraphStatus::Success;
}

DEF_PACKAGE_OP_AND_COST_AND_FLAGS((q1MatMulImpl<Tensor>), "Q1MatMul", SNAIL, Flags::RESOURCE_HVX)

END_PKG_OP_DEFINITION(PKG_BonsaiQ1Htp);
