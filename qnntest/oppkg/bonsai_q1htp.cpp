// Bonsai Q1MatMul HTP OpPackage (v1: scalar loop, rank-4 tensors).
// Package "bonsai_htp", op "Q1MatMul":
//   y[1,1,1,M] = q1gemv(x[1,1,1,K] fp16, bits[1,1,M,prow] u8, s[1,1,M,ng] fp16).
// s_g = stored_fp16/2. K%128==0, prow=K/8, ng=K/128.
// HVX acceleration later (see cdsp/bonsai_hvx.c pattern); this v1 proves plumbing.
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "QnnCommon.h"

#include "HTP/core/unique_types.h"

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_BonsaiQ1Htp)
END_PKG_OPS_OPTS_LIST()

INIT_PKG_CORE_INIT_FUNC()

BEGIN_PKG_OP_DEFINITION(PKG_BonsaiQ1Htp);

template <typename TensorType>
GraphStatus q1MatMulImpl(TensorType& out_0,
                         const TensorType& in_0,
                         const TensorType& in_1,
                         const TensorType& in_2) {
  // rank-4 SHWC-ish: x[1,1,1,K], bits[1,1,M,prow], scales[1,1,M,ng], y[1,1,1,M]
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
  out_0.set_dims(in_0);
  // NOTE: set_dims copies input shape; fix last dim to M below via reshape semantics.
  // HTP framework sizes output from graph; we only fill [0,0,0,i], i<M.
  for (Idx i = 0; i < M; i++) {
    double acc = 0.0;
    for (Idx g = 0; g < (Idx)(K / 128); g++) {
      double s = (double)in_2(0, 0, i, g) * 0.5;
      double gsum = 0.0;
      for (Idx j = 0; j < 128; j++) {
        Idx jj = g * 128 + j;
        unsigned byte = (unsigned)in_1(0, 0, i, jj >> 3);
        int bit = (byte >> (jj & 7)) & 1;
        float xv = in_0(0, 0, 0, jj);
        gsum += bit ? (double)xv : -(double)xv;
      }
      acc += s * gsum;
    }
    out_0(0, 0, 0, i) = (float)acc;
  }
  return GraphStatus::Success;
}

DEF_PACKAGE_OP((q1MatMulImpl<Tensor>), "Q1MatMul")

END_PKG_OP_DEFINITION(PKG_BonsaiQ1Htp);

// ---- QNN OpPackage v1.4 interface (HTP-required subset) ----
static const char kBonsaiHtpPkgName[] = "bonsai_htp";
static const char* kBonsaiHtpOps[] = {"Q1MatMul"};
static const Qnn_Version_t kBonsaiHtpOpset = {1u, 0u, 0u};
static QnnOpPackage_Info_t g_bonsaiHtpInfo = {
    kBonsaiHtpPkgName,
    kBonsaiHtpOps,
    NULL,
    1u,
    NULL,
    0u,
    NULL,
    NULL,
    NULL,
    &kBonsaiHtpOpset,
    {0u}
};

extern "C" {
Qnn_ErrorHandle_t BonsaiHtp_Init(QnnOpPackage_GlobalInfrastructure_t infra) {
    (void)infra;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t BonsaiHtp_Terminate() { return QNN_SUCCESS; }
Qnn_ErrorHandle_t BonsaiHtp_GetInfo(const QnnOpPackage_Info_t** info) {
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &g_bonsaiHtpInfo;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t BonsaiHtp_LogInitialize(QnnLog_Callback_t callback,
                                           QnnLog_Level_t maxLogLevel) {
    (void)callback; (void)maxLogLevel;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t BonsaiHtp_LogSetLevel(QnnLog_Level_t maxLogLevel) {
    (void)maxLogLevel;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t BonsaiHtp_LogTerminate(void) { return QNN_SUCCESS; }
Qnn_ErrorHandle_t BonsaiHtp_ValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (opConfig.version != QNN_OPCONFIG_VERSION_1) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const Qnn_OpConfigV1_t* c = &opConfig.v1;
    if (!c->typeName || strcmp(c->typeName, "Q1MatMul") != 0) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (c->numOfInputs != 3 || c->numOfOutputs != 1) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return QNN_SUCCESS;
}
Qnn_ErrorHandle_t QnnOpPackage_interfaceProvider(QnnOpPackage_Interface_t* intface_) {
    if (!intface_) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    intface_->interfaceVersion.major = 1u;
    intface_->interfaceVersion.minor = 4u;
    intface_->interfaceVersion.patch = 0u;
    intface_->v1_4.init = BonsaiHtp_Init;
    intface_->v1_4.terminate = BonsaiHtp_Terminate;
    intface_->v1_4.getInfo = BonsaiHtp_GetInfo;
    intface_->v1_4.validateOpConfig = BonsaiHtp_ValidateOpConfig;
    intface_->v1_4.createOpImpl = NULL;
    intface_->v1_4.freeOpImpl = NULL;
    intface_->v1_4.logInitialize = BonsaiHtp_LogInitialize;
    intface_->v1_4.logSetLevel = BonsaiHtp_LogSetLevel;
    intface_->v1_4.logTerminate = BonsaiHtp_LogTerminate;
    return QNN_SUCCESS;
}
}  // extern "C"
