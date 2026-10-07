// Bonsai Q1MatMul HTP OpPackage Interface and Registration
#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/unique_types.h"
#include "QnnOpPackage.h"
#include "QnnTypes.h"
#include "QnnLog.h"
#include "QnnCommon.h"

INIT_PACKAGE_OP_DEF()
INIT_PACKAGE_OPTIMIZATION_DEF()

DEFINE_UNIQ_TY()
BEGIN_PKG_OPS_OPTS_LIST()
DECLARE_PKG_OPS_OPTS_LIST(PKG_BonsaiQ1Htp)
END_PKG_OPS_OPTS_LIST()

INIT_PKG_CORE_INIT_FUNC()

// ---- QNN OpPackage interface ----
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
