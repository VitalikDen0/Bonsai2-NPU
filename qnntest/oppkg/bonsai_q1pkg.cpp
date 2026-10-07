// Bonsai Q1MatMul CPU OpPackage (QNN OpPackage API v1.4).
// Package "bonsai", op "Q1MatMul": y[1,M] = q1gemv(x[1,K] fp32, bits[M,prow] u8, s[M,ng] fp32).
// Graph inserts Cast fp16<->fp32 around it; scales pre-divided (s_g) fp32 STATIC at load.
// HTP/HVX variant later; this CPU path runs anywhere (incl. on-device ARM CPU).
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CPU/QnnCpuOpPackage.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnOpPackage.h"
#include "QNN/QnnTypes.h"

namespace {

const char kPackageName[] = "bonsai";
const char kOpName[] = "Q1MatMul";
const char* kOpNames[] = {kOpName};
const Qnn_Version_t kOpsetVersion = {1u, 0u, 0u};

QnnOpPackage_Info_t g_info = {
    kPackageName,           // packageName
    kOpNames,               // operationNames
    NULL,                   // operationInfo
    1u,                     // numOperations
    NULL,                   // optimizations
    0u,                     // numOptimizations
    NULL,                   // sdkBuildId
    NULL,                   // sdkApiVersion
    NULL,                   // packageInfo
    &kOpsetVersion,         // opsetVersion
    {0u}                    // reserved
};

struct Q1Impl {
    int M;
    int K;
    int prow;
    int ng;
};

int q1ValidateDims(int M, int K, int prow, int ng, int nscales) {
    if (M <= 0 || K <= 0 || prow <= 0 || ng <= 0) return -1;
    if ((K & 127) != 0) return -2;
    if (prow != K / 8) return -2;
    if (ng != K / 128) return -2;
    if (nscales != M * ng) return -2;
    return 0;
}

int q1Execute(int M, int K, int prow, const float* x, const unsigned char* bits,
              const float* scales, float* y) {
    int ng = K >> 7;
    if (!x || !bits || !scales || !y) return -1;
    if (M <= 0 || K <= 0 || (K & 127) != 0 || prow != K / 8) return -2;
    for (int i = 0; i < M; i++) {
        const float* sg = scales + (unsigned)i * (unsigned)ng;
        const unsigned char* bp = bits + (unsigned)i * (unsigned)prow;
        double acc = 0.0;
        for (int g = 0; g < ng; g++) {
            double s = (double)sg[g];
            double gsum = 0.0;
            int base = g * 128;
            for (int j = 0; j < 128; j++) {
                int jj = base + j;
                int bit = (bp[jj >> 3] >> (jj & 7)) & 1;
                gsum += bit ? (double)x[jj] : -(double)x[jj];
            }
            acc += s * gsum;
        }
        y[i] = (float)acc;
    }
    return 0;
}

// --- validation on core QNN types (validateOpConfig path) ---
bool dimsMatch1xN(const Qnn_Tensor_t* t, int expectSecond) {
    if (!t || t->version != QNN_TENSOR_VERSION_1) return false;
    if (t->v1.rank != 2) return false;
    if (!t->v1.dimensions) return false;
    return t->v1.dimensions[0] == 1u && (int)t->v1.dimensions[1] == expectSecond;
}

Qnn_ErrorHandle_t q1ValidateOpConfig(Qnn_OpConfig_t opConfig) {
    if (opConfig.version != QNN_OPCONFIG_VERSION_1) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const Qnn_OpConfigV1_t* c = &opConfig.v1;
    if (!c->typeName || strcmp(c->typeName, kOpName) != 0) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (c->numOfInputs != 3 || c->numOfOutputs != 1) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (!c->inputTensors || !c->outputTensors) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    const Qnn_Tensor_t* x = &c->inputTensors[0];
    const Qnn_Tensor_t* b = &c->inputTensors[1];
    const Qnn_Tensor_t* s = &c->inputTensors[2];
    const Qnn_Tensor_t* y = &c->outputTensors[0];
    if (x->version != QNN_TENSOR_VERSION_1 || b->version != QNN_TENSOR_VERSION_1 ||
        s->version != QNN_TENSOR_VERSION_1 || y->version != QNN_TENSOR_VERSION_1)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (x->v1.dataType != QNN_DATATYPE_FLOAT_32 || y->v1.dataType != QNN_DATATYPE_FLOAT_32)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (b->v1.dataType != QNN_DATATYPE_UINT_8 || s->v1.dataType != QNN_DATATYPE_FLOAT_32)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (x->v1.rank != 2 || !x->v1.dimensions) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (x->v1.dimensions[0] != 1u) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    int K = (int)x->v1.dimensions[1];
    if ((K & 127) != 0) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (b->v1.rank != 2 || !b->v1.dimensions) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (s->v1.rank != 2 || !s->v1.dimensions) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    int M = (int)b->v1.dimensions[0];
    if (M <= 0) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if ((int)b->v1.dimensions[1] != K / 8) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if ((int)s->v1.dimensions[0] != M || (int)s->v1.dimensions[1] != K / 128)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if (y->v1.rank != 2 || !y->v1.dimensions) return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    if ((int)y->v1.dimensions[0] != 1 || (int)y->v1.dimensions[1] != M)
        return QNN_OP_PACKAGE_ERROR_VALIDATION_FAILURE;
    return QNN_SUCCESS;
}

// --- CPU execute path ---
Qnn_ErrorHandle_t q1OpImplExecute(void* opImpl) {
    if (!opImpl) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    Q1Impl* impl = (Q1Impl*)opImpl;
    // NOTE: filled by createOpImpl from node tensors; execute reads cached dims.
    (void)impl;
    return QNN_OP_PACKAGE_ERROR_GENERAL;  // replaced below by node-bound impl
}

struct Q1NodeImpl {
    Q1Impl dims;
    QnnCpuOpPackage_Tensor_t* tx;
    const unsigned char* bits;
    const float* scales;
    QnnCpuOpPackage_Tensor_t* ty;
};

Qnn_ErrorHandle_t q1NodeExecute(void* opImpl) {
    if (!opImpl) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    Q1NodeImpl* n = (Q1NodeImpl*)opImpl;
    fprintf(stderr, "[Q1] execute: impl=%p tx=%p ty=%p bits=%p scales=%p M=%d K=%d\n",
            (void*)n, (void*)n->tx, (void*)n->ty, (const void*)n->bits,
            (const void*)n->scales, n->dims.M, n->dims.K);
    fflush(stderr);
    if (!n->tx || !n->ty || !n->bits || !n->scales) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    const float* x = (const float*)n->tx->data;
    float* y = (float*)n->ty->data;
    fprintf(stderr, "[Q1] execute: xdata=%p ydata=%p x[0]=%f\n",
            (const void*)x, (void*)y, x ? x[0] : -999.0f);
    fflush(stderr);
    if (!x || !y) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    int rc = q1Execute(n->dims.M, n->dims.K, n->dims.prow, x, n->bits, n->scales, y);
    fprintf(stderr, "[Q1] execute: rc=%d y[0]=%f\n", rc, y[0]);
    fflush(stderr);
    return rc == 0 ? QNN_SUCCESS : QNN_OP_PACKAGE_ERROR_GENERAL;
}

}  // namespace

extern "C" {

Qnn_ErrorHandle_t BonsaiQ1_Init(QnnOpPackage_GlobalInfrastructure_t infra) {
    (void)infra;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_Terminate() { return QNN_SUCCESS; }

Qnn_ErrorHandle_t BonsaiQ1_GetInfo(const QnnOpPackage_Info_t** info) {
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_INFO;
    *info = &g_info;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_ValidateOpConfig(Qnn_OpConfig_t opConfig) {
    return q1ValidateOpConfig(opConfig);
}

Qnn_ErrorHandle_t BonsaiQ1_CreateOpImpl(QnnOpPackage_GraphInfrastructure_t graphInfra,
                                       QnnOpPackage_Node_t node,
                                       QnnOpPackage_OpImpl_t* opImpl) {
    (void)graphInfra;
    if (!node || !opImpl) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    // CPU backend passes QnnCpuOpPackage_Node_t*
    QnnCpuOpPackage_Node_t* cn = (QnnCpuOpPackage_Node_t*)node;
    if (!cn->typeName || strcmp(cn->typeName, kOpName) != 0) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    if (cn->numOfInputs != 3 || cn->numOfOutputs != 1) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    if (!cn->inputs || !cn->outputs) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    QnnCpuOpPackage_Tensor_t* x = cn->inputs[0];
    QnnCpuOpPackage_Tensor_t* b = cn->inputs[1];
    QnnCpuOpPackage_Tensor_t* s = cn->inputs[2];
    QnnCpuOpPackage_Tensor_t* y = cn->outputs[0];
    if (!x || !b || !s || !y) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    if (x->dataType != QNN_CPU_DATATYPE_FLOAT_32 || y->dataType != QNN_CPU_DATATYPE_FLOAT_32 ||
        b->dataType != QNN_CPU_DATATYPE_UINT_8 || s->dataType != QNN_CPU_DATATYPE_FLOAT_32)
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    if (x->rank != 2 || !x->currentDimensions) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    int K = (int)x->currentDimensions[1];
    if (x->currentDimensions[0] != 1u || (K & 127) != 0) return QNN_OP_PACKAGE_ERROR_GENERAL;
    if (b->rank != 2 || !b->currentDimensions || s->rank != 2 || !s->currentDimensions)
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    int M = (int)b->currentDimensions[0];
    int ng = K >> 7, prow = K >> 3;
    if (M <= 0 || (int)b->currentDimensions[1] != prow) return QNN_OP_PACKAGE_ERROR_GENERAL;
    if ((int)s->currentDimensions[0] != M || (int)s->currentDimensions[1] != ng)
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    if (y->rank != 2 || !y->currentDimensions) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    if (y->currentDimensions[0] != 1u || (int)y->currentDimensions[1] != M)
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    Q1NodeImpl* n = (Q1NodeImpl*)malloc(sizeof(Q1NodeImpl));
    if (!n) return QNN_OP_PACKAGE_ERROR_GENERAL;
    n->dims.M = M;
    n->dims.K = K;
    n->dims.prow = prow;
    n->dims.ng = ng;
    n->tx = x;
    n->bits = (const unsigned char*)b->data;
    n->scales = (const float*)s->data;
    n->ty = y;
    if (!n->bits || !n->scales) {
        free(n);
        return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    }
    QnnCpuOpPackage_OpImpl_t* impl = (QnnCpuOpPackage_OpImpl_t*)malloc(sizeof(QnnCpuOpPackage_OpImpl_t));
    if (!impl) {
        free(n);
        return QNN_OP_PACKAGE_ERROR_GENERAL;
    }
    impl->opImplFn = q1NodeExecute;
    impl->userData = n;
    *opImpl = (QnnOpPackage_OpImpl_t)impl;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_FreeOpImpl(QnnOpPackage_OpImpl_t opImpl) {
    if (!opImpl) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    fprintf(stderr, "[Q1] freeOpImpl: opImpl=%p\n", (void*)opImpl);
    fflush(stderr);
    QnnCpuOpPackage_OpImpl_t* impl = (QnnCpuOpPackage_OpImpl_t*)opImpl;
    fprintf(stderr, "[Q1] freeOpImpl: userData=%p fn=%p (backend owns wrapper, freeing userData only)\n", impl->userData, (void*)impl->opImplFn);
    fflush(stderr);
    free(impl->userData);
    fprintf(stderr, "[Q1] freeOpImpl: done\n");
    fflush(stderr);
    return QNN_SUCCESS;
}

static int g_bonsaiPkgHandle = 0;

Qnn_ErrorHandle_t BonsaiQ1_Create(QnnOpPackage_GlobalInfrastructure_t infrastructure,
                                  QnnLog_Callback_t callback, QnnLog_Level_t maxLogLevel,
                                  Qnn_OpPackageHandle_t* opPackage) {
    (void)infrastructure; (void)callback; (void)maxLogLevel;
    if (!opPackage) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    *opPackage = (Qnn_OpPackageHandle_t)&g_bonsaiPkgHandle;
    return QNN_SUCCESS;
}

static bool bonsaiHandleOk(Qnn_OpPackageHandle_t h) {
    return h == (Qnn_OpPackageHandle_t)&g_bonsaiPkgHandle;
}

Qnn_ErrorHandle_t BonsaiQ1_ValidateOpConfigH(Qnn_OpPackageHandle_t opPackage,
                                             Qnn_OpConfig_t opConfig) {
    if (!bonsaiHandleOk(opPackage)) return QNN_OP_PACKAGE_ERROR_INVALID_HANDLE;
    return q1ValidateOpConfig(opConfig);
}

Qnn_ErrorHandle_t BonsaiQ1_CreateOpImplH(Qnn_OpPackageHandle_t opPackage,
                                        QnnOpPackage_GraphInfrastructure_t graphInfrastructure,
                                        QnnOpPackage_Node_t node,
                                        QnnOpPackage_OpImpl_t* opImpl) {
    if (!bonsaiHandleOk(opPackage)) return QNN_OP_PACKAGE_ERROR_INVALID_HANDLE;
    return BonsaiQ1_CreateOpImpl(graphInfrastructure, node, opImpl);
}

Qnn_ErrorHandle_t BonsaiQ1_FreeOpImplH(Qnn_OpPackageHandle_t opPackage,
                                       QnnOpPackage_OpImpl_t opImpl) {
    if (!bonsaiHandleOk(opPackage)) return QNN_OP_PACKAGE_ERROR_INVALID_HANDLE;
    return BonsaiQ1_FreeOpImpl(opImpl);
}

Qnn_ErrorHandle_t BonsaiQ1_Free(Qnn_OpPackageHandle_t opPackage) {
    if (!bonsaiHandleOk(opPackage)) return QNN_OP_PACKAGE_ERROR_INVALID_HANDLE;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_LogInitialize(QnnLog_Callback_t callback,
                                          QnnLog_Level_t maxLogLevel) {
    (void)callback; (void)maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_LogSetLevel(QnnLog_Level_t maxLogLevel) {
    (void)maxLogLevel;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t BonsaiQ1_LogTerminate(void) { return QNN_SUCCESS; }

Qnn_ErrorHandle_t BonsaiQ1_GetInterface(QnnOpPackage_Interface_t* intface_) {
    if (!intface_) return QNN_OP_PACKAGE_ERROR_INVALID_ARGUMENT;
    intface_->interfaceVersion.major = 1u;
    intface_->interfaceVersion.minor = 4u;
    intface_->interfaceVersion.patch = 0u;
    intface_->v1_4.init = BonsaiQ1_Init;
    intface_->v1_4.terminate = BonsaiQ1_Terminate;
    intface_->v1_4.getInfo = BonsaiQ1_GetInfo;
    intface_->v1_4.validateOpConfig = BonsaiQ1_ValidateOpConfig;
    intface_->v1_4.createOpImpl = BonsaiQ1_CreateOpImpl;
    intface_->v1_4.freeOpImpl = BonsaiQ1_FreeOpImpl;
    intface_->v1_4.logInitialize = BonsaiQ1_LogInitialize;
    intface_->v1_4.logSetLevel = BonsaiQ1_LogSetLevel;
    intface_->v1_4.logTerminate = BonsaiQ1_LogTerminate;
    return QNN_SUCCESS;
}

}  // extern "C"
