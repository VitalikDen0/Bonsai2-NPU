// qnn_attn: single-head decode attention q[1,256] @ K[64,256]^T -> softmax -> @ V[64,256] on HTP.
// Usage: qnn_attn <libQnnHtp.so path> [iters]
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnLog.h"
#include <stdarg.h>

#define HD 256
#define CTX 64
#define ITERS_DEFAULT 5

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static uint16_t f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t s = (u >> 16) & 0x8000;
    int e = (int)((u >> 23) & 0xff) - 112;
    uint32_t m = u & 0x7fffff;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7bff);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t f = (e == 0) ? (s << 31) : (e == 31) ? ((s << 31) | 0x7f800000u | (m << 13))
                                                  : ((s << 31) | ((e + 112) << 23) | (m << 13));
    float r;
    memcpy(&r, &f, 4);
    return r;
}

#define CHK(fn, msg) do { Qnn_ErrorHandle_t _e = (fn); \
    if (_e != QNN_SUCCESS) { printf("FAIL %s err=%d\n", msg, (int)_e); return 1; } } while (0)

static void qlog(const char* fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
    (void)level; (void)ts;
    fprintf(stderr, "[QNN] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

static void mktensor(Qnn_Tensor_t* t, uint32_t id, const char* name, Qnn_TensorType_t type,
                     Qnn_DataType_t dt, uint32_t rank, uint32_t* dims,
                     void* data, uint32_t dataSize) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    t->v1.id = id; t->v1.name = name; t->v1.type = type;
    t->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->v1.dataType = dt;
    t->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t->v1.rank = rank; t->v1.dimensions = dims; t->v1.memType = QNN_TENSORMEMTYPE_RAW;
    t->v1.clientBuf.data = data;
    t->v1.clientBuf.dataSize = dataSize;
}

static void mmPrm(Qnn_Param_t* p, const char* n0, uint8_t t0, const char* n1, uint8_t t1) {
    memset(p, 0, 2 * sizeof(Qnn_Param_t));
    p[0].paramType = QNN_PARAMTYPE_SCALAR; p[0].name = n0;
    p[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; p[0].scalarParam.bool8Value = t0;
    p[1].paramType = QNN_PARAMTYPE_SCALAR; p[1].name = n1;
    p[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; p[1].scalarParam.bool8Value = t1;
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_attn <libQnnHtp.so> [iters]\n"); return 2; }
    int iters = argc > 2 ? atoi(argv[2]) : ITERS_DEFAULT;
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/qnntest", 1);

    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { printf("DLOPEN FAIL %s\n", dlerror()); return 1; }
    typedef Qnn_ErrorHandle_t (*pfnGet)(const QnnInterface_t***, uint32_t*);
    pfnGet getProviders = (pfnGet)dlsym(lib, "QnnInterface_getProviders");
    if (!getProviders) { printf("DLSYM FAIL\n"); return 1; }
    const QnnInterface_t** plist = NULL;
    uint32_t nprov = 0;
    CHK(getProviders(&plist, &nprov), "getProviders");
    const QnnInterface_t* qnn = NULL;
    for (uint32_t i = 0; i < nprov; i++) {
        if (plist[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) { qnn = plist[i]; break; }
    }
    if (!qnn && nprov > 0) qnn = plist[0];
    if (!qnn) { printf("NO PROVIDER\n"); return 1; }
    printf("backend: %s api %u.%u\n", qnn->providerName ? qnn->providerName : "?",
           qnn->apiVersion.coreApiVersion.major, qnn->apiVersion.coreApiVersion.minor);

    Qnn_BackendHandle_t backend = NULL;
    Qnn_LogHandle_t logger = NULL;
    qnn->QNN_INTERFACE_VER_NAME.logCreate(qlog, QNN_LOG_LEVEL_WARN, &logger);
    CHK(qnn->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend), "backendCreate");
    Qnn_DeviceHandle_t device = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.deviceCreate(logger, NULL, &device), "deviceCreate");
    Qnn_ContextHandle_t context = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, NULL, &context), "contextCreate");
    Qnn_GraphHandle_t graph = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphCreate(context, "g", NULL, &graph), "graphCreate");

    static uint16_t qdata[HD], odata[HD];
    static uint16_t* kdata = NULL;
    static uint16_t* vdata = NULL;
    kdata = (uint16_t*)malloc((size_t)CTX * HD * 2);
    vdata = (uint16_t*)malloc((size_t)CTX * HD * 2);
    if (!kdata || !vdata) { printf("OOM kv\n"); return 1; }
    unsigned seed = 4242;
    for (int i = 0; i < HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        qdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < CTX * HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        kdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
        seed = seed * 1664525u + 1013904223u;
        vdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }

    uint32_t dQ[2] = {1, HD}, dK[2] = {CTX, HD}, dS[2] = {1, CTX};
    Qnn_Tensor_t tQ, tK, tS, tP, tV, tO;
    mktensor(&tQ, 0, "q", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dQ, NULL, 0);
    mktensor(&tK, 1, "k", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dK,
             kdata, (uint32_t)((size_t)CTX * HD * 2));
    mktensor(&tS, 2, "s", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tP, 3, "p", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tV, 4, "v", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dK,
             vdata, (uint32_t)((size_t)CTX * HD * 2));
    mktensor(&tO, 5, "o", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dQ, NULL, 0);

    Qnn_Tensor_t* all[] = {&tQ, &tK, &tS, &tP, &tV, &tO};
    for (int i = 0; i < 6; i++) {
        if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, all[i]) != QNN_SUCCESS) {
            printf("FAIL tensorCreateGraphTensor t=%d\n", i);
            return 1;
        }
    }

    // node 1: s = MatMul(q, k^T)
    Qnn_Param_t mm1P[2];
    mmPrm(mm1P, "transpose_in0", 0, "transpose_in1", 1);
    Qnn_OpConfig_t op1; memset(&op1, 0, sizeof(op1));
    op1.version = QNN_OPCONFIG_VERSION_1;
    op1.v1.name = "qk"; op1.v1.packageName = "qti.aisw"; op1.v1.typeName = "MatMul";
    op1.v1.numOfParams = 2; op1.v1.params = mm1P;
    op1.v1.numOfInputs = 2;
    Qnn_Tensor_t in1[2] = {tQ, tK}; op1.v1.inputTensors = in1;
    op1.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou1[1] = {tS}; op1.v1.outputTensors = ou1;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op1), "graphAddNode qk");

    // node 2: p = Softmax(s, axis=1)
    Qnn_Param_t smP[1];
    memset(smP, 0, sizeof(smP));
    smP[0].paramType = QNN_PARAMTYPE_SCALAR; smP[0].name = "axis";
    smP[0].scalarParam.dataType = QNN_DATATYPE_INT_32; smP[0].scalarParam.int32Value = 1;
    Qnn_OpConfig_t op2; memset(&op2, 0, sizeof(op2));
    op2.version = QNN_OPCONFIG_VERSION_1;
    op2.v1.name = "sm"; op2.v1.packageName = "qti.aisw"; op2.v1.typeName = "Softmax";
    op2.v1.numOfParams = 1; op2.v1.params = smP;
    op2.v1.numOfInputs = 1;
    Qnn_Tensor_t in2[1] = {tS}; op2.v1.inputTensors = in2;
    op2.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou2[1] = {tP}; op2.v1.outputTensors = ou2;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op2), "graphAddNode sm");

    // node 3: o = MatMul(p, v)
    Qnn_Param_t mm2P[2];
    mmPrm(mm2P, "transpose_in0", 0, "transpose_in1", 0);
    Qnn_OpConfig_t op3; memset(&op3, 0, sizeof(op3));
    op3.version = QNN_OPCONFIG_VERSION_1;
    op3.v1.name = "pv"; op3.v1.packageName = "qti.aisw"; op3.v1.typeName = "MatMul";
    op3.v1.numOfParams = 2; op3.v1.params = mm2P;
    op3.v1.numOfInputs = 2;
    Qnn_Tensor_t in3[2] = {tP, tV}; op3.v1.inputTensors = in3;
    op3.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou3[1] = {tO}; op3.v1.outputTensors = ou3;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op3), "graphAddNode pv");

    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tQ.v1.clientBuf.data = qdata;
        tQ.v1.clientBuf.dataSize = sizeof(qdata);
        tO.v1.clientBuf.data = odata;
        tO.v1.clientBuf.dataSize = sizeof(odata);
        Qnn_Tensor_t ein[1] = {tQ};
        Qnn_Tensor_t eout[1] = {tO};
        Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / (iters > 0 ? iters : 1);
    printf("EXEC OK iters=%d ms_per_run=%.3f\n", iters, ms);

    // CPU reference (fp32, no scale — must match graph)
    float qf[HD];
    for (int i = 0; i < HD; i++) qf[i] = f16_to_f32(qdata[i]);
    double sc[CTX], mx = -1e300;
    for (int c = 0; c < CTX; c++) {
        double s = 0;
        for (int k = 0; k < HD; k++) s += (double)qf[k] * f16_to_f32(kdata[c * HD + k]);
        sc[c] = s; if (s > mx) mx = s;
    }
    double psum = 0;
    for (int c = 0; c < CTX; c++) { sc[c] = exp(sc[c] - mx); psum += sc[c]; }
    for (int c = 0; c < CTX; c++) sc[c] /= psum;
    int bad = 0;
    double maxerr = 0;
    for (int j = 0; j < 4; j++) {
        double ref = 0;
        for (int c = 0; c < CTX; c++) ref += sc[c] * f16_to_f32(vdata[c * HD + j]);
        double got = f16_to_f32(odata[j]);
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.05 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr4=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    qnn->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    qnn->QNN_INTERFACE_VER_NAME.deviceFree(device);
    qnn->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
