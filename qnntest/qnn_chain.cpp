// qnn_chain: fp16 chain [1,5120] -> MatMul(w1) -> RmsNorm -> MatMul(w2) -> [1,5120] on HTP.
// Usage: qnn_chain <libQnnHtp.so path> [iters]
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

#define DIM 5120
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

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_chain <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t xdata[DIM], ydata[DIM];
    static uint16_t* w1data = NULL;
    static uint16_t* w2data = NULL;
    w1data = (uint16_t*)malloc((size_t)DIM * DIM * 2);
    w2data = (uint16_t*)malloc((size_t)DIM * DIM * 2);
    if (!w1data || !w2data) { printf("OOM weights\n"); return 1; }
    unsigned seed = 777;
    for (int i = 0; i < DIM; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.02f);
    }
    for (int i = 0; i < DIM * DIM; i++) {
        seed = seed * 1664525u + 1013904223u;
        w1data[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.02f);
        seed = seed * 1664525u + 1013904223u;
        w2data[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.02f);
    }

    uint32_t dVec[2] = {1, DIM}, dMat[2] = {DIM, DIM}, dAx[1] = {1}, d11[2] = {1, 1};
    static int32_t axesData[1] = {1};
    static uint16_t epsData[1];
    epsData[0] = f32_to_f16(1e-5f);
    Qnn_Tensor_t tX, tW1, tH, tSq, tMean, tEps, tAdd, tRr, tN, tW2, tY, tAxes;
    mktensor(&tX, 0, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dVec, NULL, 0);
    mktensor(&tW1, 1, "w1", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dMat,
             w1data, (uint32_t)((size_t)DIM * DIM * 2));
    mktensor(&tH, 2, "h", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dVec, NULL, 0);
    mktensor(&tSq, 3, "sq", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dVec, NULL, 0);
    mktensor(&tMean, 4, "mean", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tEps, 5, "eps", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, d11,
             epsData, sizeof(epsData));
    mktensor(&tAdd, 6, "add", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tRr, 7, "rr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tN, 8, "n", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dVec, NULL, 0);
    mktensor(&tW2, 9, "w2", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dMat,
             w2data, (uint32_t)((size_t)DIM * DIM * 2));
    mktensor(&tY, 10, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dVec, NULL, 0);
    mktensor(&tAxes, 11, "rms_axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dAx,
             axesData, sizeof(axesData));

    // register all graph tensors (converter flow)
    Qnn_Tensor_t* all[] = {&tX, &tW1, &tH, &tSq, &tMean, &tEps, &tAdd, &tRr, &tN, &tW2, &tY};
    for (int i = 0; i < 11; i++) {
        if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, all[i]) != QNN_SUCCESS) {
            printf("FAIL tensorCreateGraphTensor t=%d\n", i);
            return 1;
        }
    }

    // node 1: mm1 = MatMul(x, w1)
    Qnn_Param_t mmP[2];
    memset(mmP, 0, sizeof(mmP));
    mmP[0].paramType = QNN_PARAMTYPE_SCALAR; mmP[0].name = "transpose_in0";
    mmP[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mmP[0].scalarParam.bool8Value = 0;
    mmP[1].paramType = QNN_PARAMTYPE_SCALAR; mmP[1].name = "transpose_in1";
    mmP[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mmP[1].scalarParam.bool8Value = 0;
    Qnn_OpConfig_t op1; memset(&op1, 0, sizeof(op1));
    op1.version = QNN_OPCONFIG_VERSION_1;
    op1.v1.name = "mm1"; op1.v1.packageName = "qti.aisw"; op1.v1.typeName = "MatMul";
    op1.v1.numOfParams = 2; op1.v1.params = mmP;
    op1.v1.numOfInputs = 2;
    Qnn_Tensor_t in1[2] = {tX, tW1}; op1.v1.inputTensors = in1;
    op1.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou1[1] = {tH}; op1.v1.outputTensors = ou1;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op1), "graphAddNode mm1");

    // RMS subgraph: sq=h*h; mean=ReduceMean(sq); add=mean+eps; rr=rsqrt(add); n=h*rr
    // node 2: sq = ElementWiseMultiply(h, h)
    Qnn_OpConfig_t op2; memset(&op2, 0, sizeof(op2));
    op2.version = QNN_OPCONFIG_VERSION_1;
    op2.v1.name = "sq"; op2.v1.packageName = "qti.aisw"; op2.v1.typeName = "ElementWiseMultiply";
    op2.v1.numOfParams = 0; op2.v1.params = NULL;
    op2.v1.numOfInputs = 2;
    Qnn_Tensor_t in2[2] = {tH, tH}; op2.v1.inputTensors = in2;
    op2.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou2[1] = {tSq}; op2.v1.outputTensors = ou2;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op2), "graphAddNode sq");

    // node 3: mean = ReduceMean(sq; axes, keep_dims=true)
    Qnn_Param_t meanP[2];
    memset(meanP, 0, sizeof(meanP));
    meanP[0].paramType = QNN_PARAMTYPE_TENSOR; meanP[0].name = "axes";
    meanP[0].tensorParam = tAxes;
    if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &meanP[0].tensorParam) != QNN_SUCCESS) {
        printf("FAIL tensorCreateGraphTensor axes\n");
        return 1;
    }
    meanP[1].paramType = QNN_PARAMTYPE_SCALAR; meanP[1].name = "keep_dims";
    meanP[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; meanP[1].scalarParam.bool8Value = 1;
    Qnn_OpConfig_t op3; memset(&op3, 0, sizeof(op3));
    op3.version = QNN_OPCONFIG_VERSION_1;
    op3.v1.name = "mean"; op3.v1.packageName = "qti.aisw"; op3.v1.typeName = "ReduceMean";
    op3.v1.numOfParams = 2; op3.v1.params = meanP;
    op3.v1.numOfInputs = 1;
    Qnn_Tensor_t in3[1] = {tSq}; op3.v1.inputTensors = in3;
    op3.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou3[1] = {tMean}; op3.v1.outputTensors = ou3;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op3), "graphAddNode mean");

    // node 4: add = ElementWiseAdd(mean, eps)
    Qnn_OpConfig_t op4; memset(&op4, 0, sizeof(op4));
    op4.version = QNN_OPCONFIG_VERSION_1;
    op4.v1.name = "add"; op4.v1.packageName = "qti.aisw"; op4.v1.typeName = "ElementWiseAdd";
    op4.v1.numOfParams = 0; op4.v1.params = NULL;
    op4.v1.numOfInputs = 2;
    Qnn_Tensor_t in4[2] = {tMean, tEps}; op4.v1.inputTensors = in4;
    op4.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou4[1] = {tAdd}; op4.v1.outputTensors = ou4;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op4), "graphAddNode add");

    // node 5: rr = ElementWiseRsqrt(add)
    Qnn_OpConfig_t op5; memset(&op5, 0, sizeof(op5));
    op5.version = QNN_OPCONFIG_VERSION_1;
    op5.v1.name = "rr"; op5.v1.packageName = "qti.aisw"; op5.v1.typeName = "ElementWiseRsqrt";
    op5.v1.numOfParams = 0; op5.v1.params = NULL;
    op5.v1.numOfInputs = 1;
    Qnn_Tensor_t in5[1] = {tAdd}; op5.v1.inputTensors = in5;
    op5.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou5[1] = {tRr}; op5.v1.outputTensors = ou5;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op5), "graphAddNode rr");

    // node 6: n = ElementWiseMultiply(h, rr)  [broadcast 1x5120 * 1x1]
    Qnn_OpConfig_t op6; memset(&op6, 0, sizeof(op6));
    op6.version = QNN_OPCONFIG_VERSION_1;
    op6.v1.name = "norm"; op6.v1.packageName = "qti.aisw"; op6.v1.typeName = "ElementWiseMultiply";
    op6.v1.numOfParams = 0; op6.v1.params = NULL;
    op6.v1.numOfInputs = 2;
    Qnn_Tensor_t in6[2] = {tH, tRr}; op6.v1.inputTensors = in6;
    op6.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou6[1] = {tN}; op6.v1.outputTensors = ou6;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op6), "graphAddNode norm");

    // node 7: y = MatMul(n, w2)
    Qnn_OpConfig_t op7; memset(&op7, 0, sizeof(op7));
    op7.version = QNN_OPCONFIG_VERSION_1;
    op7.v1.name = "mm2"; op7.v1.packageName = "qti.aisw"; op7.v1.typeName = "MatMul";
    op7.v1.numOfParams = 2; op7.v1.params = mmP;
    op7.v1.numOfInputs = 2;
    Qnn_Tensor_t in7[2] = {tN, tW2}; op7.v1.inputTensors = in7;
    op7.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou7[1] = {tY}; op7.v1.outputTensors = ou7;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op7), "graphAddNode mm2");

    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata;
        tX.v1.clientBuf.dataSize = sizeof(xdata);
        tY.v1.clientBuf.data = ydata;
        tY.v1.clientBuf.dataSize = sizeof(ydata);
        Qnn_Tensor_t ein[1] = {tX};
        Qnn_Tensor_t eout[1] = {tY};
        Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / (iters > 0 ? iters : 1);
    printf("EXEC OK iters=%d ms_per_run=%.2f\n", iters, ms);

    // CPU reference: h = x@w1 (fp32), n = rms(h), y = n@w2
    float xf[DIM];
    for (int i = 0; i < DIM; i++) xf[i] = f16_to_f32(xdata[i]);
    int bad = 0;
    double maxerr = 0;
    for (int j = 0; j < DIM; j++) {
        double h = 0;
        for (int k = 0; k < DIM; k++) h += (double)xf[k] * f16_to_f32(w1data[k * DIM + j]);
        // rms over full row needed; approximate per-element check via full recompute below
        (void)h;
    }
    // full-row reference for first 4 outputs
    double href[DIM], nref[DIM];
    for (int j = 0; j < DIM; j++) {
        double h = 0;
        for (int k = 0; k < DIM; k++) h += (double)xf[k] * f16_to_f32(w1data[k * DIM + j]);
        href[j] = h;
    }
    double ss = 0;
    for (int j = 0; j < DIM; j++) ss += href[j] * href[j];
    double rr = 1.0 / sqrt(ss / DIM + 1e-5);
    for (int j = 0; j < DIM; j++) nref[j] = href[j] * rr;
    for (int j = 0; j < 4; j++) {
        double ref = 0;
        for (int k = 0; k < DIM; k++) ref += nref[k] * f16_to_f32(w2data[k * DIM + j]);
        double got = f16_to_f32(ydata[j]);
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
