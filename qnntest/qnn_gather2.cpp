// qnn_gather: Gather at scale — x[1,1024] -> y1[1,256] (idx 0..255), y2[1,896] (idx 128..1023).
// Usage: qnn_gather <libQnnHtp.so> [iters]
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

#define N 1024
#define N1 256
#define N2 896
#define ITERS_DEFAULT 3

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

static const QnnInterface_t* Q = NULL;
static Qnn_GraphHandle_t G = NULL;
static uint32_t NID = 0;

static void mktensor(Qnn_Tensor_t* t, const char* name, Qnn_TensorType_t type,
                     Qnn_DataType_t dt, uint32_t rank, uint32_t* dims,
                     void* data, uint32_t dataSize) {
    memset(t, 0, sizeof(*t));
    t->version = QNN_TENSOR_VERSION_1;
    t->v1.id = NID++; t->v1.name = name; t->v1.type = type;
    t->v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    t->v1.dataType = dt;
    t->v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    t->v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    t->v1.rank = rank; t->v1.dimensions = dims; t->v1.memType = QNN_TENSORMEMTYPE_RAW;
    t->v1.clientBuf.data = data;
    t->v1.clientBuf.dataSize = dataSize;
}

static void reg(Qnn_Tensor_t* t) {
    Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(G, t);
    if (e != QNN_SUCCESS) { printf("FAIL reg %s err=%d\n", t->v1.name, (int)e); exit(1); }
}

static void node(const char* name, const char* type,
                 Qnn_Param_t* prm, uint32_t nprm,
                 Qnn_Tensor_t* ins, uint32_t nin,
                 Qnn_Tensor_t* outs, uint32_t nout) {
    Qnn_OpConfig_t op; memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = name; op.v1.packageName = "qti.aisw"; op.v1.typeName = type;
    op.v1.numOfParams = nprm; op.v1.params = prm;
    op.v1.numOfInputs = nin; op.v1.inputTensors = ins;
    op.v1.numOfOutputs = nout; op.v1.outputTensors = outs;
    Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.graphAddNode(G, op);
    if (e != QNN_SUCCESS) { printf("FAIL graphAddNode %s err=%d\n", name, (int)e); exit(1); }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_gather <libQnnHtp.so> [iters]\n"); return 2; }
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
    for (uint32_t i = 0; i < nprov; i++) {
        if (plist[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR) { Q = plist[i]; break; }
    }
    if (!Q && nprov > 0) Q = plist[0];
    if (!Q) { printf("NO PROVIDER\n"); return 1; }

    Qnn_BackendHandle_t backend = NULL;
    Qnn_LogHandle_t logger = NULL;
    Q->QNN_INTERFACE_VER_NAME.logCreate(qlog, QNN_LOG_LEVEL_WARN, &logger);
    CHK(Q->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend), "backendCreate");
    Qnn_DeviceHandle_t device = NULL;
    CHK(Q->QNN_INTERFACE_VER_NAME.deviceCreate(logger, NULL, &device), "deviceCreate");
    Qnn_ContextHandle_t context = NULL;
    CHK(Q->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, NULL, &context), "contextCreate");
    CHK(Q->QNN_INTERFACE_VER_NAME.graphCreate(context, "g", NULL, &G), "graphCreate");

    static uint16_t xdata[N], y1data[N1], y2data[N2];
    static int32_t idx1[N1], idx2[N2];
    static uint16_t* wdata = NULL;
    wdata = (uint16_t*)malloc((size_t)N * N * 2);
    if (!wdata) { printf("OOM weights\n"); return 1; }
    unsigned seed = 31337;
    for (int i = 0; i < N; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.5f);
    }
    for (int i = 0; i < N * N; i++) {
        seed = seed * 1664525u + 1013904223u;
        wdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < N1; i++) idx1[i] = i;
    for (int i = 0; i < N2; i++) idx2[i] = 128 + i;

    uint32_t dX[2] = {1, N}, dY1[2] = {1, N1}, dY2[2] = {1, N2};
    uint32_t dI1[1] = {N1}, dI2[1] = {N2};
    uint32_t dW[2] = {N, N};
    Qnn_Tensor_t tX, tI1, tI2, tY1, tY2, tW, tM;
    mktensor(&tX, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dX, NULL, 0);
    mktensor(&tI1, "i1", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dI1, idx1, sizeof(idx1));
    mktensor(&tI2, "i2", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dI2, idx2, sizeof(idx2));
    mktensor(&tW, "w", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dW, wdata, (uint32_t)((size_t)N*N*2));
    mktensor(&tM, "m", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dX, NULL, 0);
    mktensor(&tY1, "y1", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dY1, NULL, 0);
    mktensor(&tY2, "y2", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dY2, NULL, 0);
    Qnn_Tensor_t* all[] = {&tX, &tI1, &tI2, &tW, &tM, &tY1, &tY2};
    for (int i = 0; i < 7; i++) reg(all[i]);

    Qnn_Param_t axP[1]; memset(axP, 0, sizeof(axP));
    axP[0].paramType = QNN_PARAMTYPE_SCALAR; axP[0].name = "axis";
    axP[0].scalarParam.dataType = QNN_DATATYPE_INT_32; axP[0].scalarParam.int32Value = 1;
    Qnn_Param_t mm[2]; memset(mm, 0, sizeof(mm));
    mm[0].paramType = QNN_PARAMTYPE_SCALAR; mm[0].name = "transpose_in0";
    mm[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mm[0].scalarParam.bool8Value = 0;
    mm[1].paramType = QNN_PARAMTYPE_SCALAR; mm[1].name = "transpose_in1";
    mm[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mm[1].scalarParam.bool8Value = 0;
    { Qnn_Tensor_t _i[2] = {tX, tW}, _o[1] = {tM}; node("mm", "MatMul", mm, 2, _i, 2, _o, 1); }

    Qnn_OpConfig_t op1; memset(&op1, 0, sizeof(op1));
    op1.version = QNN_OPCONFIG_VERSION_1;
    op1.v1.name = "g1"; op1.v1.packageName = "qti.aisw"; op1.v1.typeName = "Gather";
    op1.v1.numOfParams = 1; op1.v1.params = axP;
    op1.v1.numOfInputs = 2;
    Qnn_Tensor_t in1[2] = {tM, tI1}; op1.v1.inputTensors = in1;
    op1.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou1[1] = {tY1}; op1.v1.outputTensors = ou1;
    CHK(Q->QNN_INTERFACE_VER_NAME.graphAddNode(G, op1), "graphAddNode g1");

    Qnn_OpConfig_t op2; memset(&op2, 0, sizeof(op2));
    op2.version = QNN_OPCONFIG_VERSION_1;
    op2.v1.name = "g2"; op2.v1.packageName = "qti.aisw"; op2.v1.typeName = "Gather";
    op2.v1.numOfParams = 1; op2.v1.params = axP;
    op2.v1.numOfInputs = 2;
    Qnn_Tensor_t in2[2] = {tM, tI2}; op2.v1.inputTensors = in2;
    op2.v1.numOfOutputs = 1;
    Qnn_Tensor_t ou2[1] = {tY2}; op2.v1.outputTensors = ou2;
    CHK(Q->QNN_INTERFACE_VER_NAME.graphAddNode(G, op2), "graphAddNode g2");

    CHK(Q->QNN_INTERFACE_VER_NAME.graphFinalize(G, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = sizeof(xdata);
        tY1.v1.clientBuf.data = y1data; tY1.v1.clientBuf.dataSize = sizeof(y1data);
        tY2.v1.clientBuf.data = y2data; tY2.v1.clientBuf.dataSize = sizeof(y2data);
        Qnn_Tensor_t ein[1] = {tX}, eout[2] = {tY1, tY2};
        Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.graphExecute(G, ein, 1, eout, 2, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    printf("EXEC OK iters=%d ms_per_run=%.3f\n", iters, (t1 - t0) / (iters > 0 ? iters : 1));

    int bad = 0; double me1 = 0, me2 = 0;
    static float mf[N];
    for (int j = 0; j < N; j++) {
        double a = 0;
        for (int k = 0; k < N; k++) a += (double)f16_to_f32(xdata[k]) * f16_to_f32(wdata[k * N + j]);
        mf[j] = (float)a;
    }
    for (int j = 0; j < N1; j++) {
        double ref = mf[j];
        double got = f16_to_f32(y1data[j]);
        double e = ref - got; if (e < 0) e = -e;
        if (e > me1) me1 = e;
        if (e > 0.02 * (1.0 + (ref < 0 ? -ref : ref))) { if (bad < 4) printf("MISMATCH1 j=%d ref=%.5f got=%.5f\n", j, ref, got); bad++; }
    }
    for (int j = 0; j < N2; j++) {
        double ref = mf[128 + j];
        double got = f16_to_f32(y2data[j]);
        double e = ref - got; if (e < 0) e = -e;
        if (e > me2) me2 = e;
        if (e > 0.02 * (1.0 + (ref < 0 ? -ref : ref))) { if (bad < 8) printf("MISMATCH2 j=%d ref=%.5f got=%.5f\n", j, ref, got); bad++; }
    }
    printf("maxerr256=%.5f maxerr896=%.5f %s\n", me1, me2, bad ? "VERIFY FAIL" : "VERIFY PASS");
    Q->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    Q->QNN_INTERFACE_VER_NAME.deviceFree(device);
    Q->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad ? 1 : 0;
}
