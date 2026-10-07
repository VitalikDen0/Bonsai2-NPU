// qnn_bigmlp: full-dim SwiGLU x[1,5120] -> rms -> gate/up[5120,17408] ->
// silu(g)*u -> down[17408,5120] -> y[1,5120] on HTP. Real-signal verify.
// Usage: qnn_bigmlp <libQnnHtp.so> [iters]
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

#define H 5120
#define FF 17408
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

#define N2(nm, ty, a, b, o) do { Qnn_Tensor_t _i[2] = {a, b}, _o[1] = {o}; \
    node(nm, ty, NULL, 0, _i, 2, _o, 1); } while (0)
#define N1(nm, ty, a, o) do { Qnn_Tensor_t _i[1] = {a}, _o[1] = {o}; \
    node(nm, ty, NULL, 0, _i, 1, _o, 1); } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_bigmlp <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t* xdata = NULL;
    static uint16_t* ydata = NULL;
    xdata = (uint16_t*)malloc((size_t)H * 2);
    ydata = (uint16_t*)malloc((size_t)H * 2);
    static uint16_t *wg, *wu, *wd;
    wg = (uint16_t*)malloc((size_t)H * FF * 2);
    wu = (uint16_t*)malloc((size_t)H * FF * 2);
    wd = (uint16_t*)malloc((size_t)FF * H * 2);
    if (!xdata || !ydata || !wg || !wu || !wd) { printf("OOM\n"); return 1; }
    unsigned seed = 31337;
    for (int i = 0; i < H; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < H * FF; i++) {
        seed = seed * 1664525u + 1013904223u;
        wg[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
        seed = seed * 1664525u + 1013904223u;
        wu[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < FF * H; i++) {
        seed = seed * 1664525u + 1013904223u;
        wd[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    printf("weights ready (534MB)\n");

    uint32_t dH[2] = {1, H}, dF[2] = {1, FF};
    uint32_t dHF[2] = {H, FF}, dFH[2] = {FF, H}, d11[2] = {1, 1}, dAx[1] = {1};
    static int32_t axesD[1] = {1};
    static uint16_t epsD[1];
    epsD[0] = f32_to_f16(1e-6f);

    Qnn_Tensor_t tX, tSq, tMn, tEp, tAd, tRr, tN, tAx;
    Qnn_Tensor_t tWg, tWu, tG, tU, tSg, tSilu, tM, tWd, tY;
    mktensor(&tX, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tSq, "sq", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tMn, "mn", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tEp, "eps", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, d11, epsD, sizeof(epsD));
    mktensor(&tAd, "ad", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tRr, "rr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tN, "n", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tAx, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dAx, axesD, sizeof(axesD));
    mktensor(&tWg, "wg", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHF, wg, (uint32_t)((size_t)H*FF*2));
    mktensor(&tWu, "wu", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHF, wu, (uint32_t)((size_t)H*FF*2));
    mktensor(&tG, "g", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tU, "u", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tSg, "sg", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tSilu, "silu", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tM, "m", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tWd, "wd", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dFH, wd, (uint32_t)((size_t)FF*H*2));
    mktensor(&tY, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);

    Qnn_Tensor_t* all[] = {&tX,&tSq,&tMn,&tEp,&tAd,&tRr,&tN,&tWg,&tWu,&tG,&tU,&tSg,&tSilu,&tM,&tWd,&tY};
    for (int i = 0; i < 16; i++) reg(all[i]);
    Qnn_Param_t meanP[2]; memset(meanP, 0, sizeof(meanP));
    meanP[0].paramType = QNN_PARAMTYPE_TENSOR; meanP[0].name = "axes";
    meanP[0].tensorParam = tAx;
    reg(&meanP[0].tensorParam);
    meanP[1].paramType = QNN_PARAMTYPE_SCALAR; meanP[1].name = "keep_dims";
    meanP[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; meanP[1].scalarParam.bool8Value = 1;

    Qnn_Param_t mm[2]; memset(mm, 0, sizeof(mm));
    mm[0].paramType = QNN_PARAMTYPE_SCALAR; mm[0].name = "transpose_in0";
    mm[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mm[0].scalarParam.bool8Value = 0;
    mm[1].paramType = QNN_PARAMTYPE_SCALAR; mm[1].name = "transpose_in1";
    mm[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; mm[1].scalarParam.bool8Value = 0;
    Qnn_Param_t npP[1]; memset(npP, 0, sizeof(npP));
    npP[0].paramType = QNN_PARAMTYPE_SCALAR; npP[0].name = "operation";
    npP[0].scalarParam.dataType = QNN_DATATYPE_UINT_32; npP[0].scalarParam.uint32Value = 6; // SIGMOID

    N2("sq", "ElementWiseMultiply", tX, tX, tSq);
    { Qnn_Tensor_t _i[1] = {tSq}, _o[1] = {tMn}; node("mn", "ReduceMean", meanP, 2, _i, 1, _o, 1); }
    N2("ad", "ElementWiseAdd", tMn, tEp, tAd);
    N1("rr", "ElementWiseRsqrt", tAd, tRr);
    N2("nm", "ElementWiseMultiply", tX, tRr, tN);
    { Qnn_Tensor_t _i[2] = {tN, tWg}, _o[1] = {tG}; node("gate", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN, tWu}, _o[1] = {tU}; node("up", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[1] = {tG}, _o[1] = {tSg}; node("sig", "ElementWiseNeuron", npP, 1, _i, 1, _o, 1); }
    N2("silu", "ElementWiseMultiply", tG, tSg, tSilu);
    N2("mul", "ElementWiseMultiply", tSilu, tU, tM);
    { Qnn_Tensor_t _i[2] = {tM, tWd}, _o[1] = {tY}; node("down", "MatMul", mm, 2, _i, 2, _o, 1); }

    CHK(Q->QNN_INTERFACE_VER_NAME.graphFinalize(G, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 2);
        tY.v1.clientBuf.data = ydata; tY.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 2);
        Qnn_Tensor_t ein[1] = {tX}, eout[1] = {tY};
        Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.graphExecute(G, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / (iters > 0 ? iters : 1);
    printf("EXEC OK iters=%d ms_per_run=%.2f GBps=%.2f\n", iters, ms, (534.0 / 1000.0) / (ms / 1000.0));

    // CPU fp32 reference
    float* xf = (float*)malloc((size_t)H * 4);
    float* n = (float*)malloc((size_t)H * 4);
    float* gv = (float*)malloc((size_t)FF * 4);
    float* uv = (float*)malloc((size_t)FF * 4);
    if (!xf || !n || !gv || !uv) { printf("OOM ref\n"); return 1; }
    for (int i = 0; i < H; i++) xf[i] = f16_to_f32(xdata[i]);
    double ss = 0; for (int i = 0; i < H; i++) ss += (double)xf[i] * xf[i];
    double rr = 1.0 / sqrt(ss / H + 1e-6);
    for (int i = 0; i < H; i++) n[i] = (float)(xf[i] * rr);
    for (int j = 0; j < FF; j++) {
        double a = 0, b = 0;
        for (int k = 0; k < H; k++) { a += (double)n[k] * f16_to_f32(wg[k * FF + j]); b += (double)n[k] * f16_to_f32(wu[k * FF + j]); }
        gv[j] = (float)a; uv[j] = (float)b;
    }
    int bad = 0; double maxerr = 0;
    for (int j = 0; j < 8; j++) {
        double a = 0;
        for (int k = 0; k < FF; k++) {
            double silu = gv[k] / (1.0 + exp(-gv[k]));
            a += silu * uv[k] * f16_to_f32(wd[k * H + j]);
        }
        double got = f16_to_f32(ydata[j]);
        double err = a - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.08 * (1.0 + (a < 0 ? -a : a));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, a, got); bad = 1; }
    }
    printf("maxerr8=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    Q->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    Q->QNN_INTERFACE_VER_NAME.deviceFree(device);
    Q->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
