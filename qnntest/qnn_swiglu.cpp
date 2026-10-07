// qnn_swiglu: x[1,HD] -> gate=x@Wg, up=x@Wu [1,FF]; out=(silu(gate)*up)@Wd -> [1,HD] on HTP.
// Usage: qnn_swiglu <libQnnHtp.so> [iters]
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

#define HD 1024
#define FF 2048
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

static void mmPrm(Qnn_Param_t* p) {
    memset(p, 0, 2 * sizeof(Qnn_Param_t));
    p[0].paramType = QNN_PARAMTYPE_SCALAR; p[0].name = "transpose_in0";
    p[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; p[0].scalarParam.bool8Value = 0;
    p[1].paramType = QNN_PARAMTYPE_SCALAR; p[1].name = "transpose_in1";
    p[1].scalarParam.dataType = QNN_DATATYPE_BOOL_8; p[1].scalarParam.bool8Value = 0;
}

static void addnode(const QnnInterface_t* qnn, Qnn_GraphHandle_t graph,
                    const char* name, const char* type,
                    Qnn_Param_t* prm, uint32_t nprm,
                    Qnn_Tensor_t* ins, uint32_t nin,
                    Qnn_Tensor_t* outs, uint32_t nout) {
    Qnn_OpConfig_t op; memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = name; op.v1.packageName = "qti.aisw"; op.v1.typeName = type;
    op.v1.numOfParams = nprm; op.v1.params = prm;
    op.v1.numOfInputs = nin; op.v1.inputTensors = ins;
    op.v1.numOfOutputs = nout; op.v1.outputTensors = outs;
    Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op);
    if (e != QNN_SUCCESS) { printf("FAIL graphAddNode %s err=%d\n", name, (int)e); exit(1); }
}

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_swiglu <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t xdata[HD], ydata[HD];
    static uint16_t *wg, *wu, *wd;
    wg = (uint16_t*)malloc((size_t)HD * FF * 2);
    wu = (uint16_t*)malloc((size_t)HD * FF * 2);
    wd = (uint16_t*)malloc((size_t)FF * HD * 2);
    if (!wg || !wu || !wd) { printf("OOM\n"); return 1; }
    unsigned seed = 99;
    for (int i = 0; i < HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 1.0f);
    }
    for (int i = 0; i < HD * FF; i++) {
        seed = seed * 1664525u + 1013904223u;
        wg[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
        seed = seed * 1664525u + 1013904223u;
        wu[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < FF * HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        wd[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }

    uint32_t dH[2] = {1, HD}, dF[2] = {1, FF}, dG[2] = {HD, FF}, dD[2] = {FF, HD};
    Qnn_Tensor_t tX, tWg, tWu, tG, tU, tSg, tSilu, tM, tWd, tY;
    mktensor(&tX, 0, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tWg, 1, "wg", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dG, wg, (uint32_t)((size_t)HD*FF*2));
    mktensor(&tWu, 2, "wu", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dG, wu, (uint32_t)((size_t)HD*FF*2));
    mktensor(&tG, 3, "g", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tU, 4, "u", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tSg, 5, "sg", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tSilu, 6, "silu", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tM, 7, "m", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dF, NULL, 0);
    mktensor(&tWd, 8, "wd", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dD, wd, (uint32_t)((size_t)FF*HD*2));
    mktensor(&tY, 9, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);

    Qnn_Tensor_t* all[] = {&tX, &tWg, &tWu, &tG, &tU, &tSg, &tSilu, &tM, &tWd, &tY};
    for (int i = 0; i < 10; i++) {
        if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, all[i]) != QNN_SUCCESS) {
            printf("FAIL tensorCreateGraphTensor t=%d\n", i);
            return 1;
        }
    }

    Qnn_Param_t mm[2]; mmPrm(mm);
    Qnn_Tensor_t i1[2] = {tX, tWg}, o1[1] = {tG};
    addnode(qnn, graph, "gate", "MatMul", mm, 2, i1, 2, o1, 1);
    Qnn_Tensor_t i2[2] = {tX, tWu}, o2[1] = {tU};
    addnode(qnn, graph, "up", "MatMul", mm, 2, i2, 2, o2, 1);
    // sg = sigmoid(g)
    Qnn_Param_t np[1]; memset(np, 0, sizeof(np));
    np[0].paramType = QNN_PARAMTYPE_SCALAR; np[0].name = "operation";
    np[0].scalarParam.dataType = QNN_DATATYPE_UINT_32; np[0].scalarParam.uint32Value = 6; // SIGMOID
    Qnn_Tensor_t i3[1] = {tG}, o3[1] = {tSg};
    addnode(qnn, graph, "sig", "ElementWiseNeuron", np, 1, i3, 1, o3, 1);
    // silu = g * sg ; m = silu * u ; y = m @ wd
    Qnn_Tensor_t i4[2] = {tG, tSg}, o4[1] = {tSilu};
    addnode(qnn, graph, "silu", "ElementWiseMultiply", NULL, 0, i4, 2, o4, 1);
    Qnn_Tensor_t i5[2] = {tSilu, tU}, o5[1] = {tM};
    addnode(qnn, graph, "mul", "ElementWiseMultiply", NULL, 0, i5, 2, o5, 1);
    Qnn_Tensor_t i6[2] = {tM, tWd}, o6[1] = {tY};
    addnode(qnn, graph, "down", "MatMul", mm, 2, i6, 2, o6, 1);

    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = sizeof(xdata);
        tY.v1.clientBuf.data = ydata; tY.v1.clientBuf.dataSize = sizeof(ydata);
        Qnn_Tensor_t ein[1] = {tX}, eout[1] = {tY};
        Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    printf("EXEC OK iters=%d ms_per_run=%.3f\n", iters, (t1 - t0) / (iters > 0 ? iters : 1));

    float xf[HD];
    for (int i = 0; i < HD; i++) xf[i] = f16_to_f32(xdata[i]);
    double gf[FF], uf[FF];
    for (int j = 0; j < FF; j++) {
        double g = 0, u = 0;
        for (int k = 0; k < HD; k++) {
            g += (double)xf[k] * f16_to_f32(wg[k * FF + j]);
            u += (double)xf[k] * f16_to_f32(wu[k * FF + j]);
        }
        gf[j] = g; uf[j] = u;
    }
    int bad = 0; double maxerr = 0;
    for (int j = 0; j < 4; j++) {
        double m = 0;
        for (int k = 0; k < FF; k++) {
            double silu = gf[k] / (1.0 + exp(-gf[k]));
            m += silu * uf[k] * f16_to_f32(wd[k * HD + j]);
        }
        double got = f16_to_f32(ydata[j]);
        double err = m - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.03 * (1.0 + (m < 0 ? -m : m));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, m, got); bad = 1; }
    }
    printf("maxerr4=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    qnn->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    qnn->QNN_INTERFACE_VER_NAME.deviceFree(device);
    qnn->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
