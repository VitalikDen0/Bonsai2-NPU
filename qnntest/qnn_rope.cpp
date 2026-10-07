// qnn_rope: RotaryEmbedding (NeoX style, interleaved=0, rotary_dim=64) on HTP.
// X[1,1,1,256] + cos/sin[8,32] STATIC + pos int32 -> Y. theta=1e7.
// Usage: qnn_rope <libQnnHtp.so> [iters]
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

#define HDD 256
#define RDIM 64
#define RHALF 32
#define MAXP 8
#define POS 5
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

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_rope <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t xdata[HDD], ydata[HDD];
    static uint16_t cosd[MAXP * RHALF], sind[MAXP * RHALF];
    static int32_t posd[1] = {POS};
    unsigned seed = 2024;
    for (int i = 0; i < HDD; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.5f);
    }
    for (int p = 0; p < MAXP; p++) {
        for (int k = 0; k < RHALF; k++) {
            double ang = (double)p * pow(1e7, -(double)k / 32.0);
            cosd[p * RHALF + k] = f32_to_f16((float)cos(ang));
            sind[p * RHALF + k] = f32_to_f16((float)sin(ang));
        }
    }

    uint32_t dX[4] = {1, 1, 1, HDD}, dC[2] = {MAXP, RHALF}, dP[1] = {1};
    Qnn_Tensor_t tX, tCos, tSin, tPos, tY;
    mktensor(&tX, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 4, dX, NULL, 0);
    mktensor(&tCos, "cos", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dC, cosd, sizeof(cosd));
    mktensor(&tSin, "sin", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dC, sind, sizeof(sind));
    mktensor(&tPos, "pos", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dP, posd, sizeof(posd));
    mktensor(&tY, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 4, dX, NULL, 0);

    Qnn_Tensor_t* all[] = {&tX, &tCos, &tSin, &tPos, &tY};
    for (int i = 0; i < 5; i++) reg(all[i]);

    Qnn_Param_t rp[2]; memset(rp, 0, sizeof(rp));
    rp[0].paramType = QNN_PARAMTYPE_SCALAR; rp[0].name = "interleaved";
    rp[0].scalarParam.dataType = QNN_DATATYPE_BOOL_8; rp[0].scalarParam.bool8Value = 0;
    rp[1].paramType = QNN_PARAMTYPE_SCALAR; rp[1].name = "rotary_embedding_dim";
    rp[1].scalarParam.dataType = QNN_DATATYPE_UINT_32; rp[1].scalarParam.uint32Value = RDIM;

    Qnn_OpConfig_t op; memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "rope"; op.v1.packageName = "qti.aisw"; op.v1.typeName = "RotaryEmbedding";
    op.v1.numOfParams = 2; op.v1.params = rp;
    op.v1.numOfInputs = 4;
    Qnn_Tensor_t ins[4] = {tX, tCos, tSin, tPos}; op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    Qnn_Tensor_t outs[1] = {tY}; op.v1.outputTensors = outs;
    {
        Qnn_ErrorHandle_t ve = Q->QNN_INTERFACE_VER_NAME.backendValidateOpConfig(backend, op);
        printf("backendValidateOpConfig rc=%d\n", (int)ve);
    }
    CHK(Q->QNN_INTERFACE_VER_NAME.graphAddNode(G, op), "graphAddNode");
    CHK(Q->QNN_INTERFACE_VER_NAME.graphFinalize(G, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = sizeof(xdata);
        tY.v1.clientBuf.data = ydata; tY.v1.clientBuf.dataSize = sizeof(ydata);
        Qnn_Tensor_t ein[1] = {tX}, eout[1] = {tY};
        Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.graphExecute(G, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    printf("EXEC OK iters=%d ms_per_run=%.3f\n", iters, (t1 - t0) / (iters > 0 ? iters : 1));

    // CPU ref: NeoX pairs (k, k+32), theta=1e7, pos=POS; dims 64..255 passthrough
    int bad = 0; double maxerr = 0;
    for (int j = 0; j < 8; j++) {
        double ref;
        float xv = f16_to_f32(xdata[j]);
        if (j < 32) {
            double ang = (double)POS * pow(1e7, -(double)j / 32.0);
            double x1 = xv, x2 = f16_to_f32(xdata[j + 32]);
            ref = x1 * cos(ang) - x2 * sin(ang);
        } else {
            ref = xv;
        }
        double got = f16_to_f32(ydata[j]);
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.03 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    // also check second half j=32..39
    for (int j = 32; j < 40; j++) {
        double ang = (double)POS * pow(1e7, -(double)(j - 32) / 32.0);
        double x1 = f16_to_f32(xdata[j - 32]), x2 = f16_to_f32(xdata[j]);
        double ref = x1 * sin(ang) + x2 * cos(ang);
        double got = f16_to_f32(ydata[j]);
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.03 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr16=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    Q->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    Q->QNN_INTERFACE_VER_NAME.deviceFree(device);
    Q->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
