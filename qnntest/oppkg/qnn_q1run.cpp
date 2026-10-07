// qnn_q1run: CPU-backend runner for bonsai Q1MatMul OpPackage on REAL npubin bytes.
// Reads gate_proj (kind 0, [17408,5120]) from npubin on-device, converts scales fp16/2->fp32,
// builds graph x[1,5120]fp32 + bits[u8] + scales[fp32] -> Q1MatMul(bonsai) -> y[1,17408]fp32.
// Usage: qnn_q1run <libQnnCpu.so> <libbonsai_q1pkg.so> <npubin path> [iters]
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
#define NG 40
#define PROW 640
#define ITERS_DEFAULT 5

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
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

static uint32_t rd32(FILE* f) { uint32_t v; if (fread(&v, 1, 4, f) != 4) exit(9); return v; }
static uint64_t rd64(FILE* f) { uint64_t v; if (fread(&v, 1, 8, f) != 8) exit(9); return v; }

int main(int argc, char** argv) {
    if (argc < 4) { printf("usage: qnn_q1run <libQnnCpu.so> <pkg.so> <npubin> [iters]\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    int iters = argc > 4 ? atoi(argv[4]) : ITERS_DEFAULT;

    // ---- load npubin gate_proj bits+scales ----
    FILE* f = fopen(argv[3], "rb");
    if (!f) { printf("FOPEN FAIL %s\n", argv[3]); return 1; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NPU1", 4) != 0) { printf("BAD MAGIC\n"); return 1; }
    rd32(f); uint32_t n = rd32(f);
    const char* target = "language_model.model.layers.0.mlp.gate_proj";
    uint32_t out_dim = 0, in_dim = 0;
    uint64_t off = 0;
    int found = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint16_t nl;
        if (fread(&nl, 1, 2, f) != 2) { printf("HDR TRUNC\n"); return 1; }
        char* nm = (char*)malloc(nl + 1);
        if (fread(nm, 1, nl, f) != nl) { printf("HDR TRUNC2\n"); return 1; }
        nm[nl] = 0;
        int kind = fgetc(f);
        uint32_t od = rd32(f), id = rd32(f);
        uint64_t o = rd64(f);
        (void)rd64(f);
        if (strcmp(nm, target) == 0) { out_dim = od; in_dim = id; off = o; found = 1; }
        free(nm);
        if (found) break;
    }
    if (!found) { printf("TENSOR NOT FOUND\n"); return 1; }
    printf("gate_proj out=%u in=%u off=%llu\n", out_dim, in_dim, (unsigned long long)off);
    if (out_dim != FF || in_dim != H) { printf("UNEXPECTED SHAPE\n"); return 1; }
    uint16_t* sc16 = (uint16_t*)malloc((size_t)FF * NG * 2);
    unsigned char* bits = (unsigned char*)malloc((size_t)FF * PROW);
    float* scales = (float*)malloc((size_t)FF * NG * 4);
    float* xdata = (float*)malloc((size_t)H * 4);
    float* ydata = (float*)malloc((size_t)FF * 4);
    if (!sc16 || !bits || !scales || !xdata || !ydata) { printf("OOM\n"); return 1; }
    fseek(f, (long)off, SEEK_SET);
    if (fread(sc16, 1, (size_t)FF * NG * 2, f) != (size_t)FF * NG * 2) { printf("SCALES SHORT\n"); return 1; }
    if (fread(bits, 1, (size_t)FF * PROW, f) != (size_t)FF * PROW) { printf("BITS SHORT\n"); return 1; }
    fclose(f);
    for (int i = 0; i < FF * NG; i++) scales[i] = f16_to_f32(sc16[i]) * 0.5f;
    free(sc16);
    unsigned seed = 555;
    for (int i = 0; i < H; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = ((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f;
    }
    printf("weights loaded (14MB)\n");

    // ---- QNN CPU backend + package ----
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
    qnn->QNN_INTERFACE_VER_NAME.logCreate(qlog, QNN_LOG_LEVEL_DEBUG, &logger);
    CHK(qnn->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend), "backendCreate");
    Qnn_ErrorHandle_t re = qnn->QNN_INTERFACE_VER_NAME.backendRegisterOpPackage(
        backend, argv[2], "BonsaiQ1_GetInterface", NULL);
    printf("backendRegisterOpPackage rc=%d\n", (int)re);
    if (re != QNN_SUCCESS) {
        const char* m = NULL;
        if (qnn->QNN_INTERFACE_VER_NAME.errorGetMessage &&
            qnn->QNN_INTERFACE_VER_NAME.errorGetMessage(re, &m) == QNN_SUCCESS && m)
            printf("reg msg: %s\n", m);
        return 1;
    }
    Qnn_ContextHandle_t context = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.contextCreate(backend, NULL, NULL, &context), "contextCreate");
    Qnn_GraphHandle_t graph = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphCreate(context, "g", NULL, &graph), "graphCreate");

    uint32_t dX[2] = {1, H}, dB[2] = {FF, PROW}, dS[2] = {FF, NG}, dY[2] = {1, FF};
    Qnn_Tensor_t tX, tB, tS, tY;
    memset(&tX, 0, sizeof(tX)); memset(&tB, 0, sizeof(tB));
    memset(&tS, 0, sizeof(tS)); memset(&tY, 0, sizeof(tY));
    tX.version = QNN_TENSOR_VERSION_1;
    tX.v1.id = 0; tX.v1.name = "x"; tX.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    tX.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tX.v1.dataType = QNN_DATATYPE_FLOAT_32;
    tX.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tX.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tX.v1.rank = 2; tX.v1.dimensions = dX; tX.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tB.version = QNN_TENSOR_VERSION_1;
    tB.v1.id = 1; tB.v1.name = "bits"; tB.v1.type = QNN_TENSOR_TYPE_STATIC;
    tB.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tB.v1.dataType = QNN_DATATYPE_UINT_8;
    tB.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tB.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tB.v1.rank = 2; tB.v1.dimensions = dB; tB.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tB.v1.clientBuf.data = bits; tB.v1.clientBuf.dataSize = (uint32_t)((size_t)FF * PROW);
    tS.version = QNN_TENSOR_VERSION_1;
    tS.v1.id = 2; tS.v1.name = "scales"; tS.v1.type = QNN_TENSOR_TYPE_STATIC;
    tS.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tS.v1.dataType = QNN_DATATYPE_FLOAT_32;
    tS.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tS.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tS.v1.rank = 2; tS.v1.dimensions = dS; tS.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tS.v1.clientBuf.data = scales; tS.v1.clientBuf.dataSize = (uint32_t)((size_t)FF * NG * 4);
    tY.version = QNN_TENSOR_VERSION_1;
    tY.v1.id = 3; tY.v1.name = "y"; tY.v1.type = QNN_TENSOR_TYPE_APP_READ;
    tY.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tY.v1.dataType = QNN_DATATYPE_FLOAT_32;
    tY.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tY.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tY.v1.rank = 2; tY.v1.dimensions = dY; tY.v1.memType = QNN_TENSORMEMTYPE_RAW;

    Qnn_Tensor_t* all[] = {&tX, &tB, &tS, &tY};
    for (int i = 0; i < 4; i++) {
        if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, all[i]) != QNN_SUCCESS) {
            printf("FAIL tensorCreateGraphTensor t=%d\n", i);
            return 1;
        }
    }
    Qnn_OpConfig_t op; memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "q1";
    op.v1.packageName = "bonsai";
    op.v1.typeName = "Q1MatMul";
    op.v1.numOfParams = 0; op.v1.params = NULL;
    op.v1.numOfInputs = 3;
    Qnn_Tensor_t ins[3] = {tX, tB, tS}; op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    Qnn_Tensor_t outs[1] = {tY}; op.v1.outputTensors = outs;
    {
        Qnn_ErrorHandle_t ve = qnn->QNN_INTERFACE_VER_NAME.backendValidateOpConfig(backend, op);
        printf("backendValidateOpConfig rc=%d\n", (int)ve);
    }
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op), "graphAddNode");
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 4);
        tY.v1.clientBuf.data = ydata; tY.v1.clientBuf.dataSize = (uint32_t)((size_t)FF * 4);
        Qnn_Tensor_t ein[1] = {tX}, eout[1] = {tY};
        Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / (iters > 0 ? iters : 1);
    printf("EXEC OK iters=%d ms_per_run=%.2f\n", iters, ms);

    // fp64 reference on real bytes
    int bad = 0; double maxerr = 0;
    for (int j = 0; j < 8; j++) {
        long double acc = 0;
        for (int g = 0; g < NG; g++) {
            long double s = scales[j * NG + g];
            long double gs = 0;
            for (int k = 0; k < 128; k++) {
                int jj = g * 128 + k;
                int bit = (bits[(size_t)j * PROW + (jj >> 3)] >> (jj & 7)) & 1;
                gs += bit ? (long double)xdata[jj] : -(long double)xdata[jj];
            }
            acc += s * gs;
        }
        double ref = (double)acc;
        double got = ydata[j];
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 1e-3 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.6f got=%.6f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr8=%.3e %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    qnn->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    qnn->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
