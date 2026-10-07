// qnn_gemv: single fp16 MatMul [1,5120]x[5120,5120] on HTP, timed + verified.
// Usage: qnn_gemv <libQnnHtp.so path> [iters]
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
#define ITERS_DEFAULT 20

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

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_gemv <libQnnHtp.so> [iters]\n"); return 2; }
    int iters = argc > 2 ? atoi(argv[3 - 1]) : ITERS_DEFAULT;
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/qnntest", 1);

    void* lib = dlopen(argv[1], RTLD_NOW);
    if (!lib) { printf("DLOPEN FAIL %s\n", dlerror()); return 1; }
    typedef Qnn_ErrorHandle_t (*pfnGet)(const QnnInterface_t***, uint32_t*);
    pfnGet getProviders = (pfnGet)dlsym(lib, "QnnInterface_getProviders");
    if (!getProviders) { printf("DLSYM FAIL\n"); return 1; }
    const QnnInterface_t** plist = NULL;
    uint32_t nprov = 0;
    CHK(getProviders(&plist, &nprov), "getProviders");
    printf("providers=%u\n", nprov);
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
    if (qnn->QNN_INTERFACE_VER_NAME.logCreate(qlog, QNN_LOG_LEVEL_DEBUG, &logger) != QNN_SUCCESS) {
        printf("LOG CREATE FAIL (continuing without logger)\n");
    }
    CHK(qnn->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend), "backendCreate");
    Qnn_DeviceHandle_t device = NULL;
    int needDevice = (qnn->providerName == NULL) ||
                     (strstr(qnn->providerName, "CPU") == NULL);
    if (needDevice) {
        CHK(qnn->QNN_INTERFACE_VER_NAME.deviceCreate(logger, NULL, &device), "deviceCreate");
    }
    Qnn_ContextHandle_t context = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, NULL, &context), "contextCreate");
    Qnn_GraphHandle_t graph = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphCreate(context, "g", NULL, &graph), "graphCreate");

    static uint16_t xdata[DIM], ydata[DIM];
    unsigned seed = 12345;
    for (int i = 0; i < DIM; i++) {
        seed = seed * 1664525u + 1013904223u;
        float v = ((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.02f;
        xdata[i] = f32_to_f16(v);
    }

    uint32_t dimsX[2] = {1, DIM}, dimsY[2] = {1, DIM};
    Qnn_Tensor_t tX, tY;
    memset(&tX, 0, sizeof(tX));
    memset(&tY, 0, sizeof(tY));
    tX.version = QNN_TENSOR_VERSION_1;
    tX.v1.id = 0; tX.v1.name = "x"; tX.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    tX.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tX.v1.dataType = QNN_DATATYPE_FLOAT_16;
    tX.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tX.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tX.v1.rank = 2; tX.v1.dimensions = dimsX; tX.v1.memType = QNN_TENSORMEMTYPE_RAW;
    tY.version = QNN_TENSOR_VERSION_1;
    tY.v1.id = 1; tY.v1.name = "y"; tY.v1.type = QNN_TENSOR_TYPE_APP_READ;
    tY.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tY.v1.dataType = QNN_DATATYPE_FLOAT_16;
    tY.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tY.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tY.v1.rank = 2; tY.v1.dimensions = dimsY; tY.v1.memType = QNN_TENSORMEMTYPE_RAW;
    Qnn_OpConfig_t op;    memset(&op, 0, sizeof(op));
    op.version = QNN_OPCONFIG_VERSION_1;
    op.v1.name = "mm";
    op.v1.packageName = "qti.aisw";
    op.v1.typeName = "ElementWiseNeuron";
    Qnn_Param_t np[1]; memset(np, 0, sizeof(np));
    np[0].paramType = QNN_PARAMTYPE_SCALAR; np[0].name = "operation";
    np[0].scalarParam.dataType = QNN_DATATYPE_UINT_32; np[0].scalarParam.uint32Value = 6; // SIGMOID
    op.v1.numOfParams = 1;
    op.v1.params = np;
    // wire tensors: Neuron has single input {x}, output {y}
    op.v1.numOfInputs = 1;
    Qnn_Tensor_t ins[1] = {tX};
    op.v1.inputTensors = ins;
    op.v1.numOfOutputs = 1;
    Qnn_Tensor_t outs[1] = {tY};
    op.v1.outputTensors = outs;
    // canonical: register first, addNode with backend-ID structs
    for (int i = 0; i < 1; i++) {
        if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &ins[i]) != QNN_SUCCESS) {
            printf("FAIL tensorCreateGraphTensor in=%d\n", i);
            return 1;
        }
    }
    if (qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &outs[0]) != QNN_SUCCESS) {
        printf("FAIL tensorCreateGraphTensor out\n");
        return 1;
    }
    {
        Qnn_ErrorHandle_t ve = qnn->QNN_INTERFACE_VER_NAME.backendValidateOpConfig(backend, op);
        printf("backendValidateOpConfig rc=%d\n", (int)ve);
        if (ve != QNN_SUCCESS && qnn->QNN_INTERFACE_VER_NAME.errorGetMessage) {
            const char* m = NULL;
            if (qnn->QNN_INTERFACE_VER_NAME.errorGetMessage(ve, &m) == QNN_SUCCESS && m)
                printf("validate msg: %s\n", m);
            if (qnn->QNN_INTERFACE_VER_NAME.errorGetVerboseMessage) {
                const char* vm = NULL;
                if (qnn->QNN_INTERFACE_VER_NAME.errorGetVerboseMessage(ve, &vm) == QNN_SUCCESS && vm) {
                    printf("validate verbose: %s\n", vm);
                    qnn->QNN_INTERFACE_VER_NAME.errorFreeVerboseMessage(vm);
                }
            }
        }
    }
    {
        Qnn_ErrorHandle_t ae = qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op);
        printf("graphAddNode rc=%d\n", (int)ae);
        if (ae != QNN_SUCCESS) {
            if (qnn->QNN_INTERFACE_VER_NAME.errorGetMessage) {
                const char* m = NULL;
                if (qnn->QNN_INTERFACE_VER_NAME.errorGetMessage(ae, &m) == QNN_SUCCESS && m)
                    printf("addNode msg: %s\n", m);
            }
            printf("FAIL graphAddNode err=%d\n", (int)ae);
            return 1;
        }
    }
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    // reference y[0..3] on CPU fp32
    float xref[DIM];
    for (int i = 0; i < DIM; i++) xref[i] = f16_to_f32(xdata[i]);
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        ins[0].v1.clientBuf.data = xdata;
        ins[0].v1.clientBuf.dataSize = sizeof(xdata);
        outs[0].v1.clientBuf.data = ydata;
        outs[0].v1.clientBuf.dataSize = sizeof(ydata);
        Qnn_ErrorHandle_t e = qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ins, 1, outs, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / iters;
    // 52MB weights streamed per run (26M fp16)
    printf("EXEC OK iters=%d ms_per_run=%.2f\n", iters, ms);
    int bad = 0;
    double maxerr = 0;
    for (int j = 0; j < 8; j++) {
        float xv = xref[j];
        double ref = 1.0 / (1.0 + exp(-(double)xv));
        double got = f16_to_f32(ydata[j]);
        double err = ref - got;
        if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.02 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr8=%.5f ", maxerr);
    printf(bad ? "VERIFY FAIL\n" : "VERIFY PASS\n");
    qnn->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    qnn->QNN_INTERFACE_VER_NAME.deviceFree(device);
    qnn->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
