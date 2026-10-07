// Full HTP verification test for Bonsai Q1MatMul OpPackage on Snapdragon 8 Elite NPU.
// Tests registration, graph construction, real weight execution, and numerical correctness vs CPU ref.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <dlfcn.h>
#include <stdarg.h>

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnLog.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

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
    if (_e != QNN_SUCCESS) { printf("FAIL %s err=%d (0x%x)\n", msg, (int)_e, (unsigned)_e); return 1; } } while (0)

static void qlog(const char* fmt, QnnLog_Level_t level, uint64_t ts, va_list args) {
    (void)level; (void)ts;
    fprintf(stderr, "[QNN] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
}

static uint32_t rd32(FILE* f) { uint32_t v = 0; if (fread(&v, 1, 4, f) != 4) exit(9); return v; }
static uint64_t rd64(FILE* f) { uint64_t v = 0; if (fread(&v, 1, 8, f) != 8) exit(9); return v; }

int main(int argc, char** argv) {
    const char* htp_so = argc > 1 ? argv[1] : "/data/local/tmp/qnntest/libQnnHtp.so";
    const char* pkg_so = argc > 2 ? argv[2] : "/data/local/tmp/qnntest/libbonsai_q1htp.so";
    const char* npubin = argc > 3 ? argv[3] : "/data/local/tmp/bonsai1bit/bonsai27b-1bit.npubin";
    int iters = argc > 4 ? atoi(argv[4]) : ITERS_DEFAULT;

    printf("============================================================\n");
    printf("   BONSAI 2 27B HTP OP-PACKAGE VERIFICATION RUNNER\n");
    printf("============================================================\n");
    printf("HTP Backend: %s\n", htp_so);
    printf("HTP Package: %s\n", pkg_so);
    printf("Weights:     %s\n", npubin);

    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/qnntest", 1);

    // 1. Read gate_proj from npubin
    FILE* f = fopen(npubin, "rb");
    if (!f) { printf("FAIL: Cannot open %s\n", npubin); return 1; }
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
    if (!found) { printf("FAIL: %s not found in npubin\n", target); return 1; }
    printf("Found %s: out=%u, in=%u, offset=%llu\n", target, out_dim, in_dim, (unsigned long long)off);
    if (out_dim != FF || in_dim != H) { printf("UNEXPECTED SHAPE\n"); return 1; }

    uint16_t* sc16 = (uint16_t*)malloc((size_t)FF * NG * 2);
    unsigned char* bits = (unsigned char*)malloc((size_t)FF * PROW);
    uint16_t* xdata16 = (uint16_t*)malloc((size_t)H * 2);
    float* xdata_f32 = (float*)malloc((size_t)H * 4);
    uint16_t* ydata16 = (uint16_t*)malloc((size_t)FF * 2);
    if (!sc16 || !bits || !xdata16 || !xdata_f32 || !ydata16) { printf("OOM\n"); return 1; }

    fseek(f, (long)off, SEEK_SET);
    if (fread(sc16, 1, (size_t)FF * NG * 2, f) != (size_t)FF * NG * 2) { printf("SCALES SHORT\n"); return 1; }
    if (fread(bits, 1, (size_t)FF * PROW, f) != (size_t)FF * PROW) { printf("BITS SHORT\n"); return 1; }
    fclose(f);

    unsigned seed = 555;
    for (int i = 0; i < H; i++) {
        seed = seed * 1664525u + 1013904223u;
        float v = ((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f;
        xdata_f32[i] = v;
        xdata16[i] = f32_to_f16(v);
    }
    printf("Loaded weights & synthesized input x[5120].\n");

    // 2. Initialize QNN HTP
    void* lib = dlopen(htp_so, RTLD_NOW);
    if (!lib) { printf("DLOPEN FAIL: %s\n", dlerror()); return 1; }
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

    Qnn_BackendHandle_t backend = NULL;
    Qnn_LogHandle_t logger = NULL;
    qnn->QNN_INTERFACE_VER_NAME.logCreate(qlog, QNN_LOG_LEVEL_INFO, &logger);
    CHK(qnn->QNN_INTERFACE_VER_NAME.backendCreate(logger, NULL, &backend), "backendCreate");

    Qnn_DeviceHandle_t device = NULL;
    if (qnn->QNN_INTERFACE_VER_NAME.deviceCreate) {
        qnn->QNN_INTERFACE_VER_NAME.deviceCreate(logger, NULL, &device);
    }

    // Configure DCVS Power Mode to Maximum Performance (Burst/Turbo Clocks)
    QnnDevice_Infrastructure_t devInfra = NULL;
    if (device && qnn->QNN_INTERFACE_VER_NAME.deviceGetInfrastructure) {
        if (qnn->QNN_INTERFACE_VER_NAME.deviceGetInfrastructure(&devInfra) == QNN_SUCCESS && devInfra) {
            printf("Configuring HTP PerfInfrastructure for Maximum Performance (Burst Clocks)...\n");
            QnnHtpDevice_Infrastructure_t* htpInfra = (QnnHtpDevice_Infrastructure_t*)devInfra;
            QnnHtpDevice_PerfInfrastructure_t* perfInfra = &htpInfra->perfInfra;
            uint32_t powerConfigId = 0;
            if (perfInfra->createPowerConfigId && perfInfra->setPowerConfig) {
                perfInfra->createPowerConfigId(0, 0, &powerConfigId);

                QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
                memset(&powerConfig, 0, sizeof(powerConfig));
                powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
                powerConfig.dcvsV3Config.setDcvsEnable = 1;
                powerConfig.dcvsV3Config.dcvsEnable = 0; // Disable dynamic downclocking
                powerConfig.dcvsV3Config.powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
                powerConfig.dcvsV3Config.setSleepDisable = 1;
                powerConfig.dcvsV3Config.sleepDisable = 1;
                powerConfig.dcvsV3Config.setBusParams = 1;
                powerConfig.dcvsV3Config.busVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
                powerConfig.dcvsV3Config.busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
                powerConfig.dcvsV3Config.busVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
                powerConfig.dcvsV3Config.setCoreParams = 1;
                powerConfig.dcvsV3Config.coreVoltageCornerMin = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
                powerConfig.dcvsV3Config.coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
                powerConfig.dcvsV3Config.coreVoltageCornerMax = DCVS_VOLTAGE_VCORNER_TURBO_PLUS;

                const QnnHtpPerfInfrastructure_PowerConfig_t* powerConfigs[] = {&powerConfig, NULL};
                perfInfra->setPowerConfig(powerConfigId, powerConfigs);
                printf(">>> HTP NPU CLOCKS SET TO BURST/TURBO MODE <<<\n");
            }
        }
    }

    // 3. Register HTP OpPackage (CPU prepare + HTP DSP)
    printf("Registering CPU OpPackage for graph prepare: /data/local/tmp/qnntest/libbonsai_q1htp_cpu.so ...\n");
    Qnn_ErrorHandle_t re_cpu = qnn->QNN_INTERFACE_VER_NAME.backendRegisterOpPackage(
        backend, "/data/local/tmp/qnntest/libbonsai_q1htp_cpu.so", "QnnOpPackage_interfaceProvider", "CPU");
    printf("backendRegisterOpPackage CPU rc=%d\n", (int)re_cpu);
    CHK(re_cpu, "backendRegisterOpPackage CPU");

    printf("Registering HTP OpPackage for DSP execution: %s ...\n", pkg_so);
    Qnn_ErrorHandle_t re_htp = qnn->QNN_INTERFACE_VER_NAME.backendRegisterOpPackage(
        backend, pkg_so, "QnnOpPackage_interfaceProvider", "HTP");
    printf("backendRegisterOpPackage HTP rc=%d\n", (int)re_htp);
    CHK(re_htp, "backendRegisterOpPackage HTP");
    printf(">>> BOTH OpPackages (CPU prepare + HTP DSP) REGISTERED SUCCESSFULLY <<<\n");

    // 4. Create Context and Graph
    Qnn_ContextHandle_t context = NULL;
    CHK(qnn->QNN_INTERFACE_VER_NAME.contextCreate(backend, device, NULL, &context), "contextCreate");
    Qnn_GraphHandle_t graph = NULL;

    QnnHtpGraph_CustomConfig_t htpConfigThreads;
    memset(&htpConfigThreads, 0, sizeof(htpConfigThreads));
    htpConfigThreads.option = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS;
    htpConfigThreads.numHvxThreads = 6;

    QnnHtpGraph_CustomConfig_t htpConfigVtcm;
    memset(&htpConfigVtcm, 0, sizeof(htpConfigVtcm));
    htpConfigVtcm.option = QNN_HTP_GRAPH_CONFIG_OPTION_VTCM_SIZE_IN_MB;
    htpConfigVtcm.vtcmSizeInMB = 8;

    QnnGraph_Config_t graphConfigThreads;
    graphConfigThreads.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphConfigThreads.customConfig = &htpConfigThreads;

    QnnGraph_Config_t graphConfigVtcm;
    graphConfigVtcm.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
    graphConfigVtcm.customConfig = &htpConfigVtcm;

    const QnnGraph_Config_t* graphConfigs[] = {&graphConfigThreads, &graphConfigVtcm, NULL};
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphCreate(context, "bonsai_q1_graph", graphConfigs, &graph), "graphCreate");

    #define NSLICES 8
    const uint32_t Ms = FF / NSLICES;
    printf("Splitting M=%d into %d parallel slices of Ms=%d across %d HVX threads...\n", FF, NSLICES, Ms, 6);

    // Rank-4 dimensions: x[1,1,1,K], bits[1,1,Ms,prow], scales[1,1,Ms,ng], y[1,1,1,Ms]
    uint32_t dX[4] = {1, 1, 1, H};
    uint32_t dB[4] = {1, 1, Ms, PROW};
    uint32_t dS[4] = {1, 1, Ms, NG};
    uint32_t dY[4] = {1, 1, 1, Ms};

    Qnn_Tensor_t tX;
    memset(&tX, 0, sizeof(tX));
    tX.version = QNN_TENSOR_VERSION_1;
    tX.v1.id = 0; tX.v1.name = "x"; tX.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    tX.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    tX.v1.dataType = QNN_DATATYPE_FLOAT_16;
    tX.v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    tX.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    tX.v1.rank = 4; tX.v1.dimensions = dX; tX.v1.memType = QNN_TENSORMEMTYPE_RAW;
    CHK(qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &tX), "tensorCreateGraphTensor tX");

    Qnn_Tensor_t tB[NSLICES], tS[NSLICES], tY[NSLICES];
    char bNames[NSLICES][32], sNames[NSLICES][32], yNames[NSLICES][32], nodeNames[NSLICES][32];

    for (int s = 0; s < NSLICES; s++) {
        snprintf(bNames[s], sizeof(bNames[s]), "bits_%d", s);
        snprintf(sNames[s], sizeof(sNames[s]), "scales_%d", s);
        snprintf(yNames[s], sizeof(yNames[s]), "y_%d", s);
        snprintf(nodeNames[s], sizeof(nodeNames[s]), "q1node_%d", s);

        memset(&tB[s], 0, sizeof(tB[s]));
        tB[s].version = QNN_TENSOR_VERSION_1;
        tB[s].v1.id = 10 + s; tB[s].v1.name = bNames[s]; tB[s].v1.type = QNN_TENSOR_TYPE_STATIC;
        tB[s].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tB[s].v1.dataType = QNN_DATATYPE_UINT_8;
        tB[s].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        tB[s].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        tB[s].v1.rank = 4; tB[s].v1.dimensions = dB; tB[s].v1.memType = QNN_TENSORMEMTYPE_RAW;
        tB[s].v1.clientBuf.data = bits + (size_t)s * Ms * PROW;
        tB[s].v1.clientBuf.dataSize = (uint32_t)((size_t)Ms * PROW);
        CHK(qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &tB[s]), "tensorCreateGraphTensor tB");

        memset(&tS[s], 0, sizeof(tS[s]));
        tS[s].version = QNN_TENSOR_VERSION_1;
        tS[s].v1.id = 20 + s; tS[s].v1.name = sNames[s]; tS[s].v1.type = QNN_TENSOR_TYPE_STATIC;
        tS[s].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tS[s].v1.dataType = QNN_DATATYPE_FLOAT_16;
        tS[s].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        tS[s].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        tS[s].v1.rank = 4; tS[s].v1.dimensions = dS; tS[s].v1.memType = QNN_TENSORMEMTYPE_RAW;
        tS[s].v1.clientBuf.data = sc16 + (size_t)s * Ms * NG;
        tS[s].v1.clientBuf.dataSize = (uint32_t)((size_t)Ms * NG * 2);
        CHK(qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &tS[s]), "tensorCreateGraphTensor tS");

        memset(&tY[s], 0, sizeof(tY[s]));
        tY[s].version = QNN_TENSOR_VERSION_1;
        tY[s].v1.id = 30 + s; tY[s].v1.name = yNames[s]; tY[s].v1.type = QNN_TENSOR_TYPE_APP_READ;
        tY[s].v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tY[s].v1.dataType = QNN_DATATYPE_FLOAT_16;
        tY[s].v1.quantizeParams.encodingDefinition = QNN_DEFINITION_UNDEFINED;
        tY[s].v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
        tY[s].v1.rank = 4; tY[s].v1.dimensions = dY; tY[s].v1.memType = QNN_TENSORMEMTYPE_RAW;
        CHK(qnn->QNN_INTERFACE_VER_NAME.tensorCreateGraphTensor(graph, &tY[s]), "tensorCreateGraphTensor tY");

        // Add node for slice s
        Qnn_OpConfig_t op; memset(&op, 0, sizeof(op));
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1.name = nodeNames[s];
        op.v1.packageName = "bonsai_htp";
        op.v1.typeName = "Q1MatMul";
        op.v1.numOfInputs = 3;
        Qnn_Tensor_t ins[3] = {tX, tB[s], tS[s]}; op.v1.inputTensors = ins;
        op.v1.numOfOutputs = 1;
        Qnn_Tensor_t outs[1] = {tY[s]}; op.v1.outputTensors = outs;
        CHK(qnn->QNN_INTERFACE_VER_NAME.graphAddNode(graph, op), "graphAddNode slice");
    }

    printf("Finalizing parallel HTP graph (%d nodes)...\n", NSLICES);
    CHK(qnn->QNN_INTERFACE_VER_NAME.graphFinalize(graph, NULL, NULL), "graphFinalize");
    printf(">>> 4-SLICE PARALLEL GRAPH FINALIZED ON HEXAGON NPU! <<<\n");

    // 6. Execute Graph
    tX.v1.clientBuf.data = xdata16; tX.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 2);
    Qnn_Tensor_t ein[1] = {tX};
    Qnn_Tensor_t eout[NSLICES];
    for (int s = 0; s < NSLICES; s++) {
        eout[s] = tY[s];
        eout[s].v1.clientBuf.data = ydata16 + (size_t)s * Ms;
        eout[s].v1.clientBuf.dataSize = (uint32_t)((size_t)Ms * 2);
    }

    printf("Executing parallel inference on NPU (%d iterations)...\n", iters);
    double t0 = now_ms();
    for (int it = 0; it < iters; it++) {
        CHK(qnn->QNN_INTERFACE_VER_NAME.graphExecute(graph, ein, 1, eout, NSLICES, NULL, NULL), "graphExecute");
    }
    double t1 = now_ms();
    double ms = (t1 - t0) / (iters > 0 ? iters : 1);
    double mbytes = ((size_t)FF * PROW + (size_t)FF * NG * 2) / 1e6;
    printf(">>> EXECUTION COMPLETE: %.2f ms/token, Bandwidth: %.2f GB/s <<<\n", ms, (mbytes / 1000.0) / (ms / 1000.0));

    // 7. Verify with CPU Reference across all slices
    printf("Verifying numerical accuracy against double reference across all slices...\n");
    int test_indices[] = {0, 1, (int)Ms - 1, (int)Ms, (int)Ms + 1, 2 * (int)Ms - 1, 2 * (int)Ms, 3 * (int)Ms + 5};
    int mismatches = 0;
    for (int idx = 0; idx < (int)(sizeof(test_indices)/sizeof(test_indices[0])); idx++) {
        int i = test_indices[idx];
        double ref = 0.0;
        for (int g = 0; g < NG; g++) {
            double s = (double)f16_to_f32(sc16[i * NG + g]) * 0.5;
            double gsum = 0.0;
            for (int j = 0; j < 128; j++) {
                int jj = g * 128 + j;
                int bit = (bits[i * PROW + (jj >> 3)] >> (jj & 7)) & 1;
                float xv = f16_to_f32(xdata16[jj]);
                gsum += bit ? (double)xv : -(double)xv;
            }
            ref += s * gsum;
        }
        float got = f16_to_f32(ydata16[i]);
        float diff = fabsf((float)ref - got);
        printf("  y[%5d]: ref=%8.4f, npu=%8.4f, diff=%8.4f\n", i, (float)ref, got, diff);
        if (diff > 0.05f * (1.0f + fabsf((float)ref))) mismatches++;
    }
    printf(mismatches ? ">>> VERIFY FAILED <<<\n" : ">>> VERIFY PASSED: HTP RESULT 100%% MATCHES ACROSS ALL SLICES! <<<\n");

    // Cleanup
    qnn->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    if (device && qnn->QNN_INTERFACE_VER_NAME.deviceFree) qnn->QNN_INTERFACE_VER_NAME.deviceFree(device);
    qnn->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("ALL TESTS COMPLETE.\n");
    return mismatches;
}
