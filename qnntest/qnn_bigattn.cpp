// qnn_bigattn: full-width GQA decode step (4 q-heads share 1 kv-head), H=5120 HD=256.
// x -> rms -> q[1,1024]/k,v[1,256] -> fused flat RoPE (Gather idx, theta 1e7, rot 64) ->
// per-head Gather -> scores=q@Kf^T (Kf=Concat[Kc,k], ctx 8+1) -> softmax -> @Vf ->
// hcat -> @Wo -> +x -> y. Usage: qnn_bigattn <libQnnHtp.so> [iters]
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
#define HD 256
#define NQ 4
#define RDIM 64
#define RHALF 32
#define RPASS_Q (NQ * HD - 2 * NQ * RHALF)
#define RPASS_K (HD - RDIM)
#define CTX 8
#define CTF (CTX + 1)
#define POS 5
#define ITERS_DEFAULT 2

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

#define N2(nm, ty, a, b, o, prm, nprm) do { Qnn_Tensor_t _i[2] = {a, b}, _o[1] = {o}; \
    node(nm, ty, prm, nprm, _i, 2, _o, 1); } while (0)
#define N1(nm, ty, a, o) do { Qnn_Tensor_t _i[1] = {a}, _o[1] = {o}; \
    node(nm, ty, NULL, 0, _i, 1, _o, 1); } while (0)
#define N3(nm, ty, a, b, c, o, prm, nprm) do { Qnn_Tensor_t _i[3] = {a, b, c}, _o[1] = {o}; \
    node(nm, ty, prm, nprm, _i, 3, _o, 1); } while (0)

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: qnn_bigattn <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t *xdata, *ydata;
    xdata = (uint16_t*)malloc((size_t)H * 2);
    ydata = (uint16_t*)malloc((size_t)H * 2);
    static uint16_t *wq, *wk, *wv, *wo, *kc, *vc;
    wq = (uint16_t*)malloc((size_t)H * NQ * HD * 2);
    wk = (uint16_t*)malloc((size_t)H * HD * 2);
    wv = (uint16_t*)malloc((size_t)H * HD * 2);
    wo = (uint16_t*)malloc((size_t)(NQ * HD) * H * 2);
    kc = (uint16_t*)malloc((size_t)CTX * HD * 2);
    vc = (uint16_t*)malloc((size_t)CTX * HD * 2);
    if (!xdata || !ydata || !wq || !wk || !wv || !wo || !kc || !vc) { printf("OOM\n"); return 1; }
    unsigned seed = 424242;
    for (int i = 0; i < H; i++) {
        seed = seed * 1664525u + 1013904223u;
        xdata[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < H * NQ * HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        wq[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < H * HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        wk[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
        seed = seed * 1664525u + 1013904223u;
        wv[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < NQ * HD * H; i++) {
        seed = seed * 1664525u + 1013904223u;
        wo[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    for (int i = 0; i < CTX * HD; i++) {
        seed = seed * 1664525u + 1013904223u;
        kc[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
        seed = seed * 1664525u + 1013904223u;
        vc[i] = f32_to_f16(((float)(seed >> 8) / 8388608.0f - 1.0f) * 0.05f);
    }
    printf("weights ready\n");

    // fused flat RoPE index sets (NeoX split-half, per head rotary 64 of 256)
    static int32_t qA[NQ * RHALF], qB[NQ * RHALF], qP[RPASS_Q];
    static int32_t kA[RHALF], kB[RHALF], kP[RPASS_K];
    static int32_t hI[NQ][HD];
    for (int h = 0; h < NQ; h++)
        for (int k = 0; k < RHALF; k++) { qA[h * RHALF + k] = h * HD + k; qB[h * RHALF + k] = h * HD + k + RHALF; }
    { int n = 0; for (int i = 0; i < NQ * HD; i++) { int hh = i / HD, kk = i % HD; if (kk >= RDIM) qP[n++] = i; } }
    for (int k = 0; k < RHALF; k++) { kA[k] = k; kB[k] = k + RHALF; }
    for (int k = 0; k < RPASS_K; k++) kP[k] = k + RDIM;
    // per-head gather from fused flat roped qr layout [y1_all(128)|y2_all(128)|pass_all(768)]:
    // head h -> y1_h=[h*32,h*32+32), y2_h=[128+h*32,+32), pass_h=[256+h*192,+192)
    for (int h = 0; h < NQ; h++) {
        for (int k = 0; k < RHALF; k++) hI[h][k] = h * RHALF + k;
        for (int k = 0; k < RHALF; k++) hI[h][RHALF + k] = NQ * RHALF + h * RHALF + k;
        for (int j = 0; j < HD - RDIM; j++) hI[h][RDIM + j] = 2 * NQ * RHALF + h * (HD - RDIM) + j;
    }
    static uint16_t cosd[RHALF], sind[RHALF];
    for (int k = 0; k < RHALF; k++) {
        double ang = (double)POS * pow(1e7, -(double)k / 32.0);
        cosd[k] = f32_to_f16((float)cos(ang));
        sind[k] = f32_to_f16((float)sin(ang));
    }
    // tiled copy for fused flat q-rope parts [1,128] (same angles every head)
    static uint16_t cosQ[NQ * RHALF], sinQ[NQ * RHALF];
    for (int h = 0; h < NQ; h++)
        for (int k = 0; k < RHALF; k++) { cosQ[h * RHALF + k] = cosd[k]; sinQ[h * RHALF + k] = sind[k]; }
    static int32_t axesD[1] = {1};
    static uint16_t epsD[1];
    epsD[0] = f32_to_f16(1e-6f);

    uint32_t dH[2] = {1, H}, dQ[2] = {1, NQ * HD}, dHd[2] = {1, HD};
    uint32_t dWq[2] = {H, NQ * HD}, dWh[2] = {H, HD}, dWo[2] = {NQ * HD, H};
    uint32_t dCx[2] = {CTX, HD}, dKf[2] = {CTF, HD}, dS[2] = {1, CTF};
    uint32_t dR[2] = {1, NQ * RHALF}, dRk[2] = {1, RHALF}, dRpQ[2] = {1, RPASS_Q}, dRpK[2] = {1, RPASS_K};
    uint32_t dIAq[1] = {NQ * RHALF}, dIPq[1] = {RPASS_Q}, dIAk[1] = {RHALF}, dIPk[1] = {RPASS_K};
    uint32_t dHh[1] = {HD}, d11[2] = {1, 1}, dAx[1] = {1};

    Qnn_Tensor_t tX, tSq, tMn, tEp, tAd, tRr, tN1, tAx;
    Qnn_Tensor_t tWq, tWk, tWv, tQ, tK, tV, tKc, tVc;
    Qnn_Tensor_t tQIA, tQIB, tQIP, tKIA, tKIB, tKIP, tCos, tSin, tCosQ, tSinQ;
    Qnn_Tensor_t tQa, tQb, tQp, tQ1, tQ2, tQy1, tQ3, tQ4, tQy2, tQr;
    Qnn_Tensor_t tKa, tKb, tKp, tK1, tK2, tKy1, tK3, tK4, tKy2, tKr;
    Qnn_Tensor_t tH0, tH1, tH2, tH3, tKf, tVf;
    Qnn_Tensor_t tS0, tP0, tCv0, tS1, tP1, tCv1, tS2, tP2, tCv2, tS3, tP3, tCv3;
    Qnn_Tensor_t tHc, tWo, tAo, tY;
    mktensor(&tX, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tSq, "sq", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tMn, "mn", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tEp, "eps", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, d11, epsD, sizeof(epsD));
    mktensor(&tAd, "ad", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tRr, "rr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tN1, "n1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tAx, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dAx, axesD, sizeof(axesD));
    mktensor(&tWq, "wq", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dWq, wq, (uint32_t)((size_t)H*NQ*HD*2));
    mktensor(&tWk, "wk", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dWh, wk, (uint32_t)((size_t)H*HD*2));
    mktensor(&tWv, "wv", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dWh, wv, (uint32_t)((size_t)H*HD*2));
    mktensor(&tQ, "q", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dQ, NULL, 0);
    mktensor(&tK, "k", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tV, "v", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tKc, "kc", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dCx, kc, (uint32_t)((size_t)CTX*HD*2));
    mktensor(&tVc, "vc", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dCx, vc, (uint32_t)((size_t)CTX*HD*2));
    mktensor(&tQIA, "qIA", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIAq, qA, sizeof(qA));
    mktensor(&tQIB, "qIB", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIAq, qB, sizeof(qB));
    mktensor(&tQIP, "qIP", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIPq, qP, sizeof(qP));
    mktensor(&tKIA, "kIA", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIAk, kA, sizeof(kA));
    mktensor(&tKIB, "kIB", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIAk, kB, sizeof(kB));
    mktensor(&tKIP, "kIP", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIPk, kP, sizeof(kP));
    mktensor(&tCos, "cos", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dRk, cosd, sizeof(cosd));
    mktensor(&tSin, "sin", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dRk, sind, sizeof(sind));
    mktensor(&tCosQ, "cosQ", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dR, cosQ, sizeof(cosQ));
    mktensor(&tSinQ, "sinQ", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dR, sinQ, sizeof(sinQ));
    mktensor(&tQa, "qa", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQb, "qb", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQp, "qp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRpQ, NULL, 0);
    mktensor(&tQ1, "q1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQ2, "q2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQy1, "qy1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQ3, "q3", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQ4, "q4", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQy2, "qy2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0);
    mktensor(&tQr, "qr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dQ, NULL, 0);
    mktensor(&tKa, "ka", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tKb, "kb", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tKp, "kp", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRpK, NULL, 0);
    mktensor(&tK1, "k1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tK2, "k2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tKy1, "ky1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tK3, "k3", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tK4, "k4", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tKy2, "ky2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRk, NULL, 0);
    mktensor(&tKr, "kr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    Qnn_Tensor_t tH[4];
    mktensor(&tH[0], "h0", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dHh, hI[0], sizeof(hI[0]));
    mktensor(&tH[1], "h1", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dHh, hI[1], sizeof(hI[1]));
    mktensor(&tH[2], "h2", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dHh, hI[2], sizeof(hI[2]));
    mktensor(&tH[3], "h3", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dHh, hI[3], sizeof(hI[3]));
    Qnn_Tensor_t tQh[4], tS[4], tP[4], tCv[4];
    const char* qhn[4] = {"qh0", "qh1", "qh2", "qh3"};
    const char* sn[4] = {"s0", "s1", "s2", "s3"};
    const char* pn[4] = {"p0", "p1", "p2", "p3"};
    const char* cvn[4] = {"cv0", "cv1", "cv2", "cv3"};
    for (int h = 0; h < NQ; h++) {
        mktensor(&tQh[h], qhn[h], QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
        mktensor(&tS[h], sn[h], QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
        mktensor(&tP[h], pn[h], QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
        mktensor(&tCv[h], cvn[h], QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    }
    mktensor(&tKf, "kf", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dKf, NULL, 0);
    mktensor(&tVf, "vf", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dKf, NULL, 0);
    mktensor(&tHc, "hc", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dQ, NULL, 0);
    mktensor(&tWo, "wo", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dWo, wo, (uint32_t)((size_t)NQ*HD*H*2));
    mktensor(&tAo, "ao", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tY, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);

    Qnn_Tensor_t* all[] = {&tX,&tSq,&tMn,&tEp,&tAd,&tRr,&tN1,&tWq,&tWk,&tWv,&tQ,&tK,&tV,
        &tKc,&tVc,&tQIA,&tQIB,&tQIP,&tKIA,&tKIB,&tKIP,&tCos,&tSin,&tCosQ,&tSinQ,
        &tQa,&tQb,&tQp,&tQ1,&tQ2,&tQy1,&tQ3,&tQ4,&tQy2,&tQr,
        &tKa,&tKb,&tKp,&tK1,&tK2,&tKy1,&tK3,&tK4,&tKy2,&tKr,
        &tH[0],&tH[1],&tH[2],&tH[3],&tQh[0],&tQh[1],&tQh[2],&tQh[3],
        &tS[0],&tS[1],&tS[2],&tS[3],&tP[0],&tP[1],&tP[2],&tP[3],
        &tCv[0],&tCv[1],&tCv[2],&tCv[3],&tKf,&tVf,&tHc,&tWo,&tAo,&tY};
    int nall = sizeof(all) / sizeof(all[0]);
    for (int i = 0; i < nall; i++) reg(all[i]);
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
    Qnn_Param_t mmT[2]; memcpy(mmT, mm, sizeof(mm));
    mmT[1].scalarParam.bool8Value = 1;
    Qnn_Param_t smP[1]; memset(smP, 0, sizeof(smP));
    smP[0].paramType = QNN_PARAMTYPE_SCALAR; smP[0].name = "axis";
    smP[0].scalarParam.dataType = QNN_DATATYPE_INT_32; smP[0].scalarParam.int32Value = 1;
    Qnn_Param_t axP[1]; memset(axP, 0, sizeof(axP));
    axP[0].paramType = QNN_PARAMTYPE_SCALAR; axP[0].name = "axis";
    axP[0].scalarParam.dataType = QNN_DATATYPE_INT_32; axP[0].scalarParam.int32Value = 1;
    Qnn_Param_t cx0P[1]; memset(cx0P, 0, sizeof(cx0P));
    cx0P[0].paramType = QNN_PARAMTYPE_SCALAR; cx0P[0].name = "axis";
    cx0P[0].scalarParam.dataType = QNN_DATATYPE_INT_32; cx0P[0].scalarParam.int32Value = 0;
    Qnn_Param_t cx1P[1]; memset(cx1P, 0, sizeof(cx1P));
    cx1P[0].paramType = QNN_PARAMTYPE_SCALAR; cx1P[0].name = "axis";
    cx1P[0].scalarParam.dataType = QNN_DATATYPE_INT_32; cx1P[0].scalarParam.int32Value = 1;

    N2("r1sq", "ElementWiseMultiply", tX, tX, tSq, NULL, 0);
    { Qnn_Tensor_t _i[1] = {tSq}, _o[1] = {tMn}; node("r1mn", "ReduceMean", meanP, 2, _i, 1, _o, 1); }
    N2("r1ad", "ElementWiseAdd", tMn, tEp, tAd, NULL, 0);
    N1("r1rr", "ElementWiseRsqrt", tAd, tRr);
    N2("r1nm", "ElementWiseMultiply", tX, tRr, tN1, NULL, 0);
    { Qnn_Tensor_t _i[2] = {tN1, tWq}, _o[1] = {tQ}; node("mmq", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN1, tWk}, _o[1] = {tK}; node("mmk", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN1, tWv}, _o[1] = {tV}; node("mmv", "MatMul", mm, 2, _i, 2, _o, 1); }
    // fused rope on qflat
    N2("qgA", "Gather", tQ, tQIA, tQa, axP, 1);
    N2("qgB", "Gather", tQ, tQIB, tQb, axP, 1);
    N2("qgP", "Gather", tQ, tQIP, tQp, axP, 1);
    N2("qt1", "ElementWiseMultiply", tQa, tCosQ, tQ1, NULL, 0);
    N2("qt2", "ElementWiseMultiply", tQb, tSinQ, tQ2, NULL, 0);
    N2("qy1", "ElementWiseSubtract", tQ1, tQ2, tQy1, NULL, 0);
    N2("qt3", "ElementWiseMultiply", tQa, tSinQ, tQ3, NULL, 0);
    N2("qt4", "ElementWiseMultiply", tQb, tCosQ, tQ4, NULL, 0);
    N2("qy2", "ElementWiseAdd", tQ3, tQ4, tQy2, NULL, 0);
    N3("qct", "Concat", tQy1, tQy2, tQp, tQr, cx1P, 1);
    // fused rope on k
    N2("kgA", "Gather", tK, tKIA, tKa, axP, 1);
    N2("kgB", "Gather", tK, tKIB, tKb, axP, 1);
    N2("kgP", "Gather", tK, tKIP, tKp, axP, 1);
    N2("kt1", "ElementWiseMultiply", tKa, tCos, tK1, NULL, 0);
    N2("kt2", "ElementWiseMultiply", tKb, tSin, tK2, NULL, 0);
    N2("ky1", "ElementWiseSubtract", tK1, tK2, tKy1, NULL, 0);
    N2("kt3", "ElementWiseMultiply", tKa, tSin, tK3, NULL, 0);
    N2("kt4", "ElementWiseMultiply", tKb, tCos, tK4, NULL, 0);
    N2("ky2", "ElementWiseAdd", tK3, tK4, tKy2, NULL, 0);
    N3("kct", "Concat", tKy1, tKy2, tKp, tKr, cx1P, 1);
    // cache append
    N2("kfull", "Concat", tKc, tKr, tKf, cx0P, 1);
    N2("vfull", "Concat", tVc, tV, tVf, cx0P, 1);
#undef N3
#define N3(nm, ty, a, b, c, d, o, prm, nprm) do { Qnn_Tensor_t _i[4] = {a, b, c, d}, _o[1] = {o}; \
    node(nm, ty, prm, nprm, _i, 4, _o, 1); } while (0)
    // per-head attention
    for (int h = 0; h < NQ; h++) {
        char nb[16];
        snprintf(nb, sizeof(nb), "hg%d", h);
        N2(nb, "Gather", tQr, tH[h], tQh[h], axP, 1);
    }
    for (int h = 0; h < NQ; h++) {
        char n1[16], n2[16], n3[16];
        snprintf(n1, sizeof(n1), "qk%d", h);
        snprintf(n2, sizeof(n2), "sm%d", h);
        snprintf(n3, sizeof(n3), "pv%d", h);
        Qnn_Tensor_t _i1[2] = {tQh[h], tKf}, _o1[1] = {tS[h]};
        node(n1, "MatMul", mmT, 2, _i1, 2, _o1, 1);
        Qnn_Tensor_t _i2[1] = {tS[h]}, _o2[1] = {tP[h]};
        // NOTE: name pointer dangles; literals only below
        (void)_i2; (void)_o2;
        node(n2, "Softmax", smP, 1, _i2, 1, _o2, 1);
        Qnn_Tensor_t _i3[2] = {tP[h], tVf}, _o3[1] = {tCv[h]};
        node(n3, "MatMul", mm, 2, _i3, 2, _o3, 1);
    }
    N3("hcat", "Concat", tCv[0], tCv[1], tCv[2], tCv[3], tHc, cx1P, 1);
    { Qnn_Tensor_t _i[2] = {tHc, tWo}, _o[1] = {tAo}; node("mmo", "MatMul", mm, 2, _i, 2, _o, 1); }
    N2("add", "ElementWiseAdd", tX, tAo, tY, NULL, 0);

    CHK(Q->QNN_INTERFACE_VER_NAME.graphFinalize(G, NULL, NULL), "graphFinalize");
    printf("GRAPH OK\n");

    double t0 = now_ms();
    static uint16_t qrd[NQ * HD], qh0d[HD], s0d[CTF], cv0d[HD], hcd[NQ * HD], aod[H], qad[NQ * RHALF];
    for (int it = 0; it < iters; it++) {
        tX.v1.clientBuf.data = xdata; tX.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 2);
        tY.v1.clientBuf.data = ydata; tY.v1.clientBuf.dataSize = (uint32_t)((size_t)H * 2);
        Qnn_Tensor_t ein[1] = {tX}, eout[1] = {tY};
        Qnn_ErrorHandle_t e = Q->QNN_INTERFACE_VER_NAME.graphExecute(G, ein, 1, eout, 1, NULL, NULL);
        if (e != QNN_SUCCESS) { printf("EXEC FAIL it=%d err=%d\n", it, (int)e); return 1; }
    }
    double t1 = now_ms();
    printf("EXEC OK iters=%d ms_per_run=%.2f\n", iters, (t1 - t0) / (iters > 0 ? iters : 1));

    // CPU fp32 reference
    float* xf = (float*)malloc((size_t)H * 4);
    float* n1 = (float*)malloc((size_t)H * 4);
    float* qf = (float*)malloc((size_t)(NQ * HD) * 4);
    float* qfp = (float*)malloc((size_t)(NQ * HD) * 4);
    float* kf = (float*)malloc((size_t)HD * 4);
    float* vf = (float*)malloc((size_t)HD * 4);
    if (!xf || !n1 || !qf || !qfp || !kf || !vf) { printf("OOM ref\n"); return 1; }
    for (int i = 0; i < H; i++) xf[i] = f16_to_f32(xdata[i]);
    double ss = 0; for (int i = 0; i < H; i++) ss += (double)xf[i] * xf[i];
    double rr = 1.0 / sqrt(ss / H + 1e-6);
    for (int i = 0; i < H; i++) n1[i] = (float)(xf[i] * rr);
    for (int j = 0; j < NQ * HD; j++) {
        double a = 0;
        for (int k = 0; k < H; k++) a += (double)n1[k] * f16_to_f32(wq[k * NQ * HD + j]);
        qf[j] = (float)a;
    }
    memcpy(qfp, qf, (size_t)(NQ * HD) * 4);
    for (int j = 0; j < HD; j++) {
        double a = 0, b = 0;
        for (int k = 0; k < H; k++) { a += (double)n1[k] * f16_to_f32(wk[k * HD + j]); b += (double)n1[k] * f16_to_f32(wv[k * HD + j]); }
        kf[j] = (float)a; vf[j] = (float)b;
    }
    // rope helper (NeoX split-half on 64 rotary dims of 256)
    for (int h = 0; h < NQ; h++) {
        float tmp[HD];
        for (int j = 0; j < HD; j++) tmp[j] = qf[h * HD + j];
        for (int k = 0; k < RHALF; k++) {
            double ang = (double)POS * pow(1e7, -(double)k / 32.0);
            double c = cos(ang), s = sin(ang);
            qf[h * HD + k] = (float)(tmp[k] * c - tmp[k + RHALF] * s);
            qf[h * HD + k + RHALF] = (float)(tmp[k] * s + tmp[k + RHALF] * c);
        }
    }
    { float tmp[HD]; for (int j = 0; j < HD; j++) tmp[j] = kf[j];
      for (int k = 0; k < RHALF; k++) {
        double ang = (double)POS * pow(1e7, -(double)k / 32.0);
        double c = cos(ang), s = sin(ang);
        kf[k] = (float)(tmp[k] * c - tmp[k + RHALF] * s);
        kf[k + RHALF] = (float)(tmp[k] * s + tmp[k + RHALF] * c);
      } }
    static float cvh[NQ][HD];
    static float sc0[CTF];
    static float sc0log[CTF];
    for (int h = 0; h < NQ; h++) {
        double sc[CTF], mx = -1e300;
        for (int c = 0; c < CTX; c++) {
            double s = 0;
            for (int k = 0; k < HD; k++) s += (double)qf[h * HD + k] * f16_to_f32(kc[c * HD + k]);
            sc[c] = s; if (s > mx) mx = s;
        }
        { double s = 0; for (int k = 0; k < HD; k++) s += (double)qf[h * HD + k] * kf[k]; sc[CTX] = s; if (s > mx) mx = s; }
        if (h == 0) for (int c = 0; c < CTF; c++) sc0log[c] = (float)sc[c];
        double ps = 0;
        for (int c = 0; c < CTF; c++) { sc[c] = exp(sc[c] - mx); ps += sc[c]; }
        for (int c = 0; c < CTF; c++) sc[c] /= ps;
        if (h == 0) for (int c = 0; c < CTF; c++) sc0[c] = (float)sc[c];
        for (int j = 0; j < HD; j++) {
            double a = 0;
            for (int c = 0; c < CTX; c++) a += sc[c] * f16_to_f32(vc[c * HD + j]);
            a += sc[CTX] * vf[j];
            cvh[h][j] = (float)a;
        }
    }
    int bad = 0; double maxerr = 0;
    static float aref[8];
    for (int j = 0; j < 8; j++) {
        double a = 0;
        for (int h = 0; h < NQ; h++)
            for (int k = 0; k < HD; k++) a += (double)cvh[h][k] * f16_to_f32(wo[(h * HD + k) * H + j]);
        aref[j] = (float)a;
        double ref = xf[j] + a;
        double got = f16_to_f32(ydata[j]);
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.05 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr8=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    {
        // qrf: rope ref in GRAPH fused-flat layout [y1_all(128)|y2_all(128)|pass_all(768)]
        float qrf[NQ * HD];
        for (int h = 0; h < NQ; h++)
            for (int k = 0; k < RHALF; k++) {
                double ang = (double)POS * pow(1e7, -(double)k / 32.0);
                double c = cos(ang), s = sin(ang);
                double qx1 = qfp[h * HD + k], qx2 = qfp[h * HD + k + RHALF];
                qrf[h * RHALF + k] = (float)(qx1 * c - qx2 * s);
                qrf[NQ * RHALF + h * RHALF + k] = (float)(qx1 * s + qx2 * c);
            }
        for (int h = 0; h < NQ; h++)
            for (int j = 0; j < HD - RDIM; j++) qrf[2 * NQ * RHALF + h * (HD - RDIM) + j] = qfp[h * HD + RDIM + j];
        uint16_t* gg[] = {qrd, qh0d, s0d, cv0d, hcd, aod, qad};
        const char* nn[] = {"qr", "qh0", "s0", "cv0", "hc", "ao", "qa"};
        int fulllim[] = {NQ * HD, HD, CTF, HD, NQ * HD, H, NQ * RHALF};
        for (int t = 0; t < 7; t++) {
            double me = 0;
            int lim = (t == 2) ? CTF : 8;
            double r0 = 0, g0 = 0;
            for (int j = 0; j < lim; j++) {
                double ref;
                if (t == 0) ref = qf[j];
                else if (t == 1) ref = qf[j];
                else if (t == 2) ref = sc0log[j];
                else if (t == 3) ref = cvh[0][j];
                else if (t == 4) ref = cvh[0][j];
                else if (t == 5) ref = aref[j];
                else ref = qfp[qA[j]];
                double got = f16_to_f32(gg[t][j]);
                if (j == 0) { r0 = ref; g0 = got; }
                double e = ref - got; if (e < 0) e = -e;
                if (e > me) me = e;
                if (t == 2) printf("dbg s0[%d] ref=%0.5f got=%0.5f\n", j, ref, got);
            }
            printf("dbg %s maxerr=%0.5f ref0=%0.5f got0=%0.5f\n", nn[t], me, r0, g0);
            if (t == 0) {
                double e1 = 0, e2 = 0, e3 = 0;
                for (int j = 0; j < 128; j++) {
                    double e = qrf[j] - f16_to_f32(qrd[j]); if (e < 0) e = -e;
                    if (e > e1) e1 = e;
                }
                for (int j = 128; j < 256; j++) {
                    double e = qrf[j] - f16_to_f32(qrd[j]); if (e < 0) e = -e;
                    if (e > e2) e2 = e;
                }
                for (int j = 256; j < NQ * HD; j++) {
                    double e = qrf[j] - f16_to_f32(qrd[j]); if (e < 0) e = -e;
                    if (e > e3) e3 = e;
                }
                printf("dbg qr y1[0..127]=%0.5f y2[128..255]=%0.5f pass[256..]=%0.5f\n", e1, e2, e3);
            }
            if (t != 2 && t != 5) {
                double mf = 0;
                for (int j = 0; j < fulllim[t]; j++) {
                    double ref;
                    if (t == 0) ref = qrf[j];
                    else if (t == 1) ref = qf[j];
                    else if (t == 3) ref = cvh[0][j];
                    else if (t == 4) ref = cvh[0][j];
                    else if (t == 5) ref = aref[j];
                    else ref = kf[j];
                    double e = ref - f16_to_f32(gg[t][j]); if (e < 0) e = -e;
                    if (e > mf) mf = e;
                }
                printf("dbg %s FULLmaxerr=%0.5f\n", nn[t], mf);
            }
        }
    }
    Q->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    Q->QNN_INTERFACE_VER_NAME.deviceFree(device);
    Q->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
