// qnn_mha: GQA decode step (2 q-heads share 1 kv-head) with primitive RoPE + cache append.
// x[1,512] -> rms -> q0,q1,k0,v0 (HD=64) -> rope(q,k,rotary 16,theta 1e7) ->
// Kfull=Concat[Kc,k0r] -> s=q@Kf^T -> softmax -> cv=@Vf -> hcat -> @Wo -> +x -> y.
// CTX=8 cached + 1 current. Usage: qnn_mha <libQnnHtp.so> [iters]
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

#define H 512
#define HD 64
#define RDIM 16
#define RHALF 8
#define RPASS (HD - RDIM)
#define CTX 8
#define CTF (CTX + 1)
#define NQ 2
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
    if (argc < 2) { printf("usage: qnn_mha <libQnnHtp.so> [iters]\n"); return 2; }
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

    static uint16_t xdata[H], ydata[H];
    static uint16_t *wq0, *wq1, *wk0, *wv0, *wo, *kc, *vc;
    wq0 = (uint16_t*)malloc((size_t)H * HD * 2);
    wq1 = (uint16_t*)malloc((size_t)H * HD * 2);
    wk0 = (uint16_t*)malloc((size_t)H * HD * 2);
    wv0 = (uint16_t*)malloc((size_t)H * HD * 2);
    wo = (uint16_t*)malloc((size_t)(NQ * HD) * H * 2);
    kc = (uint16_t*)malloc((size_t)CTX * HD * 2);
    vc = (uint16_t*)malloc((size_t)CTX * HD * 2);
    if (!wq0 || !wq1 || !wk0 || !wv0 || !wo || !kc || !vc) { printf("OOM\n"); return 1; }
    unsigned seed = 777;
#define RAND16() (f32_to_f16(((float)(seed = seed * 1664525u + 1013904223u, seed >> 8) / 8388608.0f - 1.0f) * 0.05f))
    for (int i = 0; i < H; i++) xdata[i] = RAND16();
    for (int i = 0; i < H * HD; i++) { wq0[i] = RAND16(); wq1[i] = RAND16(); wk0[i] = RAND16(); wv0[i] = RAND16(); }
    for (int i = 0; i < NQ * HD * H; i++) wo[i] = RAND16();
    for (int i = 0; i < CTX * HD; i++) { kc[i] = RAND16(); vc[i] = RAND16(); }

    static uint16_t cosd[RHALF], sind[RHALF];
    static int32_t idxA[RHALF], idxB[RHALF], idxP[RPASS];
    for (int k = 0; k < RHALF; k++) {
        double ang = (double)POS * pow(1e7, -(double)k / 8.0);
        cosd[k] = f32_to_f16((float)cos(ang));
        sind[k] = f32_to_f16((float)sin(ang));
        idxA[k] = k;
        idxB[k] = k + RHALF;
    }
    for (int k = 0; k < RPASS; k++) idxP[k] = k + RDIM;
    static int32_t axesD[1] = {1};
    static uint16_t epsD[1];
    epsD[0] = f32_to_f16(1e-5f);

    uint32_t dH[2] = {1, H}, dHd[2] = {1, HD}, dHQ[2] = {H, HD};
    uint32_t dCx[2] = {CTX, HD}, dKf[2] = {CTF, HD}, dS[2] = {1, CTF};
    uint32_t dR[2] = {1, RHALF}, dRp[2] = {1, RPASS}, dIA[1] = {RHALF}, dIP[1] = {RPASS};
    uint32_t d11[2] = {1, 1}, dAx[1] = {1}, dCat[2] = {1, NQ * HD}, dWo[2] = {NQ * HD, H};

    Qnn_Tensor_t tX, tSq, tMn, tEp, tAd, tRr, tN1, tAx;
    Qnn_Tensor_t tWq0, tWq1, tWk0, tWv0, tQ0, tQ1, tK0, tV0, tKc, tVc;
    // rope temps per instance (q0, q1, k0)
    Qnn_Tensor_t tQ0a, tQ0b, tQ0s, tQ01, tQ02, tQ0y1, tQ03, tQ04, tQ0y2, tQ0r;
    Qnn_Tensor_t tQ1a, tQ1b, tQ1s, tQ11, tQ12, tQ1y1, tQ13, tQ14, tQ1y2, tQ1r;
    Qnn_Tensor_t tK0a, tK0b, tK0s, tK01, tK02, tK0y1, tK03, tK04, tK0y2, tK0r;
    Qnn_Tensor_t tIA, tIB, tIP, tCos, tSin;
    Qnn_Tensor_t tKf, tVf, tS0, tP0, tCv0, tS1, tP1, tCv1, tHc, tWo, tAo, tY;
    mktensor(&tX, "x", QNN_TENSOR_TYPE_APP_WRITE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tSq, "sq", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tMn, "mn", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tEp, "eps", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, d11, epsD, sizeof(epsD));
    mktensor(&tAd, "ad", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tRr, "rr", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, d11, NULL, 0);
    mktensor(&tN1, "n1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tAx, "axes", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dAx, axesD, sizeof(axesD));
    mktensor(&tWq0, "wq0", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHQ, wq0, (uint32_t)((size_t)H*HD*2));
    mktensor(&tWq1, "wq1", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHQ, wq1, (uint32_t)((size_t)H*HD*2));
    mktensor(&tWk0, "wk0", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHQ, wk0, (uint32_t)((size_t)H*HD*2));
    mktensor(&tWv0, "wv0", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dHQ, wv0, (uint32_t)((size_t)H*HD*2));
    mktensor(&tQ0, "q0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tQ1, "q1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tK0, "k0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tV0, "v0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tKc, "kc", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dCx, kc, (uint32_t)((size_t)CTX*HD*2));
    mktensor(&tVc, "vc", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dCx, vc, (uint32_t)((size_t)CTX*HD*2));
    mktensor(&tIA, "idxA", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIA, idxA, sizeof(idxA));
    mktensor(&tIB, "idxB", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIA, idxB, sizeof(idxB));
    mktensor(&tIP, "idxP", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_INT_32, 1, dIP, idxP, sizeof(idxP));
    mktensor(&tCos, "cos", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dR, cosd, sizeof(cosd));
    mktensor(&tSin, "sin", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dR, sind, sizeof(sind));
#define RDECL(p, nm, dims) \
    mktensor(&t##p##a, #nm "a", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##b, #nm "b", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##s, #nm "s", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dRp, NULL, 0); \
    mktensor(&t##p##1, #nm "1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##2, #nm "2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##y1, #nm "y1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##3, #nm "3", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##4, #nm "4", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##y2, #nm "y2", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dR, NULL, 0); \
    mktensor(&t##p##r, #nm "r", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    RDECL(Q0, q0r, dR)
    RDECL(Q1, q1r, dR)
    RDECL(K0, k0r, dR)
    mktensor(&tKf, "kf", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dKf, NULL, 0);
    mktensor(&tVf, "vf", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dKf, NULL, 0);
    mktensor(&tS0, "s0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tP0, "p0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tCv0, "cv0", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tS1, "s1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tP1, "p1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dS, NULL, 0);
    mktensor(&tCv1, "cv1", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dHd, NULL, 0);
    mktensor(&tHc, "hc", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dCat, NULL, 0);
    mktensor(&tWo, "wo", QNN_TENSOR_TYPE_STATIC, QNN_DATATYPE_FLOAT_16, 2, dWo, wo, (uint32_t)((size_t)NQ*HD*H*2));
    mktensor(&tAo, "ao", QNN_TENSOR_TYPE_NATIVE, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);
    mktensor(&tY, "y", QNN_TENSOR_TYPE_APP_READ, QNN_DATATYPE_FLOAT_16, 2, dH, NULL, 0);

    Qnn_Tensor_t* all[] = {&tX,&tSq,&tMn,&tEp,&tAd,&tRr,&tN1,&tWq0,&tWq1,&tWk0,&tWv0,
        &tQ0,&tQ1,&tK0,&tV0,&tKc,&tVc,&tIA,&tIB,&tIP,&tCos,&tSin,
        &tQ0a,&tQ0b,&tQ0s,&tQ01,&tQ02,&tQ0y1,&tQ03,&tQ04,&tQ0y2,&tQ0r,
        &tQ1a,&tQ1b,&tQ1s,&tQ11,&tQ12,&tQ1y1,&tQ13,&tQ14,&tQ1y2,&tQ1r,
        &tK0a,&tK0b,&tK0s,&tK01,&tK02,&tK0y1,&tK03,&tK04,&tK0y2,&tK0r,
        &tKf,&tVf,&tS0,&tP0,&tCv0,&tS1,&tP1,&tCv1,&tHc,&tWo,&tAo,&tY};
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
    Qnn_Param_t gaxP[1]; memset(gaxP, 0, sizeof(gaxP));
    gaxP[0].paramType = QNN_PARAMTYPE_SCALAR; gaxP[0].name = "axis";
    gaxP[0].scalarParam.dataType = QNN_DATATYPE_INT_32; gaxP[0].scalarParam.int32Value = 1;
    Qnn_Param_t cax0P[1]; memset(cax0P, 0, sizeof(cax0P));
    cax0P[0].paramType = QNN_PARAMTYPE_SCALAR; cax0P[0].name = "axis";
    cax0P[0].scalarParam.dataType = QNN_DATATYPE_INT_32; cax0P[0].scalarParam.int32Value = 0;
    Qnn_Param_t cax1P[1]; memset(cax1P, 0, sizeof(cax1P));
    cax1P[0].paramType = QNN_PARAMTYPE_SCALAR; cax1P[0].name = "axis";
    cax1P[0].scalarParam.dataType = QNN_DATATYPE_INT_32; cax1P[0].scalarParam.int32Value = 1;

    // rms1
    N2("r1sq", "ElementWiseMultiply", tX, tX, tSq, NULL, 0);
    { Qnn_Tensor_t _i[1] = {tSq}, _o[1] = {tMn}; node("r1mn", "ReduceMean", meanP, 2, _i, 1, _o, 1); }
    N2("r1ad", "ElementWiseAdd", tMn, tEp, tAd, NULL, 0);
    N1("r1rr", "ElementWiseRsqrt", tAd, tRr);
    N2("r1nm", "ElementWiseMultiply", tX, tRr, tN1, NULL, 0);
    // qkv
    { Qnn_Tensor_t _i[2] = {tN1, tWq0}, _o[1] = {tQ0}; node("mmq0", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN1, tWq1}, _o[1] = {tQ1}; node("mmq1", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN1, tWk0}, _o[1] = {tK0}; node("mmk0", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tN1, tWv0}, _o[1] = {tV0}; node("mmv0", "MatMul", mm, 2, _i, 2, _o, 1); }
    // rope macro: out = rope(x)
#define ROPE(p, x, tag) \
    N2(tag "gA", "Gather", x, tIA, t##p##a, gaxP, 1); \
    N2(tag "gB", "Gather", x, tIB, t##p##b, gaxP, 1); \
    N2(tag "gP", "Gather", x, tIP, t##p##s, gaxP, 1); \
    N2(tag "t1", "ElementWiseMultiply", t##p##a, tCos, t##p##1, NULL, 0); \
    N2(tag "t2", "ElementWiseMultiply", t##p##b, tSin, t##p##2, NULL, 0); \
    N2(tag "y1", "ElementWiseSubtract", t##p##1, t##p##2, t##p##y1, NULL, 0); \
    N2(tag "t3", "ElementWiseMultiply", t##p##a, tSin, t##p##3, NULL, 0); \
    N2(tag "t4", "ElementWiseMultiply", t##p##b, tCos, t##p##4, NULL, 0); \
    N2(tag "y2", "ElementWiseAdd", t##p##3, t##p##4, t##p##y2, NULL, 0); \
    N3(tag "ct", "Concat", t##p##y1, t##p##y2, t##p##s, t##p##r, cax1P, 1);
    ROPE(Q0, tQ0, "q0")
    ROPE(Q1, tQ1, "q1")
    ROPE(K0, tK0, "k0")
    // cache append
    N2("kfull", "Concat", tKc, tK0r, tKf, cax0P, 1);
    N2("vfull", "Concat", tVc, tV0, tVf, cax0P, 1);
    // attention per q-head
    { Qnn_Tensor_t _i[2] = {tQ0r, tKf}, _o[1] = {tS0}; node("qk0", "MatMul", mmT, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[1] = {tS0}, _o[1] = {tP0}; node("sm0", "Softmax", smP, 1, _i, 1, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tP0, tVf}, _o[1] = {tCv0}; node("pv0", "MatMul", mm, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tQ1r, tKf}, _o[1] = {tS1}; node("qk1", "MatMul", mmT, 2, _i, 2, _o, 1); }
    { Qnn_Tensor_t _i[1] = {tS1}, _o[1] = {tP1}; node("sm1", "Softmax", smP, 1, _i, 1, _o, 1); }
    { Qnn_Tensor_t _i[2] = {tP1, tVf}, _o[1] = {tCv1}; node("pv1", "MatMul", mm, 2, _i, 2, _o, 1); }
    // heads concat + Wo + residual
    N2("hcat", "Concat", tCv0, tCv1, tHc, cax1P, 1);
    { Qnn_Tensor_t _i[2] = {tHc, tWo}, _o[1] = {tAo}; node("mmo", "MatMul", mm, 2, _i, 2, _o, 1); }
    N2("add", "ElementWiseAdd", tX, tAo, tY, NULL, 0);

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

    // CPU fp32 reference
    static float xf[H], n1[H];
    for (int i = 0; i < H; i++) xf[i] = f16_to_f32(xdata[i]);
    double ss = 0; for (int i = 0; i < H; i++) ss += (double)xf[i] * xf[i];
    double rr = 1.0 / sqrt(ss / H + 1e-5);
    for (int i = 0; i < H; i++) n1[i] = (float)(xf[i] * rr);
    static float qh[NQ][HD], kh[HD], vh[HD];
    uint16_t* wqh[NQ] = {wq0, wq1};
    for (int h = 0; h < NQ; h++)
        for (int j = 0; j < HD; j++) {
            double a = 0;
            for (int k = 0; k < H; k++) a += (double)n1[k] * f16_to_f32(wqh[h][k * HD + j]);
            qh[h][j] = (float)a;
        }
    for (int j = 0; j < HD; j++) {
        double a = 0, b = 0;
        for (int k = 0; k < H; k++) { a += (double)n1[k] * f16_to_f32(wk0[k * HD + j]); b += (double)n1[k] * f16_to_f32(wv0[k * HD + j]); }
        kh[j] = (float)a; vh[j] = (float)b;
    }
    // rope (NeoX split-half, rotary first RDIM)
    static float qhr[NQ][HD], khr[HD];
    for (int h = 0; h < NQ; h++) {
        for (int j = 0; j < HD; j++) qhr[h][j] = qh[h][j];
        for (int k = 0; k < RHALF; k++) {
            double ang = (double)POS * pow(1e7, -(double)k / 8.0);
            double c = cos(ang), s = sin(ang);
            double x1 = qh[h][k], x2 = qh[h][k + RHALF];
            qhr[h][k] = (float)(x1 * c - x2 * s);
            qhr[h][k + RHALF] = (float)(x1 * s + x2 * c);
        }
    }
    for (int j = 0; j < HD; j++) khr[j] = kh[j];
    for (int k = 0; k < RHALF; k++) {
        double ang = (double)POS * pow(1e7, -(double)k / 8.0);
        double c = cos(ang), s = sin(ang);
        double x1 = kh[k], x2 = kh[k + RHALF];
        khr[k] = (float)(x1 * c - x2 * s);
        khr[k + RHALF] = (float)(x1 * s + x2 * c);
    }
    // attention per head over cache + current
    static float cvh[NQ][HD];
    for (int h = 0; h < NQ; h++) {
        double sc[CTF], mx = -1e300;
        for (int c = 0; c < CTX; c++) {
            double s = 0;
            for (int k = 0; k < HD; k++) s += (double)qhr[h][k] * f16_to_f32(kc[c * HD + k]);
            sc[c] = s; if (s > mx) mx = s;
        }
        { double s = 0; for (int k = 0; k < HD; k++) s += (double)qhr[h][k] * khr[k]; sc[CTX] = s; if (s > mx) mx = s; }
        double ps = 0;
        for (int c = 0; c < CTF; c++) { sc[c] = exp(sc[c] - mx); ps += sc[c]; }
        for (int c = 0; c < CTF; c++) sc[c] /= ps;
        for (int j = 0; j < HD; j++) {
            double a = 0;
            for (int c = 0; c < CTX; c++) a += sc[c] * f16_to_f32(vc[c * HD + j]);
            a += sc[CTX] * vh[j];
            cvh[h][j] = (float)a;
        }
    }
    static float ao[H];
    for (int j = 0; j < H; j++) {
        double a = 0;
        for (int h = 0; h < NQ; h++)
            for (int k = 0; k < HD; k++) a += (double)cvh[h][k] * f16_to_f32(wo[(h * HD + k) * H + j]);
        ao[j] = (float)a;
    }
    int bad = 0; double maxerr = 0;
    for (int j = 0; j < 8; j++) {
        double ref = xf[j] + ao[j];
        double got = f16_to_f32(ydata[j]);
        double err = ref - got; if (err < 0) err = -err;
        if (err > maxerr) maxerr = err;
        double tol = 0.05 * (1.0 + (ref < 0 ? -ref : ref));
        if (err > tol) { printf("MISMATCH j=%d ref=%.5f got=%.5f\n", j, ref, got); bad = 1; }
    }
    printf("maxerr8=%.5f %s\n", maxerr, bad ? "VERIFY FAIL" : "VERIFY PASS");
    Q->QNN_INTERFACE_VER_NAME.contextFree(context, NULL);
    Q->QNN_INTERFACE_VER_NAME.deviceFree(device);
    Q->QNN_INTERFACE_VER_NAME.backendFree(backend);
    printf("DONE\n");
    return bad;
}
