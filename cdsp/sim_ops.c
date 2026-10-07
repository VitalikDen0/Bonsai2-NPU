// Sim test for bonsai_ops: property + reference checks, all deterministic.
#include <stdio.h>
#include <string.h>
#include <math.h>

void bonsai_rmsnorm(const float* x, const float* w, int n, float* out);
void bonsai_rope(float* q, float* k, int heads, int head_dim, int pos);
void bonsai_swiglu(const float* gate, const float* up, int n, float* out);
float bonsai_softmax(float* x, int n);
int bonsai_sample(const float* logits, int vocab, float temp,
                  int top_k, float top_p, unsigned* seed);

static unsigned rs = 0xABCDEFu;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (float)(rs >> 8) / 8388608.0f - 1.0f; }

int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL %s\n", msg); fails++; } } while (0)

int main(void) {
    // rmsnorm: constant input -> +-1 outputs; unit-RMS property
    {
        float x[5120], w[5120], o[5120];
        for (int i = 0; i < 5120; i++) { x[i] = 2.5f; w[i] = 1.0f + 0.001f * frand(); }
        bonsai_rmsnorm(x, w, 5120, o);
        double s = 0;
        for (int i = 0; i < 5120; i++) { double v = o[i] / w[i]; s += v * v; }
        double rms = sqrt(s / 5120);
        CHECK(rms > 0.99999 && rms < 1.00001, "rmsnorm unit rms");
        // zero input stays zero
        for (int i = 0; i < 5120; i++) x[i] = 0;
        bonsai_rmsnorm(x, w, 5120, o);
        CHECK(o[0] == 0 && o[5119] == 0, "rmsnorm zero");
    }
    // rope: pos=0 identity; norm preserved at pos>0; untouched tail
    {
        float q[4 * 256], k[4 * 256], q0[4 * 256];
        for (int i = 0; i < 4 * 256; i++) { q[i] = frand(); k[i] = frand(); q0[i] = q[i]; }
        bonsai_rope(q, k, 4, 256, 0);
        int same = 1;
        for (int i = 0; i < 4 * 256; i++) if (q[i] != q0[i]) same = 0;
        CHECK(same, "rope pos0 identity");
        bonsai_rope(q, k, 4, 256, 100);
        for (int h = 0; h < 4; h++) {
            double a = 0, b = 0;
            for (int i = 0; i < 64; i++) {
                a += (double)q0[h * 256 + i] * q0[h * 256 + i];
                b += (double)q[h * 256 + i] * q[h * 256 + i];
            }
            if (!(b > 0.999 * a && b < 1.001 * a)) { CHECK(0, "rope norm"); break; }
            for (int i = 64; i < 256; i++)
                if (q[h * 256 + i] != q0[h * 256 + i]) { CHECK(0, "rope tail"); break; }
        }
    }
    // swiglu spot check vs double
    {
        float g[16], u[16], o[16];
        for (int i = 0; i < 16; i++) { g[i] = frand() * 3; u[i] = frand() * 3; }
        bonsai_swiglu(g, u, 16, o);
        double me = 0;
        for (int i = 0; i < 16; i++) {
            double gd = g[i], ref = gd / (1.0 + exp(-gd)) * (double)u[i];
            double e = ref - o[i]; if (e < 0) e = -e;
            if (e > me) me = e;
        }
        CHECK(me < 1e-5, "swiglu");
    }
    // softmax sums to 1, argmax kept
    {
        float x[512];
        int am = 0;
        for (int i = 0; i < 512; i++) { x[i] = frand() * 5; if (x[i] > x[am]) am = i; }
        bonsai_softmax(x, 512);
        double s = 0; int am2 = 0;
        for (int i = 0; i < 512; i++) { s += x[i]; if (x[i] > x[am2]) am2 = i; }
        CHECK(s > 0.99999 && s < 1.00001, "softmax sum");
        CHECK(am == am2, "softmax argmax");
    }
    // sample: greedy + seeded determinism
    {
        float lg[300];
        for (int i = 0; i < 300; i++) lg[i] = frand();
        lg[77] = 10.0f;
        CHECK(bonsai_sample(lg, 300, 0, 20, 0.95f, &rs) == 77, "sample greedy");
        unsigned s1 = 12345, s2 = 12345;
        int a = bonsai_sample(lg, 300, 0.7f, 20, 0.95f, &s1);
        int b = bonsai_sample(lg, 300, 0.7f, 20, 0.95f, &s2);
        CHECK(a == b && a >= 0 && a < 300, "sample deterministic");
    }
    printf(fails == 0 ? "OPS SIM PASS\n" : "OPS SIM FAIL\n");
    return fails == 0 ? 0 : 1;
}
