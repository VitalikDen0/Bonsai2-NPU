// Sim test: HVX kernel vs scalar reference on synthetic deterministic data.
// Built for hexagon-sim (no device, no files needed).
#include <stdio.h>
#include <string.h>

int hvx_gemv_q1(int out_dim, int in_dim, int prow,
                const float* x, const unsigned char* bits,
                const short* scales, float* y);

// scalar reference (same math as PC gate)
static inline float f16r(unsigned short h) {
    unsigned s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    unsigned f = (e == 0) ? (s << 31) : (e == 31) ? ((s << 31) | 0x7f800000u | (m << 13))
                                                  : ((s << 31) | ((e + 112) << 23) | (m << 13));
    float r; memcpy(&r, &f, 4); return r;
}

static unsigned rng_state = 0x12345678u;
static unsigned rnd(void) { rng_state = rng_state * 1664525u + 1013904223u; return rng_state >> 8; }

#define ROWS 8
#define COLS 1024
static float sx[COLS];
static unsigned char sbits[ROWS * (COLS / 8)];
static unsigned short ssc[ROWS * (COLS / 128)];
static float y_hvx[ROWS], y_ref[ROWS];

int main(void) {
    for (int j = 0; j < COLS; j++) {
        float v = ((float)(rnd() % 2000) / 1000.0f - 1.0f);
        sx[j] = v;
    }
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS / 8; j++) sbits[i * (COLS / 8) + j] = (unsigned char)(rnd() & 0xff);
        for (int g = 0; g < COLS / 128; g++) {
            // fp16 scale pattern: 0x3C00 (1.0) scaled variants
            unsigned short s = (unsigned short)(0x3800 + (rnd() % 0x800));
            ssc[i * (COLS / 128) + g] = s;
        }
    }
    int prow = COLS / 8;
    int rc = hvx_gemv_q1(ROWS, COLS, prow, sx, sbits, (short*)ssc, y_hvx);
    if (rc != 0) { printf("HVX rc=%d FAIL\n", rc); return 1; }
    double max_err = 0;
    int fails = 0;
    for (int i = 0; i < ROWS; i++) {
        double acc = 0;
        for (int j = 0; j < COLS; j++) {
            int bit = (sbits[i * prow + (j >> 3)] >> (j & 7)) & 1;
            double s = (double)f16r(ssc[i * (COLS / 128) + (j >> 7)]) * 0.5;
            acc += (bit ? s : -s) * (double)sx[j];
        }
        y_ref[i] = (float)acc;
        double err = acc - (double)y_hvx[i];
        if (err < 0) err = -err;
        if (err > max_err) max_err = err;
        if (err > 0.01) { printf("ROW %d MISMATCH ref=%.6f hvx=%.6f\n", i, (float)acc, y_hvx[i]); fails++; }
    }
    printf("rows=%d fails=%d max_abs_err=%.6f\n", ROWS, fails, max_err);
    printf(fails == 0 ? "SIM GATE PASS\n" : "SIM GATE FAIL\n");
    return fails == 0 ? 0 : 1;
}
