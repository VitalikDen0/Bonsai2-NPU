// probe_q1r: tiny gemv_q1r calls to isolate the crash.
// Usage: probe_q1r
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C" {
void rpcmem_init(void);
void* rpcmem_alloc(int heapid, unsigned flags, int size);
void rpcmem_free(void* po);
}
#include "bonsai.h"

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    extern int remote_session_control(unsigned, void*, unsigned);
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();
    remote_handle64 h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc) { printf("OPEN FAIL rc=%d\n", rc); return 1; }
    printf("OPEN OK\n");

    // tiny: out=1, in=128, prow=16, ng=1
    float* x = (float*)rpcmem_alloc(25, 1, 128 * 4);
    unsigned char* bits = (unsigned char*)rpcmem_alloc(25, 1, 16);
    short* scales = (short*)rpcmem_alloc(25, 1, 2);
    float* y = (float*)rpcmem_alloc(25, 1, 4);
    if (!x || !bits || !scales || !y) { printf("ALLOC FAIL\n"); return 1; }
    for (int i = 0; i < 128; i++) x[i] = 0.01f * (i % 5);
    memset(bits, 0xAA, 16);
    // scale fp16 1.0 = 0x3C00
    scales[0] = (short)0x3C00;
    auto hi = [](uintptr_t p) { return (int)(p >> 32); };
    auto lo = [](uintptr_t p) { return (int)(p & 0xffffffff); };
    printf("ptrs x=%llx bits=%llx sc=%llx y=%llx\n",
           (unsigned long long)(uintptr_t)x, (unsigned long long)(uintptr_t)bits,
           (unsigned long long)(uintptr_t)scales, (unsigned long long)(uintptr_t)y);
    rc = bonsai_gemv_q1r(h, 1, 128, 16,
                         hi((uintptr_t)x), lo((uintptr_t)x),
                         hi((uintptr_t)bits), lo((uintptr_t)bits),
                         hi((uintptr_t)scales), lo((uintptr_t)scales),
                         hi((uintptr_t)y), lo((uintptr_t)y));
    printf("TINY rc=%d y=%.6f (expect ~0.01*avg)\n", rc, y[0]);
    // expected: s=0.5, bits 0xAA -> half +half: sum = 0.5*(sum0 - sum1)
    double ref = 0;
    for (int j = 0; j < 128; j++) {
        int bit = (0xAA >> (j & 7)) & 1;
        // careful: byte j/8 of 0xAA pattern
        bit = (bits[j >> 3] >> (j & 7)) & 1;
        ref += (bit ? 0.5 : -0.5) * (double)x[j];
    }
    printf("REF=%.6f\n", ref);
    bonsai_close(h);
    printf("PROBE DONE\n");
    return 0;
}
