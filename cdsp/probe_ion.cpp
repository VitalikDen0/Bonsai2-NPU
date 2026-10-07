// probe_ion: allocate ION dma-buf, register with FastRPC, tiny q1r through it.
// Usage: probe_ion
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>

extern "C" {
void rpcmem_init(void);
}
#include "bonsai.h"

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    extern int remote_session_control(unsigned, void*, unsigned);
    extern int remote_register_dma_handle(int fd, unsigned len);
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();
    remote_handle64 h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc) { printf("OPEN FAIL rc=%d\n", rc); return 1; }
    printf("OPEN OK\n");

    int heapfd = open("/dev/dma_heap/system", O_RDWR);
    if (heapfd < 0) { printf("HEAP OPEN FAIL\n"); return 1; }
    struct dma_heap_allocation_data alloc = {};
    alloc.len = 4096;
    alloc.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heapfd, DMA_HEAP_IOCTL_ALLOC, &alloc) != 0) {
        printf("HEAP ALLOC FAIL\n");
        return 1;
    }
    printf("ION fd=%d\n", alloc.fd);
    void* base = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, alloc.fd, 0);
    if (base == MAP_FAILED) { printf("MMAP FAIL\n"); return 1; }
    rc = remote_register_dma_handle(alloc.fd, 4096);
    printf("REGISTER_DMA_HANDLE rc=%d\n", rc);

    // layout in ION buf: x[128]f32 @0, bits[16] @512, scales[1]u16 @528, y[1]f32 @532
    float* x = (float*)base;
    unsigned char* bits = (unsigned char*)base + 512;
    short* scales = (short*)base + 264;  // (512+16)/2
    float* y = (float*)base + 133;       // (512+16+2+2pad)/4
    for (int i = 0; i < 128; i++) x[i] = 0.01f * (i % 5);
    memset(bits, 0xAA, 16);
    scales[0] = (short)0x3C00;
    y[0] = 0;
    // DSP sees same VA for ION mmap? pass offsets as absolute VA (identity hope) —
    // if mapping differs this faults; diagnostic step.
    uintptr_t bx = (uintptr_t)x, bb = (uintptr_t)bits;
    uintptr_t bs = (uintptr_t)scales, by = (uintptr_t)y;
    printf("ptrs x=%llx bits=%llx sc=%llx y=%llx\n",
           (unsigned long long)bx, (unsigned long long)bb,
           (unsigned long long)bs, (unsigned long long)by);
    auto hi = [](uintptr_t p) { return (int)(p >> 32); };
    auto lo = [](uintptr_t p) { return (int)(p & 0xffffffff); };
    rc = bonsai_gemv_q1r(h, 1, 128, 16,
                         hi(bx), lo(bx), hi(bb), lo(bb),
                         hi(bs), lo(bs), hi(by), lo(by));
    printf("TINY-ION rc=%d y=%.6f\n", rc, y[0]);
    double ref = 0;
    for (int j = 0; j < 128; j++) {
        int bit = (bits[j >> 3] >> (j & 7)) & 1;
        ref += (bit ? 0.5 : -0.5) * (double)x[j];
    }
    printf("REF=%.6f %s\n", ref, (rc == 0) ? "COMPARE-DONE" : "");
    bonsai_close(h);
    printf("PROBE DONE\n");
    return 0;
}
