#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    size_t sz = 95 * 1024 * 1024; // 95 MB (typical layer size)
    printf("[probe] Allocating 95 MB rpcmem...\n");
    double t0 = now_us();
    void* p = rpcmem_alloc(25, 0, sz);
    double t1 = now_us();
    if (!p) { printf("[probe] FAIL rpcmem_alloc\n"); return 1; }
    int fd = rpcmem_to_fd(p);
    printf("[probe] alloc took %.1f us, ptr=%p, fd=%d\n", t1 - t0, p, fd);

    // Warmup mmap
    int rc = fastrpc_mmap(3, fd, p, 0, sz, 0);
    printf("[probe] warmup fastrpc_mmap rc=%d\n", rc);
    fastrpc_munmap(3, fd, p, sz);

    double total_mmap_us = 0;
    double total_munmap_us = 0;
    int iters = 64; // simulate 64 layers

    for (int i = 0; i < iters; i++) {
        double m0 = now_us();
        rc = fastrpc_mmap(3, fd, p, 0, sz, 0);
        double m1 = now_us();
        if (rc != 0) { printf("[probe] mmap failed at iter %d rc=%d\n", i, rc); break; }

        double u0 = now_us();
        int urc = fastrpc_munmap(3, fd, p, sz);
        double u1 = now_us();
        if (urc != 0) { printf("[probe] munmap failed at iter %d rc=%d\n", i, urc); break; }

        total_mmap_us += (m1 - m0);
        total_munmap_us += (u1 - u0);
    }

    printf("[probe] 64 iters of 95 MB fastrpc_mmap avg: %.1f us (total: %.2f ms)\n",
           total_mmap_us / iters, total_mmap_us / 1000.0);
    printf("[probe] 64 iters of 95 MB fastrpc_munmap avg: %.1f us (total: %.2f ms)\n",
           total_munmap_us / iters, total_munmap_us / 1000.0);
    printf("[probe] TOTAL SMMU map/unmap overhead across 64 layers: %.2f ms (compare to 100+ ms of memcpy!)\n",
           (total_mmap_us + total_munmap_us) / 1000.0);

    rpcmem_free(p);
    return 0;
}
