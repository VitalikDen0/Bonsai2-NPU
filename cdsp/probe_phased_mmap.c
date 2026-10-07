#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    setbuf(stdout, NULL);
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    size_t sz = 95 * 1024 * 1024;
    void* ptrs[64] = {0};
    int fds[64] = {0};

    printf("[probe_phased] 1. Allocating all 64 layers (5.94 GB) in rpcmem...\n");
    double t0 = now_ms();
    for (int i = 0; i < 64; i++) {
        ptrs[i] = rpcmem_alloc(25, 0, sz);
        if (!ptrs[i]) {
            printf("[probe_phased] rpcmem_alloc failed at layer %d\n", i);
            for (int j = 0; j < i; j++) rpcmem_free(ptrs[j]);
            return 1;
        }
        fds[i] = rpcmem_to_fd(ptrs[i]);
    }
    printf("[probe_phased] Allocation complete in %.2f ms!\n", now_ms() - t0);

    // Initial map: Phase 0 (layers 0..15)
    printf("[probe_phased] 2. Initial mapping of Phase 0 (layers 0..15 = 1.48 GB)...\n");
    t0 = now_ms();
    for (int i = 0; i < 16; i++) {
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, sz, 0);
        if (rc != 0) {
            printf("[probe_phased] Initial map failed at layer %d (rc=%d)\n", i, rc);
            return 1;
        }
    }
    printf("[probe_phased] Phase 0 mapped in %.2f ms\n", now_ms() - t0);

    // Now test token forward pass transitions
    printf("[probe_phased] 3. Simulating phased transitions during 1 token forward pass:\n");
    
    // Transition Phase 0 -> Phase 1
    t0 = now_ms();
    for (int i = 0; i < 16; i++) fastrpc_munmap(3, fds[i], ptrs[i], sz);
    for (int i = 16; i < 32; i++) {
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, sz, 0);
        if (rc != 0) printf("Phase 1 map fail at %d\n", i);
    }
    double t_p1 = now_ms() - t0;
    printf("  Transition 0 -> 1 (unmap 0..15, map 16..31): %.2f ms\n", t_p1);

    // Transition Phase 1 -> Phase 2
    t0 = now_ms();
    for (int i = 16; i < 32; i++) fastrpc_munmap(3, fds[i], ptrs[i], sz);
    for (int i = 32; i < 48; i++) {
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, sz, 0);
        if (rc != 0) printf("Phase 2 map fail at %d\n", i);
    }
    double t_p2 = now_ms() - t0;
    printf("  Transition 1 -> 2 (unmap 16..31, map 32..47): %.2f ms\n", t_p2);

    // Transition Phase 2 -> Phase 3
    t0 = now_ms();
    for (int i = 32; i < 48; i++) fastrpc_munmap(3, fds[i], ptrs[i], sz);
    for (int i = 48; i < 64; i++) {
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, sz, 0);
        if (rc != 0) printf("Phase 3 map fail at %d\n", i);
    }
    double t_p3 = now_ms() - t0;
    printf("  Transition 2 -> 3 (unmap 32..47, map 48..63): %.2f ms\n", t_p3);

    // Transition Phase 3 -> Phase 0 (for next token)
    t0 = now_ms();
    for (int i = 48; i < 64; i++) fastrpc_munmap(3, fds[i], ptrs[i], sz);
    for (int i = 0; i < 16; i++) {
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, sz, 0);
        if (rc != 0) printf("Phase 0 wrap map fail at %d\n", i);
    }
    double t_wrap = now_ms() - t0;
    printf("  Transition 3 -> 0 (unmap 48..63, map 0..15): %.2f ms\n", t_wrap);

    double total_switch = t_p1 + t_p2 + t_p3 + t_wrap;
    printf("[probe_phased] TOTAL phase switching overhead per token: %.2f ms (avg per switch: %.2f ms)\n",
           total_switch, total_switch / 4.0);

    // Cleanup
    for (int i = 0; i < 16; i++) fastrpc_munmap(3, fds[i], ptrs[i], sz);
    for (int i = 0; i < 64; i++) rpcmem_free(ptrs[i]);
    printf("[probe_phased] Successfully verified and cleaned up.\n");
    return 0;
}
