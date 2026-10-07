#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>

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
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main(void) {
    setbuf(stdout, NULL);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    size_t sz_a = (size_t)1450 * 1024 * 1024; // 1.45 GB
    size_t sz_b = (size_t)1450 * 1024 * 1024; // 1.45 GB

    printf("[bench_flip] Allocating Buffer A (1.45 GB) and Buffer B (1.45 GB)...\n");
    void* p_a = rpcmem_alloc(25, 0, sz_a);
    void* p_b = rpcmem_alloc(25, 0, sz_b);
    if (!p_a || !p_b) {
        printf("Alloc failed!\n");
        return 1;
    }

    int fd_a = rpcmem_to_fd(p_a);
    int fd_b = rpcmem_to_fd(p_b);

    printf("[bench_flip] Initial mapping of Buffer A...\n");
    int rc_a = fastrpc_mmap(3, fd_a, p_a, 0, sz_a, 0);
    printf("  rc_a=%d\n", rc_a);

    // Warmup
    double t0 = now_ms();
    int rc_u0 = fastrpc_munmap(3, fd_a, p_a, sz_a);
    int rc_m0 = fastrpc_mmap(3, fd_b, p_b, 0, sz_b, 0);
    double t1 = now_ms();
    printf("[bench_flip] Flip A->B took %.2f ms (unmap rc=%d, mmap rc=%d)\n", t1 - t0, rc_u0, rc_m0);

    // Flip back B->A
    double t2 = now_ms();
    int rc_u1 = fastrpc_munmap(3, fd_b, p_b, sz_b);
    int rc_m1 = fastrpc_mmap(3, fd_a, p_a, 0, sz_a, 0);
    double t3 = now_ms();
    printf("[bench_flip] Flip B->A took %.2f ms (unmap rc=%d, mmap rc=%d)\n", t3 - t2, rc_u1, rc_m1);

    // Repeat 5 times to measure sustained flip latency
    double total_flip = 0;
    for (int i = 0; i < 5; i++) {
        double ta = now_ms();
        fastrpc_munmap(3, fd_a, p_a, sz_a);
        fastrpc_mmap(3, fd_b, p_b, 0, sz_b, 0);
        fastrpc_munmap(3, fd_b, p_b, sz_b);
        fastrpc_mmap(3, fd_a, p_a, 0, sz_a, 0);
        double tb = now_ms();
        total_flip += (tb - ta);
        printf("  iter %d: roundtrip flip (A->B->A) = %.2f ms (%.2f ms per flip)\n", i, tb - ta, (tb - ta) / 2.0);
    }
    printf("[bench_flip] Average per single flip: %.2f ms\n", total_flip / 10.0);

    fastrpc_munmap(3, fd_a, p_a, sz_a);
    rpcmem_free(p_b);
    rpcmem_free(p_a);
    return 0;
}
