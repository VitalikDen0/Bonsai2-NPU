#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    int n_layers = argc > 1 ? atoi(argv[1]) : 3;
    size_t sz_g0 = (size_t)1540 * 1024 * 1024;
    size_t sz_g1 = (size_t)1540 * 1024 * 1024;
    size_t sz_lm = (size_t)322 * 1024 * 1024;
    size_t sz_ring = (size_t)n_layers * 97 * 1024 * 1024;

    printf("[probe] Testing %d-layer ring buffer (%zu MB)...\n", n_layers, sz_ring >> 20);

    void* p_g0 = rpcmem_alloc(25, 0, sz_g0);
    int fd_g0 = p_g0 ? rpcmem_to_fd(p_g0) : -1;
    int rc_g0 = (fd_g0 >= 0) ? fastrpc_mmap(3, fd_g0, p_g0, 0, sz_g0, 0) : -1;

    void* p_g1 = rpcmem_alloc(25, 0, sz_g1);
    int fd_g1 = p_g1 ? rpcmem_to_fd(p_g1) : -1;
    int rc_g1 = (fd_g1 >= 0) ? fastrpc_mmap(3, fd_g1, p_g1, 0, sz_g1, 0) : -1;

    void* p_lm = rpcmem_alloc(25, 0, sz_lm);
    int fd_lm = p_lm ? rpcmem_to_fd(p_lm) : -1;
    int rc_lm = (fd_lm >= 0) ? fastrpc_mmap(3, fd_lm, p_lm, 0, sz_lm, 0) : -1;

    void* p_ring = rpcmem_alloc(25, 0, sz_ring);
    int fd_ring = p_ring ? rpcmem_to_fd(p_ring) : -1;
    int rc_ring = (fd_ring >= 0) ? fastrpc_mmap(3, fd_ring, p_ring, 0, sz_ring, 0) : -1;

    printf("[probe] Results: g0(1540M)=%d, g1(1540M)=%d, lm(322M)=%d, ring(%zuM)=%d\n",
           rc_g0, rc_g1, rc_lm, sz_ring >> 20, rc_ring);

    if (rc_ring == 0) fastrpc_munmap(3, fd_ring, p_ring, sz_ring);
    if (p_ring) rpcmem_free(p_ring);
    if (rc_lm == 0) fastrpc_munmap(3, fd_lm, p_lm, sz_lm);
    if (p_lm) rpcmem_free(p_lm);
    if (rc_g1 == 0) fastrpc_munmap(3, fd_g1, p_g1, sz_g1);
    if (p_g1) rpcmem_free(p_g1);
    if (rc_g0 == 0) fastrpc_munmap(3, fd_g0, p_g0, sz_g0);
    if (p_g0) rpcmem_free(p_g0);
    return 0;
}
