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

int main(void) {
    setbuf(stdout, NULL);
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    size_t sz_g0 = (size_t)1615 * 1024 * 1024;
    size_t sz_g1 = (size_t)1615 * 1024 * 1024;
    size_t sz_lm = (size_t)322 * 1024 * 1024;
    size_t sz_ring = (size_t)192 * 1024 * 1024;
    size_t sz_io = (size_t)6 * 1024 * 1024;

    printf("[probe_order] Allocating & Mapping Group 0 (%zu MB)...\n", sz_g0 >> 20);
    void* p_g0 = rpcmem_alloc(25, 0, sz_g0);
    int fd_g0 = rpcmem_to_fd(p_g0);
    int rc_g0 = fastrpc_mmap(3, fd_g0, p_g0, 0, sz_g0, 0);
    printf("  Group 0 alloc=%p fd=%d rc=%d\n", p_g0, fd_g0, rc_g0);

    printf("[probe_order] Allocating & Mapping Group 1 (%zu MB)...\n", sz_g1 >> 20);
    void* p_g1 = rpcmem_alloc(25, 0, sz_g1);
    int fd_g1 = rpcmem_to_fd(p_g1);
    int rc_g1 = fastrpc_mmap(3, fd_g1, p_g1, 0, sz_g1, 0);
    printf("  Group 1 alloc=%p fd=%d rc=%d\n", p_g1, fd_g1, rc_g1);

    printf("[probe_order] Allocating & Mapping LM Head (%zu MB)...\n", sz_lm >> 20);
    void* p_lm = rpcmem_alloc(25, 0, sz_lm);
    int fd_lm = rpcmem_to_fd(p_lm);
    int rc_lm = fastrpc_mmap(3, fd_lm, p_lm, 0, sz_lm, 0);
    printf("  LM Head alloc=%p fd=%d rc=%d\n", p_lm, fd_lm, rc_lm);

    printf("[probe_order] Allocating & Mapping Ring Arena (%zu MB)...\n", sz_ring >> 20);
    void* p_ring = rpcmem_alloc(25, 0, sz_ring);
    int fd_ring = rpcmem_to_fd(p_ring);
    int rc_ring = fastrpc_mmap(3, fd_ring, p_ring, 0, sz_ring, 0);
    printf("  Ring Arena alloc=%p fd=%d rc=%d\n", p_ring, fd_ring, rc_ring);

    printf("[probe_order] Allocating & Mapping IO Arena (%zu MB)...\n", sz_io >> 20);
    void* p_io = rpcmem_alloc(25, 1, sz_io);
    int fd_io = rpcmem_to_fd(p_io);
    int rc_io = fastrpc_mmap(3, fd_io, p_io, 0, sz_io, 0);
    printf("  IO Arena alloc=%p fd=%d rc=%d\n", p_io, fd_io, rc_io);

    size_t total = (rc_g0 == 0 ? sz_g0 : 0) + (rc_g1 == 0 ? sz_g1 : 0) +
                   (rc_lm == 0 ? sz_lm : 0) + (rc_ring == 0 ? sz_ring : 0) +
                   (rc_io == 0 ? sz_io : 0);
    printf("TOTAL MAPPED: %zu MB (%.2f GB)!\n", total >> 20, (double)total / (1024.0*1024.0*1024.0));

    // cleanup
    if (rc_io == 0) fastrpc_munmap(3, fd_io, p_io, sz_io);
    if (rc_ring == 0) fastrpc_munmap(3, fd_ring, p_ring, sz_ring);
    if (rc_lm == 0) fastrpc_munmap(3, fd_lm, p_lm, sz_lm);
    if (rc_g1 == 0) fastrpc_munmap(3, fd_g1, p_g1, sz_g1);
    if (rc_g0 == 0) fastrpc_munmap(3, fd_g0, p_g0, sz_g0);

    if (p_io) rpcmem_free(p_io);
    if (p_ring) rpcmem_free(p_ring);
    if (p_lm) rpcmem_free(p_lm);
    if (p_g1) rpcmem_free(p_g1);
    if (p_g0) rpcmem_free(p_g0);

    return 0;
}
