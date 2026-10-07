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

    // 32 layers = 3260 MB = 2 x 1630 MB
    size_t sz_half0 = (size_t)1630 * 1024 * 1024;
    size_t sz_half1 = (size_t)1630 * 1024 * 1024;
    size_t sz_lm    = (size_t)322 * 1024 * 1024;
    size_t sz_ring  = (size_t)191 * 1024 * 1024;

    printf("[probe_32l] Testing 32 layers (3260 MB) + lm_head + ring buffer...\n");
    void* p0 = rpcmem_alloc(25, 0, sz_half0);
    void* p1 = rpcmem_alloc(25, 0, sz_half1);
    void* plm = rpcmem_alloc(25, 0, sz_lm);
    void* pr = rpcmem_alloc(25, 0, sz_ring);

    if (!p0 || !p1 || !plm || !pr) {
        printf("Alloc failed!\n");
        if (p0) rpcmem_free(p0);
        if (p1) rpcmem_free(p1);
        if (plm) rpcmem_free(plm);
        if (pr) rpcmem_free(pr);
        return 1;
    }

    int fd0 = rpcmem_to_fd(p0);
    int fd1 = rpcmem_to_fd(p1);
    int fdlm = rpcmem_to_fd(plm);
    int fdr = rpcmem_to_fd(pr);

    int rc_lm = fastrpc_mmap(3, fdlm, plm, 0, sz_lm, 0);
    int rc_r  = fastrpc_mmap(3, fdr, pr, 0, sz_ring, 0);
    int rc0   = fastrpc_mmap(3, fd0, p0, 0, sz_half0, 0);
    int rc1   = fastrpc_mmap(3, fd1, p1, 0, sz_half1, 0);

    printf("  rc_lm=%d rc_r=%d rc0=%d rc1=%d\n", rc_lm, rc_r, rc0, rc1);
    size_t total_mapped = (rc_lm == 0 ? sz_lm : 0) + 
                          (rc_r == 0 ? sz_ring : 0) + 
                          (rc0 == 0 ? sz_half0 : 0) + 
                          (rc1 == 0 ? sz_half1 : 0);
    printf("TOTAL: %zu MB (%.2f GB)!\n", total_mapped >> 20, (double)total_mapped / (1024.0*1024.0*1024.0));

    if (rc1 == 0) fastrpc_munmap(3, fd1, p1, sz_half1);
    if (rc0 == 0) fastrpc_munmap(3, fd0, p0, sz_half0);
    if (rc_r == 0) fastrpc_munmap(3, fdr, pr, sz_ring);
    if (rc_lm == 0) fastrpc_munmap(3, fdlm, plm, sz_lm);

    rpcmem_free(pr);
    rpcmem_free(plm);
    rpcmem_free(p1);
    rpcmem_free(p0);
    printf("Done.\n");
    return 0;
}
