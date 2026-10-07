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

    size_t sz_chunk = (size_t)1630 * 1024 * 1024;
    size_t sz_lm = (size_t)322 * 1024 * 1024;

    printf("[probe] Testing allocation of 4 x 1.63 GB (6.52 GB)...\n");
    void* p[4] = {0};
    int fd[4] = {-1, -1, -1, -1};
    int rc[4] = {-1, -1, -1, -1};

    void* plm = rpcmem_alloc(25, 0, sz_lm);
    int fdlm = plm ? rpcmem_to_fd(plm) : -1;
    int rclm = (fdlm >= 0) ? fastrpc_mmap(3, fdlm, plm, 0, sz_lm, 0) : -1;
    printf("lm_head (322 MB): alloc=%p fd=%d mmap_rc=%d\n", plm, fdlm, rclm);

    size_t total_mapped = (rclm == 0 ? sz_lm : 0);
    for (int i = 0; i < 4; i++) {
        p[i] = rpcmem_alloc(25, 0, sz_chunk);
        if (!p[i]) {
            printf("Chunk %d (1.63 GB): ALLOC FAIL\n", i);
            break;
        }
        fd[i] = rpcmem_to_fd(p[i]);
        rc[i] = fastrpc_mmap(3, fd[i], p[i], 0, sz_chunk, 0);
        printf("Chunk %d (1.63 GB, layers %d..%d): fd=%d mmap_rc=%d\n", i, i*16, (i+1)*16-1, fd[i], rc[i]);
        if (rc[i] == 0) total_mapped += sz_chunk;
        else break;
    }

    printf("MAX STABLE SMMU MAPPED: %zu MB (%.2f GiB)!\n", total_mapped >> 20, (double)total_mapped / (1024.0*1024.0*1024.0));

    // Cleanup
    for (int i = 0; i < 4; i++) {
        if (rc[i] == 0) fastrpc_munmap(3, fd[i], p[i], sz_chunk);
        if (p[i]) rpcmem_free(p[i]);
    }
    if (rclm == 0) fastrpc_munmap(3, fdlm, plm, sz_lm);
    if (plm) rpcmem_free(plm);
    printf("[probe] Cleanup done\n");
    return 0;
}
