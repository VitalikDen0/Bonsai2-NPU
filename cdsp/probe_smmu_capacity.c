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

    size_t chunk_sz = 95 * 1024 * 1024;
    printf("[probe] Testing maximum simultaneous fastrpc_mmap capacity...\n");

    void* ptrs[64] = {0};
    int fds[64] = {0};
    int mapped = 0;
    size_t total_mapped = 0;

    for (int i = 0; i < 64; i++) {
        ptrs[i] = rpcmem_alloc(25, 0, chunk_sz);
        if (!ptrs[i]) {
            printf("[probe] rpcmem_alloc failed at index %d\n", i);
            break;
        }
        fds[i] = rpcmem_to_fd(ptrs[i]);
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, chunk_sz, 0);
        if (rc != 0) {
            printf("[probe] fastrpc_mmap FAILED at chunk %d (rc=%d, total mapped before failure: %zu MB = %.2f GB)\n",
                   i, rc, total_mapped >> 20, (double)total_mapped / (1024.0 * 1024.0 * 1024.0));
            rpcmem_free(ptrs[i]);
            ptrs[i] = NULL;
            break;
        }
        mapped++;
        total_mapped += chunk_sz;
        if ((mapped % 4) == 0) {
            printf("  mapped %d chunks = %zu MB (%.2f GB)\n", mapped, total_mapped >> 20, (double)total_mapped / (1024.0 * 1024.0 * 1024.0));
        }
    }

    printf("[probe] RESULT: Successfully mapped %d chunks simultaneously (%zu MB = %.2f GB in SMMU)!\n",
           mapped, total_mapped >> 20, (double)total_mapped / (1024.0 * 1024.0 * 1024.0));

    for (int i = 0; i < mapped; i++) {
        if (ptrs[i]) {
            fastrpc_munmap(3, fds[i], ptrs[i], chunk_sz);
            rpcmem_free(ptrs[i]);
        }
    }
    printf("[probe] Cleaned up and unmapped all chunks successfully.\n");
    return 0;
}
