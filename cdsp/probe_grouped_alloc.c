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

    // 8 groups of 8 layers. Each layer is ~95 MB.
    // Let's test allocating 8 buffers of 760 MB each (= 6.08 GB total).
    const int NUM_GROUPS = 8;
    size_t group_sz = (size_t)760 * 1024 * 1024;

    printf("[probe_grouped] Allocating and mapping %d buffers of %zu MB each (Total: %.2f GB)...\n",
           NUM_GROUPS, group_sz >> 20, (double)(NUM_GROUPS * group_sz) / (1024.0 * 1024.0 * 1024.0));

    void* ptrs[NUM_GROUPS];
    int fds[NUM_GROUPS];
    int allocated = 0;
    int mapped = 0;

    for (int i = 0; i < NUM_GROUPS; i++) {
        ptrs[i] = rpcmem_alloc(25, 0, group_sz);
        if (!ptrs[i]) {
            printf("[probe_grouped] rpcmem_alloc failed at group %d (total alloc: %zu MB)\n",
                   i, (size_t)allocated * group_sz >> 20);
            break;
        }
        allocated++;
        fds[i] = rpcmem_to_fd(ptrs[i]);
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, group_sz, 0);
        if (rc != 0) {
            printf("[probe_grouped] fastrpc_mmap failed at group %d (rc=%d)\n", i, rc);
            break;
        }
        mapped++;
        printf("  Group %d/%d mapped successfully (%zu MB total in SMMU)\n",
               mapped, NUM_GROUPS, (size_t)mapped * group_sz >> 20);
    }

    printf("[probe_grouped] RESULT: Allocated %d/%d, Mapped %d/%d into FastRPC SMMU!\n",
           allocated, NUM_GROUPS, mapped, NUM_GROUPS);

    for (int i = 0; i < mapped; i++) {
        fastrpc_munmap(3, fds[i], ptrs[i], group_sz);
    }
    for (int i = 0; i < allocated; i++) {
        rpcmem_free(ptrs[i]);
    }
    printf("[probe_grouped] Cleanup finished cleanly.\n");
    return 0;
}
