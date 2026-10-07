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

    printf("[probe_limits] Testing 1: Can we map a single large 1.5 GB buffer?\n");
    size_t sz_1_5gb = (size_t)1500 * 1024 * 1024;
    void* p1 = rpcmem_alloc(25, 0, sz_1_5gb);
    if (p1) {
        int fd1 = rpcmem_to_fd(p1);
        int rc1 = fastrpc_mmap(3, fd1, p1, 0, sz_1_5gb, 0);
        printf("[probe_limits] 1.5 GB single buffer mmap rc=%d\n", rc1);
        if (rc1 == 0) fastrpc_munmap(3, fd1, p1, sz_1_5gb);
        rpcmem_free(p1);
    } else {
        printf("[probe_limits] alloc 1.5 GB failed\n");
    }

    printf("[probe_limits] Testing 2: Can we map a single large 3.0 GB buffer?\n");
    size_t sz_3gb = (size_t)3000 * 1024 * 1024;
    void* p2 = rpcmem_alloc(25, 0, sz_3gb);
    if (p2) {
        int fd2 = rpcmem_to_fd(p2);
        int rc2 = fastrpc_mmap(3, fd2, p2, 0, sz_3gb, 0);
        printf("[probe_limits] 3.0 GB single buffer mmap rc=%d\n", rc2);
        if (rc2 == 0) fastrpc_munmap(3, fd2, p2, sz_3gb);
        rpcmem_free(p2);
    } else {
        printf("[probe_limits] alloc 3.0 GB failed\n");
    }

    printf("[probe_limits] Testing 3: Max count of small (1 MB) mappings:\n");
    int count = 0;
    void* ptrs[128] = {0};
    int fds[128] = {0};
    for (int i = 0; i < 128; i++) {
        ptrs[i] = rpcmem_alloc(25, 0, 1024 * 1024);
        if (!ptrs[i]) break;
        fds[i] = rpcmem_to_fd(ptrs[i]);
        int rc = fastrpc_mmap(3, fds[i], ptrs[i], 0, 1024 * 1024, 0);
        if (rc != 0) {
            printf("[probe_limits] small map failed at count=%d with rc=%d\n", i, rc);
            rpcmem_free(ptrs[i]);
            break;
        }
        count++;
    }
    printf("[probe_limits] small 1 MB mappings succeeded count=%d\n", count);
    for (int i = 0; i < count; i++) {
        fastrpc_munmap(3, fds[i], ptrs[i], 1024 * 1024);
        rpcmem_free(ptrs[i]);
    }

    printf("[probe_limits] DONE\n");
    return 0;
}
