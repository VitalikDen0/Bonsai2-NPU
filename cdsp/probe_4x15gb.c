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

    // 16 layers * 95 MB = 1520 MB = 1.484 GiB
    size_t sz_16l = (size_t)16 * 95 * 1024 * 1024;
    printf("[probe_4x15] Testing allocation of 4 x 1.48 GB buffers (Total: 5.94 GB)...\n");

    void* ptrs[4] = {0};
    int fds[4] = {0};
    int alloc_count = 0;

    for (int i = 0; i < 4; i++) {
        double t0 = now_ms();
        ptrs[i] = rpcmem_alloc(25, 0, sz_16l);
        if (!ptrs[i]) {
            printf("[probe_4x15] Alloc failed at buffer %d!\n", i);
            break;
        }
        fds[i] = rpcmem_to_fd(ptrs[i]);
        alloc_count++;
        printf("  Buffer %d (1.48 GB) allocated in %.2f ms (fd=%d)\n", i, now_ms() - t0, fds[i]);
    }

    if (alloc_count == 4) {
        printf("[probe_4x15] ALL 4 BUFFERS (5.94 GB) ALLOCATED SUCCESSFULLY!\n");
        printf("[probe_4x15] Now testing FastRPC SMMU mapping & unmapping speed for 1.48 GB chunks:\n");

        // Map buffer 0
        double tm0 = now_ms();
        int rc0 = fastrpc_mmap(3, fds[0], ptrs[0], 0, sz_16l, 0);
        double tm0_time = now_ms() - tm0;
        printf("  fastrpc_mmap(buffer 0, 1.48 GB): rc=%d in %.3f ms\n", rc0, tm0_time);

        // Map buffer 1
        double tm1 = now_ms();
        int rc1 = fastrpc_mmap(3, fds[1], ptrs[1], 0, sz_16l, 0);
        double tm1_time = now_ms() - tm1;
        printf("  fastrpc_mmap(buffer 1, 1.48 GB): rc=%d in %.3f ms\n", rc1, tm1_time);

        // Unmap buffer 0
        double tu0 = now_ms();
        int ru0 = fastrpc_munmap(3, fds[0], ptrs[0], sz_16l);
        double tu0_time = now_ms() - tu0;
        printf("  fastrpc_munmap(buffer 0, 1.48 GB): rc=%d in %.3f ms\n", ru0, tu0_time);

        // Map buffer 2
        double tm2 = now_ms();
        int rc2 = fastrpc_mmap(3, fds[2], ptrs[2], 0, sz_16l, 0);
        double tm2_time = now_ms() - tm2;
        printf("  fastrpc_mmap(buffer 2, 1.48 GB): rc=%d in %.3f ms\n", rc2, tm2_time);

        // Unmap buffer 1
        fastrpc_munmap(3, fds[1], ptrs[1], sz_16l);
        // Unmap buffer 2
        fastrpc_munmap(3, fds[2], ptrs[2], sz_16l);
    }

    for (int i = 0; i < alloc_count; i++) {
        if (ptrs[i]) rpcmem_free(ptrs[i]);
    }
    printf("[probe_4x15] Done cleanly.\n");
    return (alloc_count == 4) ? 0 : 1;
}
