#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    rpcmem_init();
    const char* path = "/data/local/tmp/bonsai1bit/bonsai2-27b.npubin";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("Failed to open %s\n", path); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t fsz = st.st_size;
    const uint8_t* base = (const uint8_t*)mmap(NULL, fsz, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { printf("mmap failed\n"); return 1; }

    size_t sz = 100 * 1024 * 1024; // 100 MB
    void* rpc_unc = rpcmem_alloc(25, 0, sz); // uncached
    void* rpc_cch = rpcmem_alloc(25, 1, sz); // cached
    void* mal_dst = malloc(sz);

    printf("=== BENCHMARK: 100 MB from Model File mmap ===\n");

    // Test 1: mmap to regular malloc
    double t0 = now_ms();
    for (int i = 0; i < 5; i++) memcpy(mal_dst, base + (size_t)i * sz + 1024*1024, sz);
    double t1 = now_ms();
    printf("1. mmap -> malloc (RAM): %.2f ms (%.2f GB/s)\n", (t1 - t0) / 5.0, (sz / (1024.0*1024.0*1024.0)) / ((t1 - t0) / 5000.0));

    // Test 2: mmap to rpcmem (cached, flag=1)
    t0 = now_ms();
    for (int i = 0; i < 5; i++) memcpy(rpc_cch, base + (size_t)i * sz + 1024*1024, sz);
    t1 = now_ms();
    printf("2. mmap -> rpcmem (cached, flag=1): %.2f ms (%.2f GB/s)\n", (t1 - t0) / 5.0, (sz / (1024.0*1024.0*1024.0)) / ((t1 - t0) / 5000.0));

    // Test 3: mmap to rpcmem (uncached, flag=0)
    t0 = now_ms();
    for (int i = 0; i < 5; i++) memcpy(rpc_unc, base + (size_t)i * sz + 1024*1024, sz);
    t1 = now_ms();
    printf("3. mmap -> rpcmem (uncached, flag=0): %.2f ms (%.2f GB/s)\n", (t1 - t0) / 5.0, (sz / (1024.0*1024.0*1024.0)) / ((t1 - t0) / 5000.0));

    if (rpc_unc) rpcmem_free(rpc_unc);
    if (rpc_cch) rpcmem_free(rpc_cch);
    free(mal_dst);
    munmap((void*)base, fsz);
    close(fd);
    return 0;
}
