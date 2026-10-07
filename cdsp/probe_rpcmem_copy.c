#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <arm_neon.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

// Streaming NEON copy using LDNP / STNP (non-temporal pairs)
static void neon_stnp_copy(void* dst, const void* src, size_t bytes) {
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* d = (uint8_t*)dst;
    size_t n = bytes / 128;
    for (size_t i = 0; i < n; i++) {
        float32x4_t q0 = vld1q_f32((const float*)(s + 0));
        float32x4_t q1 = vld1q_f32((const float*)(s + 16));
        float32x4_t q2 = vld1q_f32((const float*)(s + 32));
        float32x4_t q3 = vld1q_f32((const float*)(s + 48));
        float32x4_t q4 = vld1q_f32((const float*)(s + 64));
        float32x4_t q5 = vld1q_f32((const float*)(s + 80));
        float32x4_t q6 = vld1q_f32((const float*)(s + 96));
        float32x4_t q7 = vld1q_f32((const float*)(s + 112));

        vst1q_f32((float*)(d + 0), q0);
        vst1q_f32((float*)(d + 16), q1);
        vst1q_f32((float*)(d + 32), q2);
        vst1q_f32((float*)(d + 48), q3);
        vst1q_f32((float*)(d + 64), q4);
        vst1q_f32((float*)(d + 80), q5);
        vst1q_f32((float*)(d + 96), q6);
        vst1q_f32((float*)(d + 112), q7);

        s += 128;
        d += 128;
    }
    size_t rem = bytes % 128;
    if (rem > 0) memcpy(d, s, rem);
}

int main(void) {
    setbuf(stdout, NULL);
    rpcmem_init();
    size_t test_sz = 95 * 1024 * 1024; // 1 layer = 95 MB

    const char* path = "/data/local/tmp/bonsai1bit/bonsai2-27b.npubin";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("Open fail\n"); return 1; }
    uint8_t* src = (uint8_t*)mmap(NULL, test_sz, PROT_READ, MAP_SHARED, fd, (off_t)(3400ULL * 1024 * 1024));
    if (src == MAP_FAILED) { printf("mmap fail\n"); return 1; }
    madvise(src, test_sz, MADV_WILLNEED);

    printf("[probe] Benchmarking 95 MB copy from pagecache into different destinations...\n");

    // 1. Host malloc
    uint8_t* dst_malloc = malloc(test_sz);
    for (int i = 0; i < 3; i++) {
        double t0 = now_ms();
        memcpy(dst_malloc, src, test_sz);
        double t1 = now_ms();
        printf("  malloc dst (memcpy): %.2f ms (%.1f GB/s)\n", t1 - t0, (double)test_sz / ((t1 - t0) * 1e-3) / (1024*1024*1024));
    }
    free(dst_malloc);

    // 2. rpcmem uncached (flags = 0)
    uint8_t* dst_uncached = (uint8_t*)rpcmem_alloc(25, 0, test_sz);
    if (dst_uncached) {
        for (int i = 0; i < 3; i++) {
            double t0 = now_ms();
            memcpy(dst_uncached, src, test_sz);
            double t1 = now_ms();
            printf("  rpcmem UNCACHED flags=0 (libc memcpy): %.2f ms (%.1f GB/s)\n", t1 - t0, (double)test_sz / ((t1 - t0) * 1e-3) / (1024*1024*1024));
        }
        for (int i = 0; i < 3; i++) {
            double t0 = now_ms();
            neon_stnp_copy(dst_uncached, src, test_sz);
            double t1 = now_ms();
            printf("  rpcmem UNCACHED flags=0 (NEON vector copy): %.2f ms (%.1f GB/s)\n", t1 - t0, (double)test_sz / ((t1 - t0) * 1e-3) / (1024*1024*1024));
        }
        rpcmem_free(dst_uncached);
    } else {
        printf("  rpcmem alloc flags=0 failed\n");
    }

    // 3. rpcmem cached (flags = 1)
    uint8_t* dst_cached = (uint8_t*)rpcmem_alloc(25, 1, test_sz);
    if (dst_cached) {
        for (int i = 0; i < 3; i++) {
            double t0 = now_ms();
            memcpy(dst_cached, src, test_sz);
            double t1 = now_ms();
            printf("  rpcmem CACHED flags=1 (libc memcpy): %.2f ms (%.1f GB/s)\n", t1 - t0, (double)test_sz / ((t1 - t0) * 1e-3) / (1024*1024*1024));
        }
        rpcmem_free(dst_cached);
    } else {
        printf("  rpcmem alloc flags=1 failed\n");
    }

    munmap(src, test_sz);
    close(fd);
    return 0;
}
