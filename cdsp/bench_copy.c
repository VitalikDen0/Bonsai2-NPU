#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <arm_neon.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// NEON unrolled copy 64 bytes per iteration
static void neon_memcpy_64(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        uint8x16x4_t v = vld1q_u8_x4(s + i);
        vst1q_u8_x4(d + i, v);
    }
    for (; i < n; i++) d[i] = s[i];
}

typedef struct {
    uint8_t* dst;
    const uint8_t* src;
    size_t len;
    int cpu;
} ThreadCopyArg;

static void* thread_copy_fn(void* p) {
    ThreadCopyArg* a = (ThreadCopyArg*)p;
    if (a->cpu >= 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(a->cpu, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
    }
    neon_memcpy_64(a->dst, a->src, a->len);
    return NULL;
}

int main(void) {
    size_t sz = 100 * 1024 * 1024; // 100 MB (~1 layer)
    uint8_t* src = (uint8_t*)malloc(sz);
    uint8_t* dst = (uint8_t*)malloc(sz);
    memset(src, 0x5a, sz);
    memset(dst, 0, sz);

    printf("=== BENCHMARK: 100 MB Copy (1 Bonsai Layer) ===\n");

    // 1. Standard libc memcpy (unpinned)
    double t0 = now_ms();
    for (int it = 0; it < 10; it++) memcpy(dst, src, sz);
    double t1 = now_ms();
    double avg1 = (t1 - t0) / 10.0;
    printf("1. Standard memcpy (unpinned): %.2f ms (%.2f GB/s)\n", avg1, (sz / (1024.0*1024.0*1024.0)) / (avg1 / 1000.0));

    // 2. NEON copy pinned to Prime Core (cpu 6)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    t0 = now_ms();
    for (int it = 0; it < 10; it++) neon_memcpy_64(dst, src, sz);
    t1 = now_ms();
    double avg2 = (t1 - t0) / 10.0;
    printf("2. NEON copy (pinned to CPU 6 @ 4.3 GHz): %.2f ms (%.2f GB/s)\n", avg2, (sz / (1024.0*1024.0*1024.0)) / (avg2 / 1000.0));

    // 3. 2-Thread parallel NEON copy (CPU 6 and CPU 7)
    t0 = now_ms();
    for (int it = 0; it < 10; it++) {
        pthread_t t;
        ThreadCopyArg a1 = { dst, src, sz / 2, 6 };
        ThreadCopyArg a2 = { dst + sz / 2, src + sz / 2, sz / 2, 7 };
        pthread_create(&t, NULL, thread_copy_fn, &a2);
        thread_copy_fn(&a1);
        pthread_join(t, NULL);
    }
    t1 = now_ms();
    double avg3 = (t1 - t0) / 10.0;
    printf("3. 2-Thread parallel NEON (CPU 6 + CPU 7): %.2f ms (%.2f GB/s)\n", avg3, (sz / (1024.0*1024.0*1024.0)) / (avg3 / 1000.0));

    // 4. 4-Thread parallel NEON copy (CPU 4, 5, 6, 7)
    t0 = now_ms();
    for (int it = 0; it < 10; it++) {
        pthread_t t[3];
        ThreadCopyArg a[4];
        size_t chunk = sz / 4;
        int cpus[4] = { 4, 5, 6, 7 };
        for (int k = 0; k < 4; k++) {
            a[k].dst = dst + k * chunk;
            a[k].src = src + k * chunk;
            a[k].len = chunk;
            a[k].cpu = cpus[k];
            if (k < 3) pthread_create(&t[k], NULL, thread_copy_fn, &a[k]);
        }
        thread_copy_fn(&a[3]);
        for (int k = 0; k < 3; k++) pthread_join(t[k], NULL);
    }
    t1 = now_ms();
    double avg4 = (t1 - t0) / 10.0;
    printf("4. 4-Thread parallel NEON (CPU 4, 5, 6, 7): %.2f ms (%.2f GB/s)\n", avg4, (sz / (1024.0*1024.0*1024.0)) / (avg4 / 1000.0));

    free(src); free(dst);
    return 0;
}
