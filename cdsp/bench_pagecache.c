#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

int main(void) {
    setbuf(stdout, NULL);
    const char* path = "/data/local/tmp/bonsai1bit/bonsai2-27b.npubin";
    int fd = open(path, O_RDONLY);
    if (fd < 0) { printf("Open fail\n"); return 1; }
    struct stat st;
    fstat(fd, &st);
    size_t sz = st.st_size;
    uint8_t* base = (uint8_t*)mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { printf("mmap fail\n"); return 1; }

    size_t off = (size_t)3400 * 1024 * 1024;
    size_t len = (size_t)3260 * 1024 * 1024;
    if (off + len > sz) len = sz - off;

    printf("[bench] Warming up pagecache with madvise(MADV_WILLNEED)...\n");
    double tw0 = now_ms();
    madvise(base + off, len, MADV_WILLNEED);
    volatile uint64_t dummy = 0;
    for (size_t p = 0; p < len; p += 4096) {
        dummy += base[off + p];
    }
    double tw1 = now_ms();
    printf("[bench] Warmup took %.1f ms (dummy=%llu)\n", tw1 - tw0, (unsigned long long)dummy);

    uint8_t* dst = malloc(100 * 1024 * 1024);
    for (int iter = 0; iter < 3; iter++) {
        double tm0 = now_ms();
        size_t copied = 0;
        volatile uint64_t acc = 0;
        for (size_t p = 0; p < len; p += 100 * 1024 * 1024) {
            size_t chunk = len - p < 100 * 1024 * 1024 ? len - p : 100 * 1024 * 1024;
            memcpy(dst, base + off + p, chunk);
            acc += ((uint64_t*)dst)[0] + ((uint64_t*)dst)[chunk/8 - 1];
            copied += chunk;
        }
        double tm1 = now_ms();
        double ms = tm1 - tm0;
        printf("[bench] Iter %d: Copied %.2f GB in %.1f ms (Bandwidth: %.1f GB/s, acc=%llu)\n",
               iter, (double)copied / (1024.0*1024.0*1024.0), ms,
               ms > 0 ? (double)copied / (ms / 1000.0) / (1024.0*1024.0*1024.0) : 0.0,
               (unsigned long long)acc);
    }

    free(dst);
    munmap(base, sz);
    close(fd);
    return 0;
}
