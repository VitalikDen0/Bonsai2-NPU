#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);

static long get_mem_avail_mb(void) {
    FILE* f = fopen("/proc/meminfo", "r");
    if (!f) return -1;
    char line[256];
    long avail_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %ld kB", &avail_kb) == 1) break;
    }
    fclose(f);
    return avail_kb / 1024;
}

int main(void) {
    setbuf(stdout, NULL);
    printf("[probe_4g] Starting 4-group DMA-BUF allocation probe. MemAvail: %ld MB\n", get_mem_avail_mb());

    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    // 4 groups of 16 layers each
    // Total model size = ~6.4 GB (excluding embed_tokens)
    size_t sz_g0 = (size_t)1540 * 1024 * 1024;
    size_t sz_g1 = (size_t)1450 * 1024 * 1024;
    size_t sz_g2 = (size_t)1540 * 1024 * 1024;
    size_t sz_g3 = (size_t)1560 * 1024 * 1024;
    size_t sz_lm = (size_t)322  * 1024 * 1024;

    printf("[probe_4g] Allocating Group 0 (1540 MB)... MemAvail: %ld MB\n", get_mem_avail_mb());
    void* g0 = rpcmem_alloc(25, 0, sz_g0);
    printf("  g0=%p (fd=%d), MemAvail: %ld MB\n", g0, g0 ? rpcmem_to_fd(g0) : -1, get_mem_avail_mb());

    printf("[probe_4g] Allocating Group 1 (1450 MB)... MemAvail: %ld MB\n", get_mem_avail_mb());
    void* g1 = rpcmem_alloc(25, 0, sz_g1);
    printf("  g1=%p (fd=%d), MemAvail: %ld MB\n", g1, g1 ? rpcmem_to_fd(g1) : -1, get_mem_avail_mb());

    printf("[probe_4g] Allocating Group 2 (1540 MB)... MemAvail: %ld MB\n", get_mem_avail_mb());
    void* g2 = rpcmem_alloc(25, 0, sz_g2);
    printf("  g2=%p (fd=%d), MemAvail: %ld MB\n", g2, g2 ? rpcmem_to_fd(g2) : -1, get_mem_avail_mb());

    printf("[probe_4g] Allocating Group 3 (1560 MB)... MemAvail: %ld MB\n", get_mem_avail_mb());
    void* g3 = rpcmem_alloc(25, 0, sz_g3);
    printf("  g3=%p (fd=%d), MemAvail: %ld MB\n", g3, g3 ? rpcmem_to_fd(g3) : -1, get_mem_avail_mb());

    printf("[probe_4g] Allocating Static LM Head (322 MB)... MemAvail: %ld MB\n", get_mem_avail_mb());
    void* lm = rpcmem_alloc(25, 0, sz_lm);
    printf("  lm=%p (fd=%d), MemAvail: %ld MB\n", lm, lm ? rpcmem_to_fd(lm) : -1, get_mem_avail_mb());

    int all_ok = (g0 && g1 && g2 && g3 && lm);
    printf("[probe_4g] SUMMARY: All 4 Groups + LM Head allocated simultaneously: %s!\n", all_ok ? "SUCCESS" : "FAILED");
    if (all_ok) {
        size_t tot = sz_g0 + sz_g1 + sz_g2 + sz_g3 + sz_lm;
        printf("[probe_4g] TOTAL ALLOCATED DMA-BUF: %zu MB (%.2f GB), Remaining MemAvail: %ld MB\n",
               tot >> 20, (double)tot / (1024.0*1024.0*1024.0), get_mem_avail_mb());
    }

    if (lm) rpcmem_free(lm);
    if (g3) rpcmem_free(g3);
    if (g2) rpcmem_free(g2);
    if (g1) rpcmem_free(g1);
    if (g0) rpcmem_free(g0);
    printf("[probe_4g] All freed. Final MemAvail: %ld MB\n", get_mem_avail_mb());
    return all_ok ? 0 : 1;
}
