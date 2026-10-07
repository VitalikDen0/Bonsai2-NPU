#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>

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

#define NUM_LAYERS 64
#define PHASE_SZ 8
#define NUM_PHASES (NUM_LAYERS / PHASE_SZ) // 8 phases

static void* g_ptrs[NUM_LAYERS];
static int g_fds[NUM_LAYERS];
static size_t g_sz = 95 * 1024 * 1024;

static void unmap_phase(int p) {
    int start = p * PHASE_SZ;
    for (int i = 0; i < PHASE_SZ; i++) {
        fastrpc_munmap(3, g_fds[start + i], g_ptrs[start + i], g_sz);
    }
}

static int map_phase(int p) {
    int start = p * PHASE_SZ;
    for (int i = 0; i < PHASE_SZ; i++) {
        int rc = fastrpc_mmap(3, g_fds[start + i], g_ptrs[start + i], 0, g_sz, 0);
        if (rc != 0) return rc;
    }
    return 0;
}

int main(void) {
    setbuf(stdout, NULL);
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    printf("[probe_pipe] 1. Allocating 64 layers in rpcmem (5.94 GB)...\n");
    double t0 = now_ms();
    for (int i = 0; i < NUM_LAYERS; i++) {
        g_ptrs[i] = rpcmem_alloc(25, 0, g_sz);
        if (!g_ptrs[i]) {
            printf("Alloc failed at %d\n", i);
            return 1;
        }
        g_fds[i] = rpcmem_to_fd(g_ptrs[i]);
    }
    printf("[probe_pipe] Allocated in %.2f ms\n", now_ms() - t0);

    // Initial map: Phase 0 (0..7) and Phase 1 (8..15)
    printf("[probe_pipe] 2. Initial mapping of Phase 0 & 1 (16 layers = 1.48 GB)...\n");
    t0 = now_ms();
    if (map_phase(0) != 0 || map_phase(1) != 0) {
        printf("Initial map failed!\n");
        return 1;
    }
    printf("[probe_pipe] Phase 0 & 1 mapped in %.2f ms\n", now_ms() - t0);

    // Test a token cycle:
    // When executing phase p:
    // If p < NUM_PHASES - 1: background maps phase p+1, unmaps phase p-1 (if p > 0)
    printf("[probe_pipe] 3. Testing pipelined phase transitions:\n");
    double total_pipe = 0;
    for (int p = 0; p < NUM_PHASES; p++) {
        double tp = now_ms();
        if (p >= 1 && p + 1 < NUM_PHASES) {
            // Unmap phase p-1, map phase p+1
            unmap_phase(p - 1);
            int rc = map_phase(p + 1);
            if (rc != 0) printf("Map phase %d failed (rc=%d)\n", p + 1, rc);
        } else if (p == NUM_PHASES - 1) {
            // Final phase: unmap p-1, map Phase 0 (ready for next token)
            unmap_phase(p - 1);
            int rc = map_phase(0);
            if (rc != 0) printf("Wrap map phase 0 failed (rc=%d)\n", rc);
        }
        double elapsed = now_ms() - tp;
        total_pipe += elapsed;
        printf("  Phase %d transition time: %.2f ms\n", p, elapsed);
    }

    printf("[probe_pipe] Total transition overhead: %.2f ms across 8 phases (avg: %.2f ms/phase)\n",
           total_pipe, total_pipe / NUM_PHASES);

    // Wrap unmap
    unmap_phase(NUM_PHASES - 1);
    unmap_phase(0);
    for (int i = 0; i < NUM_LAYERS; i++) rpcmem_free(g_ptrs[i]);
    printf("[probe_pipe] Done cleanly!\n");
    return 0;
}
