#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

int bonsai_gemv_q1(unsigned long long h, int out_dim, int in_dim, int prow,
                   const float* x, int xLen, const unsigned char* bits, int bitsLen,
                   const short* scales, int scalesLen, float* y, int yLen);
int bonsai_open(const char* uri, unsigned long long* handle);
int bonsai_close(unsigned long long handle);

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static unsigned long long g_h = 0;
static volatile int g_thread_running = 0;
static volatile int g_gemv_count = 0;
static volatile int g_gemv_errors = 0;

typedef struct {
    void* bits;
    void* scales;
    float* x;
    float* y;
    int total_calls;
} ThreadArg;

static void* gemv_worker_fn(void* arg) {
    ThreadArg* a = (ThreadArg*)arg;
    g_thread_running = 1;
    // out_dim=34816, in_dim=5120, prow=1280, ng=40
    int out_dim = 34816;
    int in_dim = 5120;
    int prow = 1280;
    int ng = 40;

    for (int i = 0; i < a->total_calls; i++) {
        double t0 = now_ms();
        int rc = bonsai_gemv_q1(g_h, out_dim, in_dim, prow,
                                a->x, in_dim,
                                (const unsigned char*)a->bits, out_dim * prow,
                                (const short*)a->scales, out_dim * ng,
                                a->y, out_dim);
        double dt = now_ms() - t0;
        if (rc != 0) {
            g_gemv_errors++;
            printf("[worker] Call %d failed rc=%d\n", i, rc);
        } else {
            g_gemv_count++;
        }
        usleep(500); // 0.5ms between calls
    }
    g_thread_running = 0;
    return NULL;
}

int main(void) {
    setbuf(stdout, NULL);
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/bonsai1bit;/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    if (bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &g_h) != 0) {
        printf("Failed to open DSP handle\n");
        return 1;
    }

    size_t sz_group = (size_t)16 * 95 * 1024 * 1024; // 1.48 GB
    printf("[probe_async] Allocating Group 0 and Group 1 (%zu MB each)...\n", sz_group >> 20);

    void* g0 = rpcmem_alloc(25, 0, sz_group);
    void* g1 = rpcmem_alloc(25, 0, sz_group);
    float* x = (float*)rpcmem_alloc(25, 1, 5120 * 4);
    float* y = (float*)rpcmem_alloc(25, 1, 34816 * 4);

    if (!g0 || !g1 || !x || !y) {
        printf("Alloc failed!\n");
        return 1;
    }
    int fd0 = rpcmem_to_fd(g0);
    int fd1 = rpcmem_to_fd(g1);
    int fdx = rpcmem_to_fd(x);
    int fdy = rpcmem_to_fd(y);

    fastrpc_mmap(3, fdx, x, 0, 5120 * 4, 0);
    fastrpc_mmap(3, fdy, y, 0, 34816 * 4, 0);

    // Map Group 0
    double t_m0 = now_ms();
    int rc0 = fastrpc_mmap(3, fd0, g0, 0, sz_group, 0);
    printf("[probe_async] Group 0 mapped in %.2f ms (rc=%d)\n", now_ms() - t_m0, rc0);

    // Launch background GEMV thread running on Group 0
    ThreadArg targ = { g0, (char*)g0 + 34816 * 1280, x, y, 40 };
    pthread_t tid;
    pthread_create(&tid, NULL, gemv_worker_fn, &targ);

    // Wait until worker starts running
    while (!g_thread_running) usleep(100);
    usleep(5000); // let it execute a few GEMVs

    printf("[probe_async] Main thread: attempting fastrpc_mmap of Group 1 WHILE DSP is busy (gemv_count=%d)...\n", g_gemv_count);
    double t_async_mmap = now_ms();
    int rc1 = fastrpc_mmap(3, fd1, g1, 0, sz_group, 0);
    double async_mmap_time = now_ms() - t_async_mmap;
    printf("[probe_async] RESULT: fastrpc_mmap(Group 1) returned rc=%d in %.2f ms (gemv_count during mmap=%d)\n",
           rc1, async_mmap_time, g_gemv_count);

    pthread_join(tid, NULL);
    printf("[probe_async] Main thread: attempting fastrpc_munmap of Group 0...\n");
    double t_unmap = now_ms();
    int rcu0 = fastrpc_munmap(3, fd0, g0, sz_group);
    printf("[probe_async] Group 0 unmapped in %.2f ms (rc=%d)\n", now_ms() - t_unmap, rcu0);

    fastrpc_munmap(3, fd1, g1, sz_group);
    fastrpc_munmap(3, fdx, x, 5120 * 4);
    fastrpc_munmap(3, fdy, y, 34816 * 4);
    bonsai_close(g_h);
    rpcmem_free(g0);
    rpcmem_free(g1);
    rpcmem_free(x);
    rpcmem_free(y);

    printf("[probe_async] Test completed: total gemvs=%d, errors=%d\n", g_gemv_count, g_gemv_errors);
    return (rc1 == 0 && g_gemv_errors == 0) ? 0 : 1;
}
