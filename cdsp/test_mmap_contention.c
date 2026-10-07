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
static volatile int g_stop = 0;

typedef struct {
    void* bits;
    float* x;
    float* y;
} Arg;

static void* worker(void* arg) {
    Arg* a = (Arg*)arg;
    int c = 0;
    while (!g_stop) {
        double t0 = now_ms();
        bonsai_gemv_q1(g_h, 34816, 5120, 1280, a->x, 5120, a->bits, 34816*1280, (const short*)((char*)a->bits + 34816*1280), 34816*40, a->y, 34816);
        double dt = now_ms() - t0;
        if (dt > 3.0) {
            printf("  [worker GEMV SLOW] call %d took %.2f ms!\n", c, dt);
        }
        c++;
        usleep(200);
    }
    return NULL;
}

int main(void) {
    setbuf(stdout, NULL);
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp/bonsai1bit;/data/local/tmp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();
    bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &g_h);

    size_t sz = (size_t)1500 * 1024 * 1024;
    void* g0 = rpcmem_alloc(25, 0, sz);
    void* g1 = rpcmem_alloc(25, 0, sz);
    float* x = (float*)rpcmem_alloc(25, 1, 5120*4);
    float* y = (float*)rpcmem_alloc(25, 1, 34816*4);
    int fd0 = rpcmem_to_fd(g0);
    int fd1 = rpcmem_to_fd(g1);
    fastrpc_mmap(3, rpcmem_to_fd(x), x, 0, 5120*4, 0);
    fastrpc_mmap(3, rpcmem_to_fd(y), y, 0, 34816*4, 0);
    fastrpc_mmap(3, fd0, g0, 0, sz, 0);

    Arg a = { g0, x, y };
    pthread_t th;
    pthread_create(&th, NULL, worker, &a);
    usleep(10000); // let worker run a few calls

    printf("[main] Calling fastrpc_mmap(g1, 1500 MB)...\n");
    double tm0 = now_ms();
    fastrpc_mmap(3, fd1, g1, 0, sz, 0);
    double tm1 = now_ms();
    printf("[main] fastrpc_mmap took %.2f ms\n", tm1 - tm0);

    usleep(10000);
    g_stop = 1;
    pthread_join(th, NULL);

    fastrpc_munmap(3, fd1, g1, sz);
    fastrpc_munmap(3, fd0, g0, sz);
    bonsai_close(g_h);
    rpcmem_free(g0); rpcmem_free(g1); rpcmem_free(x); rpcmem_free(y);
    return 0;
}
