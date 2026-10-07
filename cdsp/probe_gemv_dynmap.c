#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);

int bonsai_open(const char* uri, unsigned long long* handle);
int bonsai_close(unsigned long long handle);
int bonsai_gemv_q1(unsigned long long h, int out_dim, int in_dim, int prow,
                   const float* x, int xLen, const unsigned char* bits, int bitsLen,
                   const short* scales, int scalesLen, float* y, int yLen);

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    unsigned long long h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc != 0) { printf("[probe] FAIL bonsai_open rc=%d\n", rc); return 1; }
    printf("[probe] bonsai_open OK handle=%llu\n", (unsigned long long)h);

    int in_dim = 5120;
    int out_dim = 5120;
    int prow = 1280; // ternary Q2: 2 * (5120/8) = 1280 bytes
    int ng = in_dim / 128; // 40
    size_t bits_sz = (size_t)out_dim * prow; // 6,553,600 bytes
    size_t scales_sz = (size_t)out_dim * ng * 2; // 409,600 bytes
    size_t layer_sz = bits_sz + scales_sz;

    float* x = (float*)rpcmem_alloc(25, 1, in_dim * sizeof(float));
    float* y = (float*)rpcmem_alloc(25, 1, out_dim * sizeof(float));
    fastrpc_mmap(3, rpcmem_to_fd(x), x, 0, in_dim * sizeof(float), 0);
    fastrpc_mmap(3, rpcmem_to_fd(y), y, 0, out_dim * sizeof(float), 0);

    for (int i = 0; i < in_dim; i++) x[i] = 0.01f * (i % 7);

    void* layer_buf = rpcmem_alloc(25, 0, layer_sz);
    int layer_fd = rpcmem_to_fd(layer_buf);
    unsigned char* bits = (unsigned char*)layer_buf;
    short* scales = (short*)((char*)layer_buf + bits_sz);
    memset(bits, 0x55, bits_sz);
    for (int i = 0; i < out_dim * ng; i++) scales[i] = (short)0x3800; // 1.0f in fp16

    printf("[probe] Testing dynamic fastrpc_mmap -> GEMV -> munmap loop...\n");
    for (int it = 0; it < 5; it++) {
        double t0 = now_us();
        rc = fastrpc_mmap(3, layer_fd, layer_buf, 0, layer_sz, 0);
        double t_map = now_us();

        rc = bonsai_gemv_q1(h, out_dim, in_dim, prow, x, in_dim, bits, bits_sz, scales, scales_sz / 2, y, out_dim);
        double t_gemv = now_us();

        fastrpc_munmap(3, layer_fd, layer_buf, layer_sz);
        double t_unmap = now_us();

        printf("  iter %d: mmap=%.1fus gemv=%.1fus munmap=%.1fus (y[0]=%.4f y[10]=%.4f)\n",
               it, t_map - t0, t_gemv - t_map, t_unmap - t_gemv, y[0], y[10]);
    }

    fastrpc_munmap(3, rpcmem_to_fd(x), x, in_dim * sizeof(float));
    fastrpc_munmap(3, rpcmem_to_fd(y), y, out_dim * sizeof(float));
    rpcmem_free(x);
    rpcmem_free(y);
    rpcmem_free(layer_buf);
    bonsai_close(h);
    printf("[probe] TEST COMPLETED SUCCESSFULLY!\n");
    return 0;
}
