#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int remote_session_control(unsigned, void*, unsigned);

int main(void) {
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();

    size_t sz = 95 * 1024 * 1024;
    void* ptrs[64] = {0};
    int fds[64] = {0};

    printf("[probe] Testing allocation of 64 x 95 MB layers (~6.08 GB)...\n");
    int success = 0;
    for (int i = 0; i < 64; i++) {
        ptrs[i] = rpcmem_alloc(25, 0, sz);
        if (!ptrs[i]) {
            printf("[probe] alloc failed at layer %d! (allocated %d layers = %.2f GB)\n",
                   i, i, (double)i * 95.0 / 1024.0);
            break;
        }
        fds[i] = rpcmem_to_fd(ptrs[i]);
        success++;
    }

    printf("[probe] Successfully allocated %d / 64 layers (%.2f GB total rpcmem)!\n",
           success, (double)success * 95.0 / 1024.0);

    for (int i = 0; i < success; i++) {
        if (ptrs[i]) rpcmem_free(ptrs[i]);
    }
    printf("[probe] Cleaned up and freed all buffers successfully.\n");
    return (success == 64) ? 0 : 1;
}
