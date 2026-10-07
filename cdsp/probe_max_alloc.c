#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);

int main(void) {
    rpcmem_init();
    size_t sizes[] = {
        (size_t)1500 * 1024 * 1024,
        (size_t)1800 * 1024 * 1024,
        (size_t)2000 * 1024 * 1024,
        (size_t)2100 * 1024 * 1024,
        (size_t)2200 * 1024 * 1024
    };
    int n = sizeof(sizes) / sizeof(sizes[0]);
    for (int i = 0; i < n; i++) {
        void* p = rpcmem_alloc(25, 0, sizes[i]);
        if (p) {
            printf("[probe] %zu MB alloc SUCCESS (fd=%d)\n", sizes[i] >> 20, rpcmem_to_fd(p));
            rpcmem_free(p);
        } else {
            printf("[probe] %zu MB alloc FAILED\n", sizes[i] >> 20);
        }
    }
    return 0;
}
