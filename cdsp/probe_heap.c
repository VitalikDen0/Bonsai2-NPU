#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);

int main(void) {
    setbuf(stdout, NULL);
    rpcmem_init();

    printf("[probe_heap] Allocating 3.6 GB rpcmem...\n");
    void* p_rpc0 = rpcmem_alloc(25, 0, (size_t)1615 * 1024 * 1024);
    void* p_rpc1 = rpcmem_alloc(25, 0, (size_t)1615 * 1024 * 1024);
    void* p_rpclm = rpcmem_alloc(25, 0, (size_t)322 * 1024 * 1024);
    void* p_rpcring = rpcmem_alloc(25, 0, (size_t)192 * 1024 * 1024);
    printf("  rpc0=%p rpc1=%p rpclm=%p rpcring=%p\n", p_rpc0, p_rpc1, p_rpclm, p_rpcring);

    size_t heap_sz = (size_t)3260 * 1024 * 1024;
    printf("[probe_heap] Allocating %.2f GB heap (malloc)...\n", (double)heap_sz / (1024.0*1024.0*1024.0));
    char* p_heap = (char*)malloc(heap_sz);
    printf("  malloc returned %p\n", p_heap);

    if (p_heap) {
        printf("[probe_heap] Touching pages to verify physical allocation...\n");
        // touch every 1MB
        for (size_t off = 0; off < heap_sz; off += 1024 * 1024) {
            p_heap[off] = (char)(off & 0xFF);
        }
        printf("  Touch OK! Physical RAM confirmed!\n");
        free(p_heap);
    }

    if (p_rpcring) rpcmem_free(p_rpcring);
    if (p_rpclm) rpcmem_free(p_rpclm);
    if (p_rpc1) rpcmem_free(p_rpc1);
    if (p_rpc0) rpcmem_free(p_rpc0);
    printf("[probe_heap] Done.\n");
    return 0;
}
