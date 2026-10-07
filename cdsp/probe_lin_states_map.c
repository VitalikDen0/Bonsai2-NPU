#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

extern void rpcmem_init(void);
extern void* rpcmem_alloc(int heapid, unsigned flags, int size);
extern void rpcmem_free(void* po);
extern int rpcmem_to_fd(void* po);
extern int fastrpc_mmap(int domain, int fd, void* addr, int offset, size_t length, int flags);
extern int fastrpc_munmap(int domain, int fd, void* addr, size_t length);
extern int remote_session_control(unsigned, void*, unsigned);
extern int bonsai_open(const char*, unsigned long long*);
extern int bonsai_close(unsigned long long);

int main(void) {
    setbuf(stdout, NULL);
    rpcmem_init();
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    unsigned long long h = 0;
    int rco = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    printf("[probe] bonsai_open rc=%d, handle=0x%llx\n", rco, h);
    if (rco != 0) return 1;

    // Simulate 24 static layers (Group 0 1155 MB + Group 1 1155 MB) + LM head (322 MB) + Ring Arena (776 MB)
    size_t g0_sz = 1155 * 1024 * 1024;
    size_t g1_sz = 1155 * 1024 * 1024;
    size_t lm_sz = 322 * 1024 * 1024;
    size_t ra_sz = 776 * 1024 * 1024;

    void* g0 = rpcmem_alloc(25, 0, (int)g0_sz);
    void* g1 = rpcmem_alloc(25, 0, (int)g1_sz);
    void* lm = rpcmem_alloc(25, 0, (int)lm_sz);
    void* ra = rpcmem_alloc(25, 0, (int)ra_sz);

    int m_g0 = fastrpc_mmap(3, rpcmem_to_fd(g0), g0, 0, g0_sz, 0);
    int m_g1 = fastrpc_mmap(3, rpcmem_to_fd(g1), g1, 0, g1_sz, 0);
    int m_lm = fastrpc_mmap(3, rpcmem_to_fd(lm), lm, 0, lm_sz, 0);
    int m_ra = fastrpc_mmap(3, rpcmem_to_fd(ra), ra, 0, ra_sz, 0);
    printf("[probe] Baseline 3408 MB mapped: g0=%d, g1=%d, lm=%d, ra=%d\n", m_g0, m_g1, m_lm, m_ra);

    // Now test mapping Linear Attention states & aux:
    size_t conv_sz = 6 * 1024 * 1024;    // 5.8 MB
    size_t rec_sz  = 151 * 1024 * 1024;  // 150.9 MB
    size_t aux_sz  = 102 * 1024 * 1024;  // 102.3 MB

    void* p_conv = rpcmem_alloc(25, 0, (int)conv_sz);
    void* p_rec  = rpcmem_alloc(25, 0, (int)rec_sz);
    void* p_aux  = rpcmem_alloc(25, 0, (int)aux_sz);

    int m_conv = p_conv ? fastrpc_mmap(3, rpcmem_to_fd(p_conv), p_conv, 0, conv_sz, 0) : -1;
    int m_rec  = p_rec  ? fastrpc_mmap(3, rpcmem_to_fd(p_rec),  p_rec,  0, rec_sz,  0) : -1;
    int m_aux  = p_aux  ? fastrpc_mmap(3, rpcmem_to_fd(p_aux),  p_aux,  0, aux_sz,  0) : -1;

    printf("[probe] Linear Attention SMMU map results:\n");
    printf("  ssm_conv (6 MB):   alloc=%s, mmap rc=%d\n", p_conv ? "OK" : "FAIL", m_conv);
    printf("  ssm_rec  (151 MB): alloc=%s, mmap rc=%d\n", p_rec  ? "OK" : "FAIL", m_rec);
    printf("  lin_aux  (102 MB): alloc=%s, mmap rc=%d\n", p_aux  ? "OK" : "FAIL", m_aux);

    size_t total_mapped = g0_sz + g1_sz + lm_sz + ra_sz +
                          (m_conv == 0 ? conv_sz : 0) +
                          (m_rec == 0 ? rec_sz : 0) +
                          (m_aux == 0 ? aux_sz : 0);
    printf("[probe] TOTAL SIMULTANEOUS SMMU MAPPED: %zu MB (%.2f GB)!\n", total_mapped >> 20, (double)total_mapped / (1024.0*1024.0*1024.0));

    // Cleanup
    if (m_aux == 0) fastrpc_munmap(3, rpcmem_to_fd(p_aux), p_aux, aux_sz);
    if (m_rec == 0) fastrpc_munmap(3, rpcmem_to_fd(p_rec), p_rec, rec_sz);
    if (m_conv == 0) fastrpc_munmap(3, rpcmem_to_fd(p_conv), p_conv, conv_sz);
    if (m_ra == 0) fastrpc_munmap(3, rpcmem_to_fd(ra), ra, ra_sz);
    if (m_lm == 0) fastrpc_munmap(3, rpcmem_to_fd(lm), lm, lm_sz);
    if (m_g1 == 0) fastrpc_munmap(3, rpcmem_to_fd(g1), g1, g1_sz);
    if (m_g0 == 0) fastrpc_munmap(3, rpcmem_to_fd(g0), g0, g0_sz);

    bonsai_close(h);
    return (m_conv == 0 && m_rec == 0 && m_aux == 0) ? 0 : 1;
}
