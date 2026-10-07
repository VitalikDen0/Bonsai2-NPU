#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

extern int remote_session_control(unsigned, void*, unsigned);
extern int bonsai_open(const char*, unsigned long long*);
extern int bonsai_close(unsigned long long);
extern int bonsai_bread(unsigned long long h, int off_hi, int off_lo, int len, long long* csum);

int main(void) {
    struct { int domain; int enable; } um = {3, 1};
    remote_session_control(2, &um, sizeof(um));
    unsigned long long h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    printf("bonsai_open rc=%d, h=0x%llx\n", rc, h);
    if (rc != 0) return 1;

    int test_sizes_mb[] = { 1, 10, 50, 100, 150, 200, 250, 500, 0 };
    for (int i = 0; test_sizes_mb[i] > 0; i++) {
        int sz_mb = test_sizes_mb[i];
        long long ptr = 0;
        int r = bonsai_bread(h, 9999, 0, sz_mb * 1024 * 1024, &ptr);
        printf("DSP malloc(%3d MB): rc=%d, ptr=0x%llx (%s)\n",
               sz_mb, r, ptr, r == 0 ? "SUCCESS" : "FAILED");
    }

    bonsai_close(h);
    return 0;
}
