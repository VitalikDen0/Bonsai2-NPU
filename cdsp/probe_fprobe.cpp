// probe_fprobe: verify DSP-side file reads match ARM-side bytes.
// Usage: probe_fprobe <npubin-path-on-device>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

extern "C" {
void rpcmem_init(void);
}
#include "bonsai.h"

int main(int argc, char** argv) {
    if (argc != 2) { printf("usage: probe_fprobe <npubin>\n"); return 2; }
    setenv("ADSP_LIBRARY_PATH", "/data/local/tmp;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp", 1);
    struct { int domain; int enable; } um = {3, 1};
    extern int remote_session_control(unsigned, void*, unsigned);
    remote_session_control(2, &um, sizeof(um));
    rpcmem_init();
    remote_handle64 h = 0;
    int rc = bonsai_open("file:///libbonsai_q1_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0&_dom=cdsp", &h);
    if (rc) { printf("OPEN FAIL rc=%d\n", rc); return 1; }
    printf("OPEN OK\n");

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { printf("ARM OPEN FAIL\n"); return 1; }
    unsigned long long offs[] = {0ULL, 12ULL, 1000000ULL, 4708628960ULL - 64, 123456789ULL};
    int fails = 0;
    for (int i = 0; i < 5; i++) {
        unsigned long long off = offs[i];
        unsigned char ddsp[64], darm[64];
        memset(ddsp, 0, 64);
        rc = bonsai_fprobe(h, (int)(off >> 32), (int)(off & 0xffffffff), ddsp, 64);
        if (rc != 0) { printf("FPROBE off=%llu rc=%d\n", off, rc); fails++; continue; }
        if (lseek(fd, (long)off, SEEK_SET) < 0) { printf("ARM SEEK FAIL\n"); fails++; continue; }
        if (read(fd, darm, 64) != 64) { printf("ARM READ FAIL\n"); fails++; continue; }
        if (memcmp(ddsp, darm, 64) != 0) { printf("MISMATCH off=%llu\n", off); fails++; continue; }
        printf("MATCH off=%llu first8=%02x%02x%02x%02x...\n", off, ddsp[0], ddsp[1], ddsp[2], ddsp[3]);
    }
    close(fd);
    // window ladder: find where DSP-side positioning breaks.
    // fd2 walks forward monotonically; arm_pos tracks it.
    {
        unsigned long long wins[] = {0x10000000ULL, 0x20000000ULL, 0x40000000ULL,
                                     0x60000000ULL, 0x80000000ULL, 0xA0000000ULL,
                                     0xC0000000ULL, 0x100000000ULL, 0x1177B0C00ULL};
        int fd2 = open(argv[1], O_RDONLY);
        unsigned long long arm_pos = 0;
        char drop[65536];
        for (int wi = 0; wi < 9 && fd2 >= 0; wi++) {
            unsigned long long win = wins[wi];
            unsigned char ddsp[64], darm[64];
            memset(ddsp, 0, 64);
            rc = bonsai_fmaptest(h, (int)(win >> 32), (int)(win & 0xffffffff), 0, ddsp, 64);
            if (rc != 0) { printf("LADDER win=%llu rc=%d\n", win, rc); continue; }
            int ok = 1;
            while (arm_pos < win) {
                unsigned long long step = win - arm_pos;
                size_t ch = step > sizeof(drop) ? sizeof(drop) : (size_t)step;
                ssize_t n = read(fd2, drop, ch);
                if (n <= 0) { ok = 0; break; }
                arm_pos += (unsigned long long)n;
            }
            if (!ok || read(fd2, darm, 64) != 64) { printf("ARM READ FAIL win=%llu\n", win); fails++; continue; }
            arm_pos += 64;
            if (memcmp(ddsp, darm, 64) != 0) printf("LADDER win=%llu MISMATCH\n", win);
            else printf("LADDER win=%llu MATCH\n", win);
        }
        if (fd2 >= 0) close(fd2);
    }
    // bulk-read throughput below 2GB cap (bread checksums on DSP, no bulk return)
    {
        struct timespec ts;
        unsigned long long wins[] = {0x10000000ULL, 0x30000000ULL};
        int lens[] = {256 << 20, 256 << 20};
        for (int wi = 0; wi < 2; wi++) {
            unsigned long long win = wins[wi];
            int len = lens[wi];
            long long csum = 0;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t0 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
            rc = bonsai_bread(h, (int)(win >> 32), (int)(win & 0xffffffff), len, &csum);
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t1 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
            if (rc != 0) printf("BREAD win=%llu rc=%d\n", win, rc);
            else printf("BREAD win=%llu len=%d ms=%.1f MBps=%.1f csum=%llx\n",
                        win, len, t1 - t0, len / (t1 - t0) * 1000.0 / 1e6,
                        (unsigned long long)csum);
        }
    }
    // bulk-read benchmark below 2GB cap (64MB + 256MB windows)
    {
        struct timespec ts;
        unsigned long long wins[] = {0x10000000ULL, 0x30000000ULL};
        int lens[] = {64 << 20, 64 << 20};
        for (int wi = 0; wi < 2; wi++) {
            unsigned long long win = wins[wi];
            int len = lens[wi];
            long long csum = 0;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t0 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
            rc = bonsai_bread(h, (int)(win >> 32), (int)(win & 0xffffffff), len, &csum);
            clock_gettime(CLOCK_MONOTONIC, &ts);
            double t1 = ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
            if (rc != 0) printf("BREAD win=%llu rc=%d\n", win, rc);
            else printf("BREAD win=%llu len=%d ms=%.1f MBps=%.1f csum=%llx\n",
                        win, len, t1 - t0, len / (t1 - t0) * 1000.0 / 1e6,
                        (unsigned long long)csum);
        }
    }
    bonsai_close(h);
    printf(fails == 0 ? "FPROBE PASS\n" : "FPROBE FAIL\n");
    return fails == 0 ? 0 : 1;
}
