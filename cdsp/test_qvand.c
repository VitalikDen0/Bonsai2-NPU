#include <stdio.h>
#include <stdint.h>
#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
static const uint32_t mask[32] __attribute__((aligned(128))) = {
    1u<<0, 1u<<1, 1u<<2, 1u<<3, 1u<<4, 1u<<5, 1u<<6, 1u<<7,
    1u<<8, 1u<<9, 1u<<10, 1u<<11, 1u<<12, 1u<<13, 1u<<14, 1u<<15,
    1u<<16, 1u<<17, 1u<<18, 1u<<19, 1u<<20, 1u<<21, 1u<<22, 1u<<23,
    1u<<24, 1u<<25, 1u<<26, 1u<<27, 1u<<28, 1u<<29, 1u<<30, 1u<<31
};
int main() {
    HVX_Vector vM = *(const HVX_Vector*)mask;
    uint32_t w = 0xA5A5A5A5u;
    HVX_VectorPred Q = Q6_Q_vand_VR(vM, (int)w);
    HVX_Vector vOut = Q6_V_vand_QR(Q, -1);
    uint32_t res[32] __attribute__((aligned(128)));
    *(HVX_Vector*)res = vOut;
    for (int i = 0; i < 16; i++) printf("lane %d (bit=%d): 0x%08x\n", i, (w>>i)&1, res[i]);
    return 0;
}
