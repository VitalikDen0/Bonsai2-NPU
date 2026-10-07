// Minimal local stand-in for HAP_farf.h (DSP runtime header, not shipped in
// HexagonSDK5.x.Core). The generated bonsai_skel.c only includes it but calls
// no farf symbols; all real DSP runtime symbols resolve on-device at load.
#ifndef _HAP_FARF_H
#define _HAP_FARF_H
#define HAP_debug_v2 0
#define FARF(level, ...) (void)0
#define RUNTIME_ERROR 0
#endif
