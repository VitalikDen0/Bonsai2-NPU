#ifndef _BONSAI_STUB_H
#define _BONSAI_STUB_H
#include "bonsai.h"
#include <string.h>
#ifndef _WIN32
#include "HAP_farf.h"
#include <inttypes.h>
#endif //_WIN32 for HAP_farf
#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H

#include <stdlib.h>
#include <stdint.h>

typedef struct _heap _heap;
struct _heap {
   _heap* pPrev;
   const char* loc;
   uint64_t buf;
};

typedef struct _allocator {
   _heap* pheap;
   uint8_t* stack;
   uint8_t* stackEnd;
   int nSize;
} _allocator;

_ATTRIBUTE_UNUSED
static __inline int _heap_alloc(_heap** ppa, const char* loc, size_t size, void** ppbuf) {
   _heap* pn = 0;
   pn = MALLOC(size + sizeof(_heap) - sizeof(uint64_t));
   if(pn != 0) {
      pn->pPrev = *ppa;
      pn->loc = loc;
      *ppa = pn;
      *ppbuf = (void*)&(pn->buf);
      return 0;
   } else {
      return -1;
   }
}
#define _ALIGN_SIZE(x, y) (((x) + (y-1)) & ~(y-1))

_ATTRIBUTE_UNUSED
static __inline int _allocator_alloc(_allocator* me,
                                    const char* loc,
                                    size_t size,
                                    unsigned int al,
                                    void** ppbuf) {
   if(size < 0) {
      return -1;
   } else if (size == 0) {
      *ppbuf = 0;
      return 0;
   }
   if((_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size) < (uintptr_t)me->stack + (size_t)me->nSize) {
      *ppbuf = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al);
      me->stackEnd = (uint8_t*)_ALIGN_SIZE((uintptr_t)me->stackEnd, al) + size;
      return 0;
   } else {
      return _heap_alloc(&me->pheap, loc, size, ppbuf);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_deinit(_allocator* me) {
   _heap* pa = me->pheap;
   while(pa != 0) {
      _heap* pn = pa;
      const char* loc = pn->loc;
      (void)loc;
      pa = pn->pPrev;
      FREE(pn);
   }
}

_ATTRIBUTE_UNUSED
static __inline void _allocator_init(_allocator* me, uint8_t* stack, int stackSize) {
   me->stack =  stack;
   me->stackEnd =  stack + stackSize;
   me->nSize = stackSize;
   me->pheap = 0;
}


#endif // _ALLOCATOR_H

#ifndef SLIM_H
#define SLIM_H

#include <stdint.h>

//a C data structure for the idl types that can be used to implement
//static and dynamic language bindings fairly efficiently.
//
//the goal is to have a minimal ROM and RAM footprint and without
//doing too many allocations.  A good way to package these things seemed
//like the module boundary, so all the idls within  one module can share
//all the type references.


#define PARAMETER_IN       0x0
#define PARAMETER_OUT      0x1
#define PARAMETER_INOUT    0x2
#define PARAMETER_ROUT     0x3
#define PARAMETER_INROUT   0x4

//the types that we get from idl
#define TYPE_OBJECT             0x0
#define TYPE_INTERFACE          0x1
#define TYPE_PRIMITIVE          0x2
#define TYPE_ENUM               0x3
#define TYPE_STRING             0x4
#define TYPE_WSTRING            0x5
#define TYPE_STRUCTURE          0x6
#define TYPE_UNION              0x7
#define TYPE_ARRAY              0x8
#define TYPE_SEQUENCE           0x9

//these require the pack/unpack to recurse
//so it's a hint to those languages that can optimize in cases where
//recursion isn't necessary.
#define TYPE_COMPLEX_STRUCTURE  (0x10 | TYPE_STRUCTURE)
#define TYPE_COMPLEX_UNION      (0x10 | TYPE_UNION)
#define TYPE_COMPLEX_ARRAY      (0x10 | TYPE_ARRAY)
#define TYPE_COMPLEX_SEQUENCE   (0x10 | TYPE_SEQUENCE)


typedef struct Type Type;

#define INHERIT_TYPE\
   int32_t nativeSize;                /*in the simple case its the same as wire size and alignment*/\
   union {\
      struct {\
         const uintptr_t         p1;\
         const uintptr_t         p2;\
      } _cast;\
      struct {\
         uint32_t  iid;\
         uint32_t  bNotNil;\
      } object;\
      struct {\
         const Type  *arrayType;\
         int32_t      nItems;\
      } array;\
      struct {\
         const Type *seqType;\
         int32_t      nMaxLen;\
      } seqSimple; \
      struct {\
         uint32_t bFloating;\
         uint32_t bSigned;\
      } prim; \
      const SequenceType* seqComplex;\
      const UnionType  *unionType;\
      const StructType *structType;\
      int32_t         stringMaxLen;\
      uint8_t        bInterfaceNotNil;\
   } param;\
   uint8_t    type;\
   uint8_t    nativeAlignment\

typedef struct UnionType UnionType;
typedef struct StructType StructType;
typedef struct SequenceType SequenceType;
struct Type {
   INHERIT_TYPE;
};

struct SequenceType {
   const Type *         seqType;
   uint32_t               nMaxLen;
   uint32_t               inSize;
   uint32_t               routSizePrimIn;
   uint32_t               routSizePrimROut;
};

//byte offset from the start of the case values for
//this unions case value array.  it MUST be aligned
//at the alignment requrements for the descriptor
//
//if negative it means that the unions cases are
//simple enumerators, so the value read from the descriptor
//can be used directly to find the correct case
typedef union CaseValuePtr CaseValuePtr;
union CaseValuePtr {
   const uint8_t*   value8s;
   const uint16_t*  value16s;
   const uint32_t*  value32s;
   const uint64_t*  value64s;
};

//these are only used in complex cases
//so I pulled them out of the type definition as references to make
//the type smaller
struct UnionType {
   const Type           *descriptor;
   uint32_t               nCases;
   const CaseValuePtr   caseValues;
   const Type * const   *cases;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
   uint8_t                inCaseAlignment;
   uint8_t                routCaseAlignmentPrimIn;
   uint8_t                routCaseAlignmentPrimROut;
   uint8_t                nativeCaseAlignment;
   uint8_t              bDefaultCase;
};

struct StructType {
   uint32_t               nMembers;
   const Type * const   *members;
   int32_t               inSize;
   int32_t               routSizePrimIn;
   int32_t               routSizePrimROut;
   uint8_t                inAlignment;
   uint8_t                routAlignmentPrimIn;
   uint8_t                routAlignmentPrimROut;
};

typedef struct Parameter Parameter;
struct Parameter {
   INHERIT_TYPE;
   uint8_t    mode;
   uint8_t  bNotNil;
};

#define SLIM_IFPTR32(is32,is64) (sizeof(uintptr_t) == 4 ? (is32) : (is64))
#define SLIM_SCALARS_IS_DYNAMIC(u) (((u) & 0x00ffffff) == 0x00ffffff)

typedef struct Method Method;
struct Method {
   uint32_t                    uScalars;            //no method index
   int32_t                     primInSize;
   int32_t                     primROutSize;
   int                         maxArgs;
   int                         numParams;
   const Parameter * const     *params;
   uint8_t                       primInAlignment;
   uint8_t                       primROutAlignment;
};

typedef struct Interface Interface;

struct Interface {
   int                            nMethods;
   const Method  * const          *methodArray;
   int                            nIIds;
   const uint32_t                   *iids;
   const uint16_t*                  methodStringArray;
   const uint16_t*                  methodStrings;
   const char*                    strings;
};


#endif //SLIM_H


#ifndef _BONSAI_SLIM_H
#define _BONSAI_SLIM_H
#include <stdint.h>

#ifndef __QAIC_SLIM
#define __QAIC_SLIM(ff) ff
#endif
#ifndef __QAIC_SLIM_EXPORT
#define __QAIC_SLIM_EXPORT
#endif

static const Type types[3];
static const Type types[3] = {{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4},{0x1,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x1},{0x2,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x2}};
static const Parameter parameters[10] = {{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)0x0,0}}, 4,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x4,0x8),{{(const uintptr_t)0xdeadc0de,(const uintptr_t)0}}, 0,SLIM_IFPTR32(0x4,0x8),0,0},{0x4,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x4,0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[0]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[1]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[2]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),0,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[0]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{SLIM_IFPTR32(0x8,0x10),{{(const uintptr_t)&(types[1]),(const uintptr_t)0x0}}, 9,SLIM_IFPTR32(0x4,0x8),3,0},{0x8,{{(const uintptr_t)0,(const uintptr_t)1}}, 2,0x8,3,0}};
static const Parameter* const parameterArrays[50] = {(&(parameters[3])),(&(parameters[4])),(&(parameters[5])),(&(parameters[6])),(&(parameters[5])),(&(parameters[6])),(&(parameters[5])),(&(parameters[6])),(&(parameters[5])),(&(parameters[6])),(&(parameters[7])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[4])),(&(parameters[5])),(&(parameters[6])),(&(parameters[5])),(&(parameters[6])),(&(parameters[7])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[4])),(&(parameters[5])),(&(parameters[6])),(&(parameters[7])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[9])),(&(parameters[3])),(&(parameters[3])),(&(parameters[3])),(&(parameters[8])),(&(parameters[4])),(&(parameters[4])),(&(parameters[5])),(&(parameters[0])),(&(parameters[1])),(&(parameters[2]))};
static const Method methods[13] = {{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x1),0x4,0x0,2,2,(&(parameterArrays[47])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x0,0x0,0x1,0x0),0x0,0x0,1,1,(&(parameterArrays[49])),0x1,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x4,0x1,0x0,0x0),0x1c,0x0,12,7,(&(parameterArrays[29])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x0,0x0,0x0),0x2c,0x0,11,11,(&(parameterArrays[11])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0xc,0x0,5,3,(&(parameterArrays[41])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0x10,0x0,6,4,(&(parameterArrays[40])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x1,0x1,0x0,0x0),0xc,0x8,4,4,(&(parameterArrays[36])),0x4,0x8},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x4,0x0,2,1,(&(parameterArrays[1])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x6,0x1,0x0,0x0),0x18,0x0,13,6,(&(parameterArrays[23])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0x2,0x0,0x0,0x0),0x8,0x0,3,2,(&(parameterArrays[0])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x4,0x0,0x0,0x0),0xc,0x0,6,3,(&(parameterArrays[44])),0x4,0x0},{REMOTE_SCALARS_MAKEX(0,0,0x6,0x1,0x0,0x0),0x1c,0x0,14,7,(&(parameterArrays[22])),0x4,0x1},{REMOTE_SCALARS_MAKEX(0,0,0xa,0x1,0x0,0x0),0x2c,0x0,22,11,(&(parameterArrays[0])),0x4,0x1}};
static const Method* const methodArrays[13] = {&(methods[0]),&(methods[1]),&(methods[2]),&(methods[3]),&(methods[4]),&(methods[5]),&(methods[6]),&(methods[7]),&(methods[8]),&(methods[9]),&(methods[10]),&(methods[11]),&(methods[12])};
static const char strings[386] = "register_lin_states\0lin_layer_fused\0lin_attn_fused\0set_signs_dim\0down_scales\0gate_scales\0out_scales\0in_scales\0layer_idx\0down_bits\0gate_bits\0mlp_fused\0set_signs\0scales_lo\0scales_hi\0out_bits\0ssm_conv\0fmaptest\0gemv_q1r\0in_bits\0lin_aux\0ssm_rec\0bits_lo\0bits_hi\0out_dim\0gemv_q1\0at_off\0win_lo\0win_hi\0off_lo\0off_hi\0fprobe\0in_dim\0bread\0close\0csum\0data\0y_lo\0y_hi\0x_lo\0x_hi\0prow\0open\0len\0uri\0y\0h\0";
static const uint16_t methodStrings[75] = {20,110,118,216,100,180,89,130,77,120,65,381,207,256,314,363,358,353,248,240,170,160,348,343,36,110,118,216,100,180,89,381,264,256,314,363,118,125,70,381,140,118,130,77,120,65,381,321,300,293,373,333,198,286,279,272,338,0,189,232,224,307,300,293,338,51,61,154,368,377,383,150,154,327,383};
static const uint16_t methodStringsArrays[13] = {68,73,32,12,61,52,47,71,40,65,57,24,0};
__QAIC_SLIM_EXPORT const Interface __QAIC_SLIM(bonsai_slim) = {13,&(methodArrays[0]),0,0,&(methodStringsArrays [0]),methodStrings,strings};
#endif //_BONSAI_SLIM_H


#ifdef __cplusplus
extern "C" {
#endif
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_open)(const char* uri, remote_handle64* h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_open)(uri, h);
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_close)(remote_handle64 h) __QAIC_STUB_ATTRIBUTE {
   return __QAIC_REMOTE(remote_handle64_close)(h);
}
static __inline int _stub_method(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], uint32_t _in1[1], uint32_t _in2[1], char* _in3[1], uint32_t _in3Len[1], char* _in4[1], uint32_t _in4Len[1], char* _in5[1], uint32_t _in5Len[1], char* _rout6[1], uint32_t _rout6Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[5] = {0};
   uint32_t _primIn[7]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 3;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in3[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in3Len[0]));
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in4[0];
   _praIn[1].buf.nLen = (1 * (size_t)(_in4Len[0]));
   _COPY(_primIn, 20, _in5Len, 0, 4);
   _praIn[2].buf.pv = (void*) _in5[0];
   _praIn[2].buf.nLen = (2 * (size_t)(_in5Len[0]));
   _COPY(_primIn, 24, _rout6Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout6[0];
   _praROut[0].buf.nLen = (4 * (size_t)(_rout6Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 4, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 4, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_gemv_q1)(remote_handle64 _handle, int out_dim, int in_dim, int prow, const float* x, int xLen, const unsigned char* bits, int bitsLen, const short* scales, int scalesLen, float* y, int yLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 2;
   return _stub_method(_handle, _mid, (uint32_t*)&out_dim, (uint32_t*)&in_dim, (uint32_t*)&prow, (char**)&x, (uint32_t*)&xLen, (char**)&bits, (uint32_t*)&bitsLen, (char**)&scales, (uint32_t*)&scalesLen, (char**)&y, (uint32_t*)&yLen);
}
static __inline int _stub_method_1(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], uint32_t _in1[1], uint32_t _in2[1], uint32_t _in3[1], uint32_t _in4[1], uint32_t _in5[1], uint32_t _in6[1], uint32_t _in7[1], uint32_t _in8[1], uint32_t _in9[1], uint32_t _in10[1]) {
   remote_arg _pra[1] = {0};
   uint32_t _primIn[11]= {0};
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _in3, 0, 4);
   _COPY(_primIn, 16, _in4, 0, 4);
   _COPY(_primIn, 20, _in5, 0, 4);
   _COPY(_primIn, 24, _in6, 0, 4);
   _COPY(_primIn, 28, _in7, 0, 4);
   _COPY(_primIn, 32, _in8, 0, 4);
   _COPY(_primIn, 36, _in9, 0, 4);
   _COPY(_primIn, 40, _in10, 0, 4);
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 0, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_gemv_q1r)(remote_handle64 _handle, int out_dim, int in_dim, int prow, int x_hi, int x_lo, int bits_hi, int bits_lo, int scales_hi, int scales_lo, int y_hi, int y_lo) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 3;
   return _stub_method_1(_handle, _mid, (uint32_t*)&out_dim, (uint32_t*)&in_dim, (uint32_t*)&prow, (uint32_t*)&x_hi, (uint32_t*)&x_lo, (uint32_t*)&bits_hi, (uint32_t*)&bits_lo, (uint32_t*)&scales_hi, (uint32_t*)&scales_lo, (uint32_t*)&y_hi, (uint32_t*)&y_lo);
}
static __inline int _stub_method_2(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], uint32_t _in1[1], char* _rout2[1], uint32_t _rout2Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[2] = {0};
   uint32_t _primIn[3]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _rout2Len, 0, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout2[0];
   _praROut[0].buf.nLen = (1 * (size_t)(_rout2Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_fprobe)(remote_handle64 _handle, int off_hi, int off_lo, unsigned char* data, int dataLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 4;
   return _stub_method_2(_handle, _mid, (uint32_t*)&off_hi, (uint32_t*)&off_lo, (char**)&data, (uint32_t*)&dataLen);
}
static __inline int _stub_method_3(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], uint32_t _in1[1], uint32_t _in2[1], char* _rout3[1], uint32_t _rout3Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[2] = {0};
   uint32_t _primIn[4]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _COPY(_primIn, 12, _rout3Len, 0, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout3[0];
   _praROut[0].buf.nLen = (1 * (size_t)(_rout3Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_fmaptest)(remote_handle64 _handle, int win_hi, int win_lo, int at_off, unsigned char* data, int dataLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 5;
   return _stub_method_3(_handle, _mid, (uint32_t*)&win_hi, (uint32_t*)&win_lo, (uint32_t*)&at_off, (char**)&data, (uint32_t*)&dataLen);
}
static __inline int _stub_method_4(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], uint32_t _in1[1], uint32_t _in2[1], uint64_t _rout3[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[2] = {0};
   uint32_t _primIn[3]= {0};
   uint64_t _primROut[1]= {0};
   int _nErr = 0;
   _numIn[0] = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _pra[(_numIn[0] + 1)].buf.pv = (void*)_primROut;
   _pra[(_numIn[0] + 1)].buf.nLen = sizeof(_primROut);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1, 0, 4);
   _COPY(_primIn, 8, _in2, 0, 4);
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _pra));
   _COPY(_rout3, 0, _primROut, 0, 8);
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 1, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_bread)(remote_handle64 _handle, int off_hi, int off_lo, int len, int64* csum) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 6;
   return _stub_method_4(_handle, _mid, (uint32_t*)&off_hi, (uint32_t*)&off_lo, (uint32_t*)&len, (uint64_t*)csum);
}
static __inline int _stub_method_5(remote_handle64 _handle, uint32_t _mid, char* _in0[1], uint32_t _in0Len[1]) {
   remote_arg _pra[2] = {0};
   uint32_t _primIn[1]= {0};
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in0[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in0Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_set_signs)(remote_handle64 _handle, const float* signs, int signsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 7;
   return _stub_method_5(_handle, _mid, (char**)&signs, (uint32_t*)&signsLen);
}
static __inline int _stub_method_6(remote_handle64 _handle, uint32_t _mid, char* _in0[1], uint32_t _in0Len[1], char* _in1[1], uint32_t _in1Len[1], char* _in2[1], uint32_t _in2Len[1], char* _in3[1], uint32_t _in3Len[1], char* _in4[1], uint32_t _in4Len[1], char* _rout5[1], uint32_t _rout5Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[7] = {0};
   uint32_t _primIn[6]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 5;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in0[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in0Len[0]));
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in1[0];
   _praIn[1].buf.nLen = (1 * (size_t)(_in1Len[0]));
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn[2].buf.pv = (void*) _in2[0];
   _praIn[2].buf.nLen = (2 * (size_t)(_in2Len[0]));
   _COPY(_primIn, 12, _in3Len, 0, 4);
   _praIn[3].buf.pv = (void*) _in3[0];
   _praIn[3].buf.nLen = (1 * (size_t)(_in3Len[0]));
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn[4].buf.pv = (void*) _in4[0];
   _praIn[4].buf.nLen = (2 * (size_t)(_in4Len[0]));
   _COPY(_primIn, 20, _rout5Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout5[0];
   _praROut[0].buf.nLen = (4 * (size_t)(_rout5Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 6, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 6, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_mlp_fused)(remote_handle64 _handle, const float* x, int xLen, const unsigned char* gate_bits, int gate_bitsLen, const short* gate_scales, int gate_scalesLen, const unsigned char* down_bits, int down_bitsLen, const short* down_scales, int down_scalesLen, float* y, int yLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 8;
   return _stub_method_6(_handle, _mid, (char**)&x, (uint32_t*)&xLen, (char**)&gate_bits, (uint32_t*)&gate_bitsLen, (char**)&gate_scales, (uint32_t*)&gate_scalesLen, (char**)&down_bits, (uint32_t*)&down_bitsLen, (char**)&down_scales, (uint32_t*)&down_scalesLen, (char**)&y, (uint32_t*)&yLen);
}
static __inline int _stub_method_7(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], char* _in1[1], uint32_t _in1Len[1]) {
   remote_arg _pra[2] = {0};
   uint32_t _primIn[2]= {0};
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in1Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 2, 0, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_set_signs_dim)(remote_handle64 _handle, int dim, const float* signs, int signsLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 9;
   return _stub_method_7(_handle, _mid, (uint32_t*)&dim, (char**)&signs, (uint32_t*)&signsLen);
}
static __inline int _stub_method_8(remote_handle64 _handle, uint32_t _mid, char* _in0[1], uint32_t _in0Len[1], char* _in1[1], uint32_t _in1Len[1], char* _in2[1], uint32_t _in2Len[1]) {
   remote_arg _pra[4] = {0};
   uint32_t _primIn[3]= {0};
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in0[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in0Len[0]));
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in1[0];
   _praIn[1].buf.nLen = (4 * (size_t)(_in1Len[0]));
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn[2].buf.pv = (void*) _in2[0];
   _praIn[2].buf.nLen = (1 * (size_t)(_in2Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 4, 0, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 4, 0, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_register_lin_states)(remote_handle64 _handle, const float* ssm_conv, int ssm_convLen, const float* ssm_rec, int ssm_recLen, const unsigned char* lin_aux, int lin_auxLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 10;
   return _stub_method_8(_handle, _mid, (char**)&ssm_conv, (uint32_t*)&ssm_convLen, (char**)&ssm_rec, (uint32_t*)&ssm_recLen, (char**)&lin_aux, (uint32_t*)&lin_auxLen);
}
static __inline int _stub_method_9(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], char* _in1[1], uint32_t _in1Len[1], char* _in2[1], uint32_t _in2Len[1], char* _in3[1], uint32_t _in3Len[1], char* _in4[1], uint32_t _in4Len[1], char* _in5[1], uint32_t _in5Len[1], char* _rout6[1], uint32_t _rout6Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[7] = {0};
   uint32_t _primIn[7]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 5;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in1Len[0]));
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in2[0];
   _praIn[1].buf.nLen = (1 * (size_t)(_in2Len[0]));
   _COPY(_primIn, 12, _in3Len, 0, 4);
   _praIn[2].buf.pv = (void*) _in3[0];
   _praIn[2].buf.nLen = (2 * (size_t)(_in3Len[0]));
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn[3].buf.pv = (void*) _in4[0];
   _praIn[3].buf.nLen = (1 * (size_t)(_in4Len[0]));
   _COPY(_primIn, 20, _in5Len, 0, 4);
   _praIn[4].buf.pv = (void*) _in5[0];
   _praIn[4].buf.nLen = (2 * (size_t)(_in5Len[0]));
   _COPY(_primIn, 24, _rout6Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout6[0];
   _praROut[0].buf.nLen = (4 * (size_t)(_rout6Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 6, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 6, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_lin_attn_fused)(remote_handle64 _handle, int layer_idx, const float* x, int xLen, const unsigned char* in_bits, int in_bitsLen, const short* in_scales, int in_scalesLen, const unsigned char* out_bits, int out_bitsLen, const short* out_scales, int out_scalesLen, float* y, int yLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 11;
   return _stub_method_9(_handle, _mid, (uint32_t*)&layer_idx, (char**)&x, (uint32_t*)&xLen, (char**)&in_bits, (uint32_t*)&in_bitsLen, (char**)&in_scales, (uint32_t*)&in_scalesLen, (char**)&out_bits, (uint32_t*)&out_bitsLen, (char**)&out_scales, (uint32_t*)&out_scalesLen, (char**)&y, (uint32_t*)&yLen);
}
static __inline int _stub_method_10(remote_handle64 _handle, uint32_t _mid, uint32_t _in0[1], char* _in1[1], uint32_t _in1Len[1], char* _in2[1], uint32_t _in2Len[1], char* _in3[1], uint32_t _in3Len[1], char* _in4[1], uint32_t _in4Len[1], char* _in5[1], uint32_t _in5Len[1], char* _in6[1], uint32_t _in6Len[1], char* _in7[1], uint32_t _in7Len[1], char* _in8[1], uint32_t _in8Len[1], char* _in9[1], uint32_t _in9Len[1], char* _rout10[1], uint32_t _rout10Len[1]) {
   int _numIn[1] = {0};
   remote_arg _pra[11] = {0};
   uint32_t _primIn[11]= {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _numIn[0] = 9;
   _pra[0].buf.pv = (void*)_primIn;
   _pra[0].buf.nLen = sizeof(_primIn);
   _COPY(_primIn, 0, _in0, 0, 4);
   _COPY(_primIn, 4, _in1Len, 0, 4);
   _praIn = (_pra + 1);
   _praIn[0].buf.pv = (void*) _in1[0];
   _praIn[0].buf.nLen = (4 * (size_t)(_in1Len[0]));
   _COPY(_primIn, 8, _in2Len, 0, 4);
   _praIn[1].buf.pv = (void*) _in2[0];
   _praIn[1].buf.nLen = (1 * (size_t)(_in2Len[0]));
   _COPY(_primIn, 12, _in3Len, 0, 4);
   _praIn[2].buf.pv = (void*) _in3[0];
   _praIn[2].buf.nLen = (2 * (size_t)(_in3Len[0]));
   _COPY(_primIn, 16, _in4Len, 0, 4);
   _praIn[3].buf.pv = (void*) _in4[0];
   _praIn[3].buf.nLen = (1 * (size_t)(_in4Len[0]));
   _COPY(_primIn, 20, _in5Len, 0, 4);
   _praIn[4].buf.pv = (void*) _in5[0];
   _praIn[4].buf.nLen = (2 * (size_t)(_in5Len[0]));
   _COPY(_primIn, 24, _in6Len, 0, 4);
   _praIn[5].buf.pv = (void*) _in6[0];
   _praIn[5].buf.nLen = (1 * (size_t)(_in6Len[0]));
   _COPY(_primIn, 28, _in7Len, 0, 4);
   _praIn[6].buf.pv = (void*) _in7[0];
   _praIn[6].buf.nLen = (2 * (size_t)(_in7Len[0]));
   _COPY(_primIn, 32, _in8Len, 0, 4);
   _praIn[7].buf.pv = (void*) _in8[0];
   _praIn[7].buf.nLen = (1 * (size_t)(_in8Len[0]));
   _COPY(_primIn, 36, _in9Len, 0, 4);
   _praIn[8].buf.pv = (void*) _in9[0];
   _praIn[8].buf.nLen = (2 * (size_t)(_in9Len[0]));
   _COPY(_primIn, 40, _rout10Len, 0, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _praROut[0].buf.pv = _rout10[0];
   _praROut[0].buf.nLen = (4 * (size_t)(_rout10Len[0]));
   _TRY_FARF(_nErr, __QAIC_REMOTE(remote_handle64_invoke)(_handle, REMOTE_SCALARS_MAKEX(0, _mid, 10, 1, 0, 0), _pra));
   _CATCH_FARF(_nErr) {
      _QAIC_FARF(RUNTIME_ERROR, "ERROR 0x%x: handle=0x%"PRIx64", scalar=0x%x, method ID=%d: %s failed\n", _nErr , _handle, REMOTE_SCALARS_MAKEX(0, _mid, 10, 1, 0, 0), _mid, __func__);
   }
   return _nErr;
}
__QAIC_STUB_EXPORT int __QAIC_STUB(bonsai_lin_layer_fused)(remote_handle64 _handle, int layer_idx, const float* x, int xLen, const unsigned char* in_bits, int in_bitsLen, const short* in_scales, int in_scalesLen, const unsigned char* out_bits, int out_bitsLen, const short* out_scales, int out_scalesLen, const unsigned char* gate_bits, int gate_bitsLen, const short* gate_scales, int gate_scalesLen, const unsigned char* down_bits, int down_bitsLen, const short* down_scales, int down_scalesLen, float* y, int yLen) __QAIC_STUB_ATTRIBUTE {
   uint32_t _mid = 12;
   return _stub_method_10(_handle, _mid, (uint32_t*)&layer_idx, (char**)&x, (uint32_t*)&xLen, (char**)&in_bits, (uint32_t*)&in_bitsLen, (char**)&in_scales, (uint32_t*)&in_scalesLen, (char**)&out_bits, (uint32_t*)&out_bitsLen, (char**)&out_scales, (uint32_t*)&out_scalesLen, (char**)&gate_bits, (uint32_t*)&gate_bitsLen, (char**)&gate_scales, (uint32_t*)&gate_scalesLen, (char**)&down_bits, (uint32_t*)&down_bitsLen, (char**)&down_scales, (uint32_t*)&down_scalesLen, (char**)&y, (uint32_t*)&yLen);
}
#ifdef __cplusplus
}
#endif
#endif //_BONSAI_STUB_H
