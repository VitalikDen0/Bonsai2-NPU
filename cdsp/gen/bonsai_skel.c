#ifndef _BONSAI_SKEL_H
#define _BONSAI_SKEL_H
#include "bonsai.h"

#include <string.h>
#ifndef _WIN32
#include "HAP_farf.h"
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
extern int adsp_mmap_fd_getinfo(int, uint32_t *);
#ifdef __cplusplus
extern "C" {
#endif
_ATTRIBUTE_VISIBILITY uint32_t bonsai_skel_handle_invoke_qaic_version = 10051;
_ATTRIBUTE_VISIBILITY char bonsai_skel_handle_invoke_uri[77+1]="file:///libbonsai_skel.so?bonsai_skel_handle_invoke&_modver=1.0&_idlver=1.0.0";
static __inline int _skel_method(int (*_pfn)(remote_handle64, int, const float*, int, const unsigned char*, int, const short*, int, const unsigned char*, int, const short*, int, const unsigned char*, int, const short*, int, const unsigned char*, int, const short*, int, float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   char* _in1[1] = {0};
   uint32_t _in1Len[1] = {0};
   char* _in2[1] = {0};
   uint32_t _in2Len[1] = {0};
   char* _in3[1] = {0};
   uint32_t _in3Len[1] = {0};
   char* _in4[1] = {0};
   uint32_t _in4Len[1] = {0};
   char* _in5[1] = {0};
   uint32_t _in5Len[1] = {0};
   char* _in6[1] = {0};
   uint32_t _in6Len[1] = {0};
   char* _in7[1] = {0};
   uint32_t _in7Len[1] = {0};
   char* _in8[1] = {0};
   uint32_t _in8Len[1] = {0};
   char* _in9[1] = {0};
   uint32_t _in9Len[1] = {0};
   char* _rout10[1] = {0};
   uint32_t _rout10Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==10);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((10 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 44);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[0].buf.pv;
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[1].buf.nLen / 1)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[1].buf.pv;
   _COPY(_in3Len, 0, _primIn, 12, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[2].buf.nLen / 2)) >= (size_t)(_in3Len[0]));
   _in3[0] = _praIn[2].buf.pv;
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[3].buf.nLen / 1)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[3].buf.pv;
   _COPY(_in5Len, 0, _primIn, 20, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[4].buf.nLen / 2)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[4].buf.pv;
   _COPY(_in6Len, 0, _primIn, 24, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[5].buf.nLen / 1)) >= (size_t)(_in6Len[0]));
   _in6[0] = _praIn[5].buf.pv;
   _COPY(_in7Len, 0, _primIn, 28, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[6].buf.nLen / 2)) >= (size_t)(_in7Len[0]));
   _in7[0] = _praIn[6].buf.pv;
   _COPY(_in8Len, 0, _primIn, 32, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[7].buf.nLen / 1)) >= (size_t)(_in8Len[0]));
   _in8[0] = _praIn[7].buf.pv;
   _COPY(_in9Len, 0, _primIn, 36, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[8].buf.nLen / 2)) >= (size_t)(_in9Len[0]));
   _in9[0] = _praIn[8].buf.pv;
   _COPY(_rout10Len, 0, _primIn, 40, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 4)) >= (size_t)(_rout10Len[0]));
   _rout10[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (const float*)*_in1, (int)*_in1Len, (const unsigned char*)*_in2, (int)*_in2Len, (const short*)*_in3, (int)*_in3Len, (const unsigned char*)*_in4, (int)*_in4Len, (const short*)*_in5, (int)*_in5Len, (const unsigned char*)*_in6, (int)*_in6Len, (const short*)*_in7, (int)*_in7Len, (const unsigned char*)*_in8, (int)*_in8Len, (const short*)*_in9, (int)*_in9Len, (float*)*_rout10, (int)*_rout10Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_1(int (*_pfn)(remote_handle64, int, const float*, int, const unsigned char*, int, const short*, int, const unsigned char*, int, const short*, int, float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   char* _in1[1] = {0};
   uint32_t _in1Len[1] = {0};
   char* _in2[1] = {0};
   uint32_t _in2Len[1] = {0};
   char* _in3[1] = {0};
   uint32_t _in3Len[1] = {0};
   char* _in4[1] = {0};
   uint32_t _in4Len[1] = {0};
   char* _in5[1] = {0};
   uint32_t _in5Len[1] = {0};
   char* _rout6[1] = {0};
   uint32_t _rout6Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==6);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((6 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 28);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[0].buf.pv;
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[1].buf.nLen / 1)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[1].buf.pv;
   _COPY(_in3Len, 0, _primIn, 12, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[2].buf.nLen / 2)) >= (size_t)(_in3Len[0]));
   _in3[0] = _praIn[2].buf.pv;
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[3].buf.nLen / 1)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[3].buf.pv;
   _COPY(_in5Len, 0, _primIn, 20, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[4].buf.nLen / 2)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[4].buf.pv;
   _COPY(_rout6Len, 0, _primIn, 24, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 4)) >= (size_t)(_rout6Len[0]));
   _rout6[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (const float*)*_in1, (int)*_in1Len, (const unsigned char*)*_in2, (int)*_in2Len, (const short*)*_in3, (int)*_in3Len, (const unsigned char*)*_in4, (int)*_in4Len, (const short*)*_in5, (int)*_in5Len, (float*)*_rout6, (int)*_rout6Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_2(int (*_pfn)(remote_handle64, const float*, int, const float*, int, const unsigned char*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   char* _in0[1] = {0};
   uint32_t _in0Len[1] = {0};
   char* _in1[1] = {0};
   uint32_t _in1Len[1] = {0};
   char* _in2[1] = {0};
   uint32_t _in2Len[1] = {0};
   uint32_t* _primIn= 0;
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==4);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((4 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[1].buf.nLen / 4)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[1].buf.pv;
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[2].buf.nLen / 1)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[2].buf.pv;
   _TRY(_nErr, _pfn(_h, (const float*)*_in0, (int)*_in0Len, (const float*)*_in1, (int)*_in1Len, (const unsigned char*)*_in2, (int)*_in2Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_3(int (*_pfn)(remote_handle64, int, const float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   char* _in1[1] = {0};
   uint32_t _in1Len[1] = {0};
   uint32_t* _primIn= 0;
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==2);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 8);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (const float*)*_in1, (int)*_in1Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_4(int (*_pfn)(remote_handle64, const float*, int, const unsigned char*, int, const short*, int, const unsigned char*, int, const short*, int, float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   char* _in0[1] = {0};
   uint32_t _in0Len[1] = {0};
   char* _in1[1] = {0};
   uint32_t _in1Len[1] = {0};
   char* _in2[1] = {0};
   uint32_t _in2Len[1] = {0};
   char* _in3[1] = {0};
   uint32_t _in3Len[1] = {0};
   char* _in4[1] = {0};
   uint32_t _in4Len[1] = {0};
   char* _rout5[1] = {0};
   uint32_t _rout5Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==6);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((6 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 24);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _COPY(_in1Len, 0, _primIn, 4, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[1].buf.nLen / 1)) >= (size_t)(_in1Len[0]));
   _in1[0] = _praIn[1].buf.pv;
   _COPY(_in2Len, 0, _primIn, 8, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[2].buf.nLen / 2)) >= (size_t)(_in2Len[0]));
   _in2[0] = _praIn[2].buf.pv;
   _COPY(_in3Len, 0, _primIn, 12, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[3].buf.nLen / 1)) >= (size_t)(_in3Len[0]));
   _in3[0] = _praIn[3].buf.pv;
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[4].buf.nLen / 2)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[4].buf.pv;
   _COPY(_rout5Len, 0, _primIn, 20, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 4)) >= (size_t)(_rout5Len[0]));
   _rout5[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (const float*)*_in0, (int)*_in0Len, (const unsigned char*)*_in1, (int)*_in1Len, (const short*)*_in2, (int)*_in2Len, (const unsigned char*)*_in3, (int)*_in3Len, (const short*)*_in4, (int)*_in4Len, (float*)*_rout5, (int)*_rout5Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_5(int (*_pfn)(remote_handle64, const float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   char* _in0[1] = {0};
   uint32_t _in0Len[1] = {0};
   uint32_t* _primIn= 0;
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==2);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (const float*)*_in0, (int)*_in0Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_6(int (*_pfn)(remote_handle64, int, int, int, int64*), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   uint32_t _in1[1] = {0};
   uint32_t _in2[1] = {0};
   uint64_t _rout3[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   uint64_t* _primROut= 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _QAIC_ASSERT(_nErr, _pra[(_numIn[0] + 1)].buf.nLen >= 8);
   _primROut = _pra[(_numIn[0] + 1)].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _TRY(_nErr, _pfn(_h, (int)*_in0, (int)*_in1, (int)*_in2, (int64*)_rout3));
   _COPY(_primROut, 0, _rout3, 0, 8);
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_7(int (*_pfn)(remote_handle64, int, int, int, unsigned char*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   uint32_t _in1[1] = {0};
   uint32_t _in2[1] = {0};
   char* _rout3[1] = {0};
   uint32_t _rout3Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 16);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_rout3Len, 0, _primIn, 12, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout3Len[0]));
   _rout3[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (int)*_in1, (int)*_in2, (unsigned char*)*_rout3, (int)*_rout3Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_8(int (*_pfn)(remote_handle64, int, int, unsigned char*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   uint32_t _in1[1] = {0};
   char* _rout2[1] = {0};
   uint32_t _rout2Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((1 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 12);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_rout2Len, 0, _primIn, 8, 4);
   _praIn = (_pra + 1);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 1)) >= (size_t)(_rout2Len[0]));
   _rout2[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (int)*_in1, (unsigned char*)*_rout2, (int)*_rout2Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_9(int (*_pfn)(remote_handle64, int, int, int, int, int, int, int, int, int, int, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   uint32_t _in1[1] = {0};
   uint32_t _in2[1] = {0};
   uint32_t _in3[1] = {0};
   uint32_t _in4[1] = {0};
   uint32_t _in5[1] = {0};
   uint32_t _in6[1] = {0};
   uint32_t _in7[1] = {0};
   uint32_t _in8[1] = {0};
   uint32_t _in9[1] = {0};
   uint32_t _in10[1] = {0};
   uint32_t* _primIn= 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((1 + 0) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 44);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3, 0, _primIn, 12, 4);
   _COPY(_in4, 0, _primIn, 16, 4);
   _COPY(_in5, 0, _primIn, 20, 4);
   _COPY(_in6, 0, _primIn, 24, 4);
   _COPY(_in7, 0, _primIn, 28, 4);
   _COPY(_in8, 0, _primIn, 32, 4);
   _COPY(_in9, 0, _primIn, 36, 4);
   _COPY(_in10, 0, _primIn, 40, 4);
   _TRY(_nErr, _pfn(_h, (int)*_in0, (int)*_in1, (int)*_in2, (int)*_in3, (int)*_in4, (int)*_in5, (int)*_in6, (int)*_in7, (int)*_in8, (int)*_in9, (int)*_in10));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_10(int (*_pfn)(remote_handle64, int, int, int, const float*, int, const unsigned char*, int, const short*, int, float*, int), remote_handle64 _h, uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   uint32_t _in0[1] = {0};
   uint32_t _in1[1] = {0};
   uint32_t _in2[1] = {0};
   char* _in3[1] = {0};
   uint32_t _in3Len[1] = {0};
   char* _in4[1] = {0};
   uint32_t _in4Len[1] = {0};
   char* _in5[1] = {0};
   uint32_t _in5Len[1] = {0};
   char* _rout6[1] = {0};
   uint32_t _rout6Len[1] = {0};
   uint32_t* _primIn= 0;
   int _numIn[1] = {0};
   remote_arg* _praIn = 0;
   remote_arg* _praROut = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==4);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((4 + 1) + (((0 + 0) + 0) + 0))) <= _praEnd);
   _numIn[0] = (REMOTE_SCALARS_INBUFS(_sc) - 1);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 28);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0, 0, _primIn, 0, 4);
   _COPY(_in1, 0, _primIn, 4, 4);
   _COPY(_in2, 0, _primIn, 8, 4);
   _COPY(_in3Len, 0, _primIn, 12, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 4)) >= (size_t)(_in3Len[0]));
   _in3[0] = _praIn[0].buf.pv;
   _COPY(_in4Len, 0, _primIn, 16, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[1].buf.nLen / 1)) >= (size_t)(_in4Len[0]));
   _in4[0] = _praIn[1].buf.pv;
   _COPY(_in5Len, 0, _primIn, 20, 4);
   _QAIC_ASSERT(_nErr, ((_praIn[2].buf.nLen / 2)) >= (size_t)(_in5Len[0]));
   _in5[0] = _praIn[2].buf.pv;
   _COPY(_rout6Len, 0, _primIn, 24, 4);
   _praROut = (_praIn + _numIn[0] + 0);
   _QAIC_ASSERT(_nErr, ((_praROut[0].buf.nLen / 4)) >= (size_t)(_rout6Len[0]));
   _rout6[0] = _praROut[0].buf.pv;
   _TRY(_nErr, _pfn(_h, (int)*_in0, (int)*_in1, (int)*_in2, (const float*)*_in3, (int)*_in3Len, (const unsigned char*)*_in4, (int)*_in4Len, (const short*)*_in5, (int)*_in5Len, (float*)*_rout6, (int)*_rout6Len));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _skel_method_11(int (*_pfn)(remote_handle64), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   remote_handle64 _in0[1] = {0};
   remote_arg* _praRHandleIn = _pra + REMOTE_SCALARS_INBUFS(_sc) +  REMOTE_SCALARS_OUTBUFS(_sc);
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==1);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, (_pra + ((0 + 0) + (((1 + 0) + 0) + 0))) <= _praEnd);
   _COPY(_in0, 0, &(_praRHandleIn[0].h64), 0, sizeof(remote_handle64));
   _TRY(_nErr, _pfn((remote_handle64)*_in0));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
static __inline int _compare_versions(char* stub_ver, char* skel_ver, int* result) {
   unsigned long int major_stub = 0, minor_stub = 0, patch_stub = 0;
   unsigned long int major_skel = 0, minor_skel = 0, patch_skel = 0;
   char *saveptr1 = NULL;
   char *token1 = NULL;
   char *saveptr2 = NULL;
   char *token2 = NULL;
   int i=0;
   for (i=0, token1 = strtok_r(stub_ver, ".", &saveptr1); i<3 && token1 != NULL; i++, token1 = strtok_r(NULL, ".", &saveptr1))
   {
      unsigned long int tn = strtoul(token1, NULL,10);
      if( tn > 999)
      {
         *result=-1;
         return 0;
      }
      else
      {
         if(i==0) major_stub=tn;
         if(i==1) minor_stub=tn;
         if(i==2) patch_stub=tn;
      }
   }
   for (i=0, token2 = strtok_r(skel_ver, ".", &saveptr2); i<3 && token2 != NULL; i++, token2 = strtok_r(NULL, ".", &saveptr2))
   {
      unsigned long int tn = strtoul(token2, NULL,10);
      if( tn > 999)
      {
         *result=-1;
         return 0;
      }
      else
      {
         if(i==0) major_skel=tn;
         if(i==1) minor_skel=tn;
         if(i==2) patch_skel=tn;
      }
   }
   if(major_stub<major_skel)
   {
      *result=1;
      return 0;
   }
   else if(major_stub==major_skel)
   {
      if( minor_stub < minor_skel )
      {
         *result=1;
         return 0;
      }
      else if((minor_stub == minor_skel) && (patch_skel>=patch_stub))
      {
         *result=1;
         return 0;
      }
   }
   *result=-1;
   return 0;
}
static __inline int _stub_skel_version_check(char*_in0, int* resVal) {
   int _nErr = 0;
   char* p = strstr(_in0, "_idlver=");
   if(!p)
   {
      *resVal = -1;
      return 0;
   }
   p+=8;
   int i=0,len=0, comVer=0,num_delimit=0, updtInxStub=0, updtInxSkel=0;
   for(i=0;i<strlen(p);i++)
   {
      if(num_delimit>2)
      {
         *resVal = -1;
         return 0;
      }
      if ((p[i]>='0' && p[i]<='9') || (p[i]=='.'))
      {
         len++;
         if(p[i]=='.')
         {
            num_delimit++;
         }
      }
      else if(p[i]=='&')
      {
         break;
      }
      else
      {
         *resVal = -1;
         return 0;
      }
   }
   char* stubVer=(char*)MALLOC(len+1);
   _QAIC_ASSERT(_nErr, stubVer!=NULL);
   for(i=0;i<strlen(p);i++)
   {
      if((p[i]>='0' && p[i]<='9') || (p[i]=='.'))
      {
         stubVer[updtInxStub]=p[i];
         updtInxStub++;
      }
      else if(p[i]=='&')
      {
         break;
      }
   }
   stubVer[len]='\0';
   char* skelVer=(char*)MALLOC(strlen(IDL_VERSION)+1);
   _QAIC_ASSERT(_nErr, skelVer!=NULL);
   for(i=0;i< strlen(IDL_VERSION);i++)
   {
      skelVer[updtInxSkel]=IDL_VERSION[i];
      updtInxSkel++;
   }
   skelVer[strlen(IDL_VERSION)]='\0';
   _TRY(_nErr, _compare_versions(stubVer, skelVer, &comVer));
   *resVal = 0;
   if (comVer==-1)
   {
      *resVal = -1;
   }
   FREE(stubVer);
   FREE(skelVer);
   _QAIC_CATCH(_nErr) {}
   return 0;
}
static __inline int _skel_method_12(int (*_pfn)(const char*, remote_handle64*), uint32_t _sc, remote_arg* _pra) {
   remote_arg* _praEnd = 0;
   char* _in0[1] = {0};
   uint32_t _in0Len[1] = {0};
   remote_handle64 _rout1[1] = {0};
   uint32_t* _primIn= 0;
   remote_arg* _praRHandleROut = _pra + REMOTE_SCALARS_INBUFS(_sc) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) ;
   remote_arg* _praIn = 0;
   int _nErr = 0;
   _praEnd = ((_pra + REMOTE_SCALARS_INBUFS(_sc)) + REMOTE_SCALARS_OUTBUFS(_sc) + REMOTE_SCALARS_INHANDLES(_sc) + REMOTE_SCALARS_OUTHANDLES(_sc));
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INBUFS(_sc)==2);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTBUFS(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_INHANDLES(_sc)==0);
   _QAIC_ASSERT(_nErr, REMOTE_SCALARS_OUTHANDLES(_sc)==1);
   _QAIC_ASSERT(_nErr, (_pra + ((2 + 0) + (((0 + 1) + 0) + 0))) <= _praEnd);
   _QAIC_ASSERT(_nErr, _pra[0].buf.nLen >= 4);
   _primIn = _pra[0].buf.pv;
   _COPY(_in0Len, 0, _primIn, 0, 4);
   _praIn = (_pra + 1);
   _QAIC_ASSERT(_nErr, ((_praIn[0].buf.nLen / 1)) >= (size_t)(_in0Len[0]));
   _in0[0] = _praIn[0].buf.pv;
   _QAIC_ASSERT(_nErr, (_in0Len[0] > 0) && (_in0[0][(_in0Len[0] - 1)] == 0));
   int resVal;
   _TRY(_nErr, _stub_skel_version_check(*_in0, &resVal));
   if(resVal==-1)
   {
      return AEE_ESTUBSKELVERMISMATCH;
   }
   _TRY(_nErr, _pfn((const char*)*_in0, (remote_handle64*)_rout1));
   _COPY(&(_praRHandleROut[0].h64), 0, _rout1, 0, sizeof(remote_handle64));
   _QAIC_CATCH(_nErr) {}
   return _nErr;
}
__QAIC_SKEL_EXPORT int __QAIC_SKEL(bonsai_skel_handle_invoke)(remote_handle64 _h, uint32_t _sc, remote_arg* _pra) __QAIC_SKEL_ATTRIBUTE {
   switch(REMOTE_SCALARS_METHOD(_sc)){
      case 0:
      return _skel_method_12(__QAIC_IMPL(bonsai_open), _sc, _pra);
      case 1:
      return _skel_method_11(__QAIC_IMPL(bonsai_close), _sc, _pra);
      case 2:
      return _skel_method_10(__QAIC_IMPL(bonsai_gemv_q1), _h, _sc, _pra);
      case 3:
      return _skel_method_9(__QAIC_IMPL(bonsai_gemv_q1r), _h, _sc, _pra);
      case 4:
      return _skel_method_8(__QAIC_IMPL(bonsai_fprobe), _h, _sc, _pra);
      case 5:
      return _skel_method_7(__QAIC_IMPL(bonsai_fmaptest), _h, _sc, _pra);
      case 6:
      return _skel_method_6(__QAIC_IMPL(bonsai_bread), _h, _sc, _pra);
      case 7:
      return _skel_method_5(__QAIC_IMPL(bonsai_set_signs), _h, _sc, _pra);
      case 8:
      return _skel_method_4(__QAIC_IMPL(bonsai_mlp_fused), _h, _sc, _pra);
      case 9:
      return _skel_method_3(__QAIC_IMPL(bonsai_set_signs_dim), _h, _sc, _pra);
      case 10:
      return _skel_method_2(__QAIC_IMPL(bonsai_register_lin_states), _h, _sc, _pra);
      case 11:
      return _skel_method_1(__QAIC_IMPL(bonsai_lin_attn_fused), _h, _sc, _pra);
      case 12:
      return _skel_method(__QAIC_IMPL(bonsai_lin_layer_fused), _h, _sc, _pra);
   }
   return AEE_EUNSUPPORTED;
}
#ifdef __cplusplus
}
#endif
#endif //_BONSAI_SKEL_H
