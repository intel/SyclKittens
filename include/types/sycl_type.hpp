#pragma once
#include <sycl/sycl.hpp>
#include <cstdlib>

namespace intel {
#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N>
using vector_t = T __attribute__((ext_vector_type(N)));
template <class T, class F> T vec_as(const F &x) {
  return sycl::bit_cast<T>(x);
}
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
template <class T, class F> T vec_as(const F &x) {
  return sycl::bit_cast<T>(x);
}
#endif

struct SPIRV_MMAOperands {
  static constexpr int SPIRV_MatrixASigned = 0x1;
  static constexpr int SPIRV_MatrixBSigned = 0x2;
  static constexpr int SPIRV_MatrixAInt8 = 0x10;
  static constexpr int SPIRV_MatrixBInt8 = 0x20;
  static constexpr int SPIRV_MatrixAFp16 = 0x400;
  static constexpr int SPIRV_MatrixBFp16 = 0x800;
  static constexpr int SPIRV_MatrixABf16 = 0x1000;
  static constexpr int SPIRV_MatrixBBf16 = 0x2000;
  static constexpr int SPIRV_MatrixCBf16 = 0xC;
  static constexpr int SPIRV_MatrixATf32 = 0x100;
  static constexpr int SPIRV_MatrixBTf32 = 0x200;
};

using uint = unsigned int;
using ushort = unsigned short;
using ulong = unsigned long;
using uchar = unsigned char;

using short8 = vector_t<short, 8>;
using ushort8 = vector_t<ushort, 8>;
using float8 = vector_t<float, 8>;
using uint8 = vector_t<uint, 8>;
using uint16 = vector_t<uint, 16>;
using int8 = vector_t<int, 8>;
using int4 = vector_t<int, 4>;
using coord_t = vector_t<int, 2>;
template <class T> short8 as_short8(const T &x) { return vec_as<short8>(x); }
template <class T> int8 as_int8(const T &x) { return vec_as<int8>(x); }
template <class T> uint8 as_uint8(const T &x) { return vec_as<uint8>(x); }

} // namespace intel

#define INVALID_CONTROL_PATH(x)                                                \
  assert(0 && x);                                                              \
  printf(x)

#ifndef __SYCL_DEVICE_ONLY__
[[noreturn]] inline void device_only_intrinsic_on_host() noexcept {
  std::abort();
}
#endif

inline void fence_sw() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
  asm volatile("fence_sw\n");
#endif
}

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#define SYCL_DEVICE_BUILTIN(x)                                                 \
  inline x {                                                                   \
    device_only_intrinsic_on_host();                                           \
  }
#endif
#ifndef __SYCL_DEVICE_ONLY__
inline void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer) {
  device_only_intrinsic_on_host();
}
inline intel::float8 __spirv_SubgroupMatrixMultiplyAccumulateINTEL(
    int32_t, intel::short8, intel::int8, intel::float8, int32_t) {
  device_only_intrinsic_on_host();
}
inline void __spirv_Subgroup2DBlockPrefetchINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate) {
  device_only_intrinsic_on_host();
}

inline void __spirv_Subgroup2DBlockStoreINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    void *src_pointer, const void *dst_base_pointer, int memory_width,
    int memory_height, int memory_pitch, intel::coord_t coordinate) {
  device_only_intrinsic_on_host();
}

inline void __spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer) {
  device_only_intrinsic_on_host();
}

SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::short8 data));

inline void __spirv_Subgroup2DBlockLoadTransposeINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer) {
  device_only_intrinsic_on_host();
}

#else
// SPIRV
SYCL_EXTERNAL __attribute__((convergent)) void __spirv_Subgroup2DBlockLoadINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer);

SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_Subgroup2DBlockLoadTransformINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer);

SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_Subgroup2DBlockStoreINTEL(int ElementSize, int BlockWidth,
                                  int BlockHeight, int BlockCount,
                                  void *src_pointer,
                                  const void *dst_base_pointer,
                                  int memory_width, int memory_height,
                                  int memory_pitch, intel::coord_t coordinate);
SYCL_EXTERNAL __attribute__((convergent)) intel::float8
    __spirv_SubgroupMatrixMultiplyAccumulateINTEL(int32_t, intel::short8,
                                                  intel::int8, intel::float8,
                                                  int32_t);
SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_Subgroup2DBlockPrefetchINTEL(int ElementSize, int BlockWidth,
                                     int BlockHeight, int BlockCount,
                                     const void *src_base_pointer,
                                     int memory_width, int memory_height,
                                     int memory_pitch,
                                     intel::coord_t coordinate);
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_write_flat_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, intel::short8 data));

SYCL_EXTERNAL __attribute__((convergent)) void
__spirv_Subgroup2DBlockLoadTransposeINTEL(
    int ElementSize, int BlockWidth, int BlockHeight, int BlockCount,
    const void *src_base_pointer, int memory_width, int memory_height,
    int memory_pitch, intel::coord_t coordinate, void *dst_pointer);

#endif

// ---------------------------------------------------------------------------
// L2-TARGETED 2D-block prefetch (LSC cache-control) for long-latency XeLink /
// remote-peer tiles.  In LSC naming "L3" == the large shared L2 on PVC and
// "L1" == the small per-Xe-core cache.  kL1UC_L3C means: do NOT cache in L1
// (so a long-latency remote tile does not thrash the tiny L1), cache only in
// the big shared L2.  A later short-range DEFAULT prefetch/load then pulls the
// tile L2->L1->reg cheaply.  Maps to the same IGC builtin sycl-tla/CUTLASS-Xe
// uses (they default to kL1C_L3C; for XeLink we want L1 bypassed).
// ---------------------------------------------------------------------------
enum class LscCacheControl {
  kDefault   = 0,
  kL1UC_L3UC = 1,
  kL1UC_L3C  = 2,   // L1 uncached, L3(shared-L2) cached  <-- XeLink prefetch
  kL1C_L3UC  = 3,
  kL1C_L3C   = 4,
  kL1S_L3UC  = 5,
  kL1S_L3C   = 6,
  kL1IAR_L3C = 7,
};
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_block_read_prefetch_u16_m32k16v2(
    intptr_t baseoffset, int width_minus_one, int height_minus_one,
    int pitch_minus_one, intel::coord_t coord, LscCacheControl cache_control));

// ---------------------------------------------------------------------------
// 2D-block ADDRESS-PAYLOAD builtins (hoisted-descriptor block IO).
//
// ADDITIVE / OPT-IN: these are *declarations only*. Existing code paths (which
// use the flat __spirv_Subgroup2DBlock*INTEL intrinsics) are untouched and do
// not reference these symbols. They enable a "build the block2d descriptor
// once, mutate only X/Y per load" pattern (see ops/warp/memory/tile/gl_desc.dp.hpp),
// mirroring the payload model used by sycl-tla / CUTLASS-Xe. On host these are
// trivial no-op stubs (device-only intrinsics; never executed on host).
// ---------------------------------------------------------------------------
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL extern "C" int *__builtin_IB_subgroup_createBlock2DAddressPayload(
    long base, int width_minus_one, int height_minus_one, int pitch_minus_one,
    int blockX, int blockY, int blockWidth, int blockHeight, int numBlocks);
SYCL_EXTERNAL extern "C" int *
__builtin_IB_subgroup_copyBlock2DAddressPayload(int *AP);
#else
inline int *__builtin_IB_subgroup_createBlock2DAddressPayload(
    long, int, int, int, int, int, int, int, int) {
  return nullptr;
}
inline int *__builtin_IB_subgroup_copyBlock2DAddressPayload(int *AP) {
  return AP;
}
#endif
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockX(
    int *addrPayload, int blockX));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadBlockY(
    int *addrPayload, int blockY));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockX(
    int *addrPayload, int blockX));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_addBlock2DAddressPayloadBlockY(
    int *addrPayload, int blockY));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadBase(
    int *addrPayload, long base));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadWidth(
    int *addrPayload, int width_minus_one));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadHeigth(
    int *addrPayload, int height_minus_one));
SYCL_DEVICE_BUILTIN(void __builtin_IB_subgroup_setBlock2DAddressPayloadPitch(
    int *addrPayload, int pitch_minus_one));