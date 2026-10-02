#pragma once
// SPV_INTEL_split_barrier intrinsic declarations.
// Device-side:  SYCL_EXTERNAL declaration → device compiler maps to SPIRV
// Host-side:    empty inline stub → linker satisfied, never called at runtime
#ifdef KITTENS_INTEL
#ifdef __SYCL_DEVICE_ONLY__
SYCL_EXTERNAL __attribute__((convergent))
void __spirv_ControlBarrierArriveINTEL(unsigned int exec_scope,
                                        unsigned int mem_scope,
                                        unsigned int semantics);
SYCL_EXTERNAL __attribute__((convergent))
void __spirv_ControlBarrierWaitINTEL(unsigned int exec_scope,
                                      unsigned int mem_scope,
                                      unsigned int semantics);
#else
// Host-side stubs — never executed, exist only to satisfy the linker.
inline void __spirv_ControlBarrierArriveINTEL(unsigned int, unsigned int, unsigned int) {}
inline void __spirv_ControlBarrierWaitINTEL(unsigned int, unsigned int, unsigned int) {}
#endif
#endif // KITTENS_INTEL
