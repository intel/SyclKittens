/**
 * @file
 * @brief Basic operations on generic types.
 */

#pragma once
// #define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <limits>
#include "base_types.dp.hpp"
#include <sycl/ext/intel/math.hpp>
#include "types/sycl_type.hpp"

#include <cmath>

#if defined(__SYCL_DEVICE_ONLY__) && defined(__SPIR__)
#define INLINE __attribute__((always_inline))
#else
#define INLINE inline
#endif
using namespace::intel;
#define STR(x) #x
#define XSTR(x) STR(x)
#if defined(__SYCL_DEVICE_ONLY__)
namespace inline_asm {

//==--------------------------- REDUCE_U32_8x16 ----------------------------==//

// clang-format off
    #define REDUCE_U32_8x16(NAME, OP, DT, TT)                                                          \
    INLINE DT __##NAME##_##TT##_8x16(const vector_t<DT, 8> &src) {                                 \
        vector_t<DT, 8> tmp;                                                                         \
        DT res;                                                                                        \
        asm volatile("{\n"                                                                             \
                    STR(OP) " (M1,16) %0(0,0)<1> %1(0,0)<16;8,1> %1(0,8)<16;8,1>\n"                   \
                    STR(OP) " (M1,16) %0(1,0)<1> %1(2,0)<16;8,1> %1(2,8)<16;8,1>\n"                   \
                    STR(OP) " (M1,16) %0(2,0)<1> %1(4,0)<16;8,1> %1(4,8)<16;8,1>\n"                   \
                    STR(OP) " (M1,16) %0(3,0)<1> %1(6,0)<16;8,1> %1(6,8)<16;8,1>\n"                   \
                    "}\n"                                                                             \
                    : "=rw"(tmp) : "rw"(src));                                                        \
        asm volatile("{\n"                                                                             \
                    ".decl aliasDst v_type=G type=" STR(TT) " num_elts=16 align=GRF alias=<%0,0>\n"   \
                    STR(OP) " (M1,16) %1(4,0)<1> %1(0,0)<8;4,1> %1(0,4)<8;4,1>\n"                     \
                    STR(OP) " (M1,16) %1(5,0)<1> %1(2,0)<8;4,1> %1(2,4)<8;4,1>\n"                     \
                    STR(OP) " (M1,16) %1(6,0)<1> %1(4,0)<4;2,1> %1(4,2)<4;2,1>\n"                     \
                    STR(OP) " (M1,16) aliasDst(0,0)<1> %1(6,0)<2;1,0> %1(6,1)<2;1,0>\n"               \
                    "}\n"                                                                             \
                    : "=rw"(res) : "+rw"(tmp));                                                       \
        return res;                                                                                    \
    }

    REDUCE_U32_8x16(max, max, float, f)
    REDUCE_U32_8x16(sum, add, float, f)
    #undef REDUCE_U32_8x16

    //==--------------------------- REDUCE_U32_16x16 ---------------------------==//

    #define REDUCE_U32_16x16(NAME, OP, DT, TT)                                                         \
    INLINE DT __##NAME##_##TT##_16x16(const vector_t<DT, 16> &src) {                               \
        vector_t<DT, 16> tmp;                                                                        \
        DT res;                                                                                        \
        asm volatile("{\n"                                                                             \
                    STR(OP) " (M1,16) %0(0,0)<1> %1(0, 0)<16;8,1> %1(0, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(1,0)<1> %1(2, 0)<16;8,1> %1(2, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(2,0)<1> %1(4, 0)<16;8,1> %1(4, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(3,0)<1> %1(6, 0)<16;8,1> %1(6, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(4,0)<1> %1(8, 0)<16;8,1> %1(8, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(5,0)<1> %1(10,0)<16;8,1> %1(10,8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(6,0)<1> %1(12,0)<16;8,1> %1(12,8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(7,0)<1> %1(14,0)<16;8,1> %1(14,8)<16;8,1>\n"                 \
                    "}\n"                                                                             \
                    : "=rw"(tmp) : "rw"(src));                                                        \
        asm volatile("{\n"                                                                             \
                    ".decl aliasDst v_type=G type=" STR(TT) " num_elts=16 align=GRF alias=<%0,0>\n"   \
                    STR(OP) " (M1,16) %1(8, 0)<1> %1(0, 0)<8;4,1> %1(0, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(9, 0)<1> %1(2, 0)<8;4,1> %1(2, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(10,0)<1> %1(4, 0)<8;4,1> %1(4, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(11,0)<1> %1(6, 0)<8;4,1> %1(6, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(12,0)<1> %1(8, 0)<4;2,1> %1(8, 2)<4;2,1>\n"                  \
                    STR(OP) " (M1,16) %1(13,0)<1> %1(10,0)<4;2,1> %1(10,2)<4;2,1>\n"                  \
                    STR(OP) " (M1,16) aliasDst(0,0)<1> %1(12,0)<2;1,0> %1(12,1)<2;1,0>\n"             \
                    "}\n"                                                                             \
                    : "=rw"(res) : "+rw"(tmp));                                                       \
        return res;                                                                                    \
    }
   //==--------------------------- REDUCE_U32_16x16_2  with 2 8x16---------------------------==//

    #define REDUCE_U32_16x16_2(NAME, OP, DT, TT)                                                         \
    INLINE DT __##NAME##_##TT##_16x16(const vector_t<DT, 8> &src1, const vector_t<DT, 8> &src2) {                               \
        vector_t<DT, 16> tmp;                                                                        \
        DT res;                                                                                        \
        asm volatile("{\n"                                                                             \
                    STR(OP) " (M1,16) %0(0,0)<1> %1(0, 0)<16;8,1> %1(0, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(1,0)<1> %1(2, 0)<16;8,1> %1(2, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(2,0)<1> %1(4, 0)<16;8,1> %1(4, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(3,0)<1> %1(6, 0)<16;8,1> %1(6, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(4,0)<1> %2(0, 0)<16;8,1> %2(0, 8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(5,0)<1> %2(2,0)<16;8,1> %2(2,8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(6,0)<1> %2(4,0)<16;8,1> %2(4,8)<16;8,1>\n"                 \
                    STR(OP) " (M1,16) %0(7,0)<1> %2(6,0)<16;8,1> %2(6,8)<16;8,1>\n"                 \
                    "}\n"                                                                             \
                    : "=rw"(tmp) : "rw"(src1), "rw"(src2));                                                        \
        asm volatile("{\n"                                                                             \
                    ".decl aliasDst v_type=G type=" STR(TT) " num_elts=16 align=GRF alias=<%0,0>\n"   \
                    STR(OP) " (M1,16) %1(8, 0)<1> %1(0, 0)<8;4,1> %1(0, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(9, 0)<1> %1(2, 0)<8;4,1> %1(2, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(10,0)<1> %1(4, 0)<8;4,1> %1(4, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(11,0)<1> %1(6, 0)<8;4,1> %1(6, 4)<8;4,1>\n"                  \
                    STR(OP) " (M1,16) %1(12,0)<1> %1(8, 0)<4;2,1> %1(8, 2)<4;2,1>\n"                  \
                    STR(OP) " (M1,16) %1(13,0)<1> %1(10,0)<4;2,1> %1(10,2)<4;2,1>\n"                  \
                    STR(OP) " (M1,16) aliasDst(0,0)<1> %1(12,0)<2;1,0> %1(12,1)<2;1,0>\n"             \
                    "}\n"                                                                             \
                    : "=rw"(res) : "+rw"(tmp));                                                       \
        return res;                                                                                    \
    }

    REDUCE_U32_16x16_2(max, max, float, f)
    REDUCE_U32_16x16_2(sum, add, float, f)
    #undef REDUCE_U32_16x16
// clang-format on

} // namespace inline_asm
#endif
namespace kittens {

/**
 * @namespace base_ops
 *
 * @brief A namespace for operations on basic data types.
 */
namespace base_ops {

/* ----------  CONST OPS  ---------- */

/**
 * @brief Represents the zero constant operation.
 *
 * This operation returns the zero value of the specified type.
 *
 * @tparam T The data type for which to return the zero value.
 * @return The zero value of type T.
 */
struct zero {
    template<typename T, typename... args> static inline constexpr T op(args... _) { return base_types::constants<T>::zero();      }
};
/**
 * @brief Represents the one constant operation.
 *
 * This operation returns the one value of the specified type.
 *
 * @tparam T The data type for which to return the one value.
 * @return The one value of type T.
 */
struct one {
    template<typename T, typename... args> static inline constexpr T op(args... _) { return base_types::constants<T>::one();       }
};
/**
 * @brief Represents the positive infinity constant operation.
 *
 * This operation returns the positive infinity value of the specified type.
 *
 * @tparam T The data type for which to return the positive infinity value.
 * @return The positive infinity value of type T.
 */
struct pos_infty {
    template<typename T, typename... args> static inline constexpr T op(args... _) { return base_types::constants<T>::pos_infty(); }
};
/**
 * @brief Represents the negative infinity constant operation.
 *
 * This operation returns the negative infinity value of the specified type.
 *
 * @tparam T The data type for which to return the negative infinity value.
 * @return The negative infinity value of type T.
 */
struct neg_infty {
    template<typename T, typename... args> static inline constexpr T op(args... _) { return base_types::constants<T>::neg_infty(); }
};


/* ----------  UNARY OPS  ---------- */

/**
 * @brief Exponential function operation.
 *
 * This operation calculates the exponential of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp {
    template<typename T> static inline T op(const T &x) { return exp(x); }
};
template <>
inline float exp::op<float>(const float &x) {
    return sycl::native::exp(x);
}
template <>
inline sycl::float2 exp::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::native::exp(x.x()), sycl::native::exp(x.y())};
}
template <>
inline bf16 exp::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::exp(x);
}
template <>
inline bf16_2 exp::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::exp(x);
}
template <>
inline sycl::half exp::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::exp(x);
}
template <>
inline half_2 exp::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::exp(x);
}

/**
 * @brief Inverse function operation.
 *
 * This operation calculates the multiplicative inverse (reciprocal) of the input value.
 *
 * @tparam T The data type of the input and output values. Must be a floating-point type.
 * @param[in] x The input value.
 * @return The inverse of the input value (1/x).
 * @throws std::domain_error if x is zero.
 * @note This function utilizes the hardware-optimized `inv` operation from the SYCL Intel extension.
 */
struct inv {
  template <typename T> static inline T op(const T &x) {
    return sycl::ext::intel::math::inv(x);
  }
};
template <> inline float inv::op<float>(const float &x) {
  return sycl::ext::intel::math::inv(x);
}
template <> inline sycl::float2 inv::op<sycl::float2>(const sycl::float2 &x) {
  return sycl::float2{sycl::ext::intel::math::inv(x.x()),
                      sycl::ext::intel::math::inv(x.y())};
}
template <> inline sycl::half inv::op<sycl::half>(const sycl::half &x) {
  return sycl::ext::intel::math::inv(x);
}
template <> inline half_2 inv::op<half_2>(const half_2 &x) {
  return sycl::ext::intel::math::inv(x);
}
/**
 * @brief Exponential function operation, in base 2
 *
 * This operation calculates the exponential of the input value, in base 2.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The exponential of the input value.
 */
struct exp2 {
    template<typename T> static inline T op(const T &x) { return exp2f(x); }
};
template <>
inline float exp2::op<float>(const float &x) {
    return sycl::exp2((float)x);
}
template <>
inline sycl::float2 exp2::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::exp2((float)(x.x())), sycl::exp2((float)(x.y()))};
}
template <>
inline bf16 exp2::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::exp2(x);
}
template <>
inline bf16_2 exp2::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::exp2(x);
}
template <>
inline sycl::half exp2::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::exp2(x);
}
template <>
inline half_2 exp2::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::exp2(x);
}
/**
 * @brief Natural log function operation.
 *
 * This operation calculates the natural logarithm of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The natural logarithm of the input value.
 */
struct log {
    template<typename T> static inline T op(const T &x) { return log(x); }
};
template <>
inline float log::op<float>(const float &x) {
    return sycl::log((float)x);
}
template <>
inline sycl::float2 log::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::log((float)(x.x())), sycl::log((float)(x.y()))};
}
template <>
inline bf16 log::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::log(x);
}
template <>
inline bf16_2 log::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::log(x);
}
template <>
inline sycl::half log::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::log(x);
}
template <>
inline half_2 log::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::log(x);
}
/**
 * @brief Logarithm base 2 operation.
 *
 * This operation calculates the logarithm base 2 of the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The logarithm base 2 of the input value.
 */
struct log2 {
    template<typename T> static inline T op(const T &x) { return log2(x); }
};
template <>
inline float log2::op<float>(const float &x) {
    return sycl::log2((float)x);
}
template <>
inline sycl::float2 log2::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::log2((float)(x.x())), sycl::log2((float)(x.y()))};
}
template <>
inline bf16 log2::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::log2(x);
}
template <>
inline bf16_2 log2::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::log2(x);
}
template <>
inline sycl::half log2::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::log2(x);
}
template <>
inline half_2 log2::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::log2(x);
}
/**
 * @brief Absolute value operation.
 *
 * This operation calculates the absolute value of the input.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The absolute value of the input.
 */
struct abs {
    template<typename T> static inline T op(const T &x) { return abs(x); }
};
template <>
inline float abs::op<float>(const float &x) {
    return sycl::fabs(x);
}
template <>
inline sycl::float2 abs::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::fabs(x.x()), sycl::fabs(x.y())};
}
template <>
inline bf16 abs::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::fabs(x);
}
template <>
inline bf16_2 abs::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::fabs(x);
}
template <>
inline sycl::half abs::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::habs(x);
}
template <>
inline half_2 abs::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::habs2(x);
}
/**
 * @brief Rectified Linear Unit (ReLU) operation.
 *
 * This operation applies the ReLU function to the input, which is the
 * maximum of zero and the input value.
 *
 * @tparam T The data type of the input and output values.
 * @param x[in] The input value.
 * @return The result of ReLU function applied to the input.
 */
struct relu {
    template <typename T>
    static inline T op(const T &x) {
        return dpct::max(x, base_types::constants<T>::zero());
    }
};
template <>
inline float relu::op<float>(const float &x) {
    return sycl::max(x, 0.f);
}
template <>
inline sycl::float2 relu::op<sycl::float2>(const sycl::float2 &x) {
    return sycl::float2{sycl::max(x.x(), 0.f), sycl::max(x.y(), 0.f)};
}
template <>
inline bf16 relu::op<bf16>(const bf16 &x) {
    return sycl::ext::oneapi::experimental::fmax(
               x, base_types::constants<bf16>::zero());
}
template <>
inline bf16_2 relu::op<bf16_2>(const bf16_2 &x) {
    return sycl::ext::oneapi::experimental::fmax(
               x, base_types::constants<bf16_2>::zero());
}
template <>
inline sycl::half relu::op<sycl::half>(const sycl::half &x) {
    return sycl::ext::intel::math::hmax(x, base_types::constants<half>::zero());
}
template <>
inline half_2 relu::op<half_2>(const half_2 &x) {
    return sycl::ext::intel::math::hmax2(x,
                                         base_types::constants<half_2>::zero());
}
/**
 * @brief Copy operation.
 *
 * This operation returns the input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The input value.
 * @return The same value as the input.
 */
struct copy { // for non-compile-time setters.
    template<typename T> static inline T op(const T &a) { return a; }
};


/* ----------  BINARY OPS  ---------- */

/**
 * @brief Copy2 operation.
 *
 * This operation returns the second input value unchanged.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value (ignored).
 * @param b[in] The second input value.
 * @return The same value as the second input.
 */
struct copy2 { // this turns out to be a slightly hacky op that makes some code cleaner :/
    template<typename T> static inline T op(const T &a, const T &b) { return b; }
};
/**
 * @brief Sum operation.
 *
 * This operation calculates the sum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The sum of the input values.
 */
struct sum {
    template<typename T> static inline T op(const T &a, const T &b) { return a+b; }
    template<typename T>
    using sycl_op = sycl::plus<T>;
#if defined(__SYCL_DEVICE_ONLY__)
    template <typename T> static T reduce(vector_t<T, 16> &vec) {
      return inline_asm::__sum_f_16x16(vec);
    }
    template <typename T>
    static T reduce(vector_t<T, 8> &vec1, const vector_t<T, 8> &vec2) {
      return inline_asm::__sum_f_16x16(vec1, vec2);
    }
#endif
};
template <>
inline sycl::float2 sum::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{a.x() + b.x(), a.y() + b.y()};
}
template <>
inline bf16 sum::op<bf16>(const bf16 &a, const bf16 &b) {
    return a + b;
}
template <>
inline bf16_2 sum::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return a + b;
}
template <>
inline sycl::half sum::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hadd(a, b);
}
template <>
inline half_2 sum::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::hadd2(a, b);
}
/**
 * @brief Subtraction operation.
 *
 * This operation calculates the difference between two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The difference between the input values.
 */
struct sub {
    template<typename T> static inline T op(const T &a, const T &b) { return a-b; }
};
template <>
inline sycl::float2 sub::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{a.x() - b.x(), a.y() - b.y()};
}
template <>
inline bf16 sub::op<bf16>(const bf16 &a, const bf16 &b) {
    return a - b;
}
template <>
inline bf16_2 sub::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return a - b;
}
template <>
inline sycl::half sub::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hsub(a, b);
}
template <>
inline half_2 sub::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::hsub2(a, b);
}
/**
 * @brief Multiplication operation.
 *
 * This operation calculates the product of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The product of the input values.
 */
struct mul {
    template<typename T> static inline T op(const T &a, const T &b) { return a*b; }
    template<typename T>
    using sycl_op = sycl::multiplies<T>;
#if defined(__SYCL_DEVICE_ONLY__)
    template <typename T> static T reduce(vector_t<T, 16> &vec) {
        T result = vec[0];
        #pragma unroll
        for (int i = 1; i < 16; i++) result = op<T>(result, vec[i]);
        return result;
    }
    template <typename T>
    static T reduce(vector_t<T, 8> &vec1, const vector_t<T, 8> &vec2) {
        T result = vec1[0];
        #pragma unroll
        for (int i = 1; i < 8; i++) result = op<T>(result, vec1[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++) result = op<T>(result, vec2[i]);
        return result;
    }
#endif
};
template <>
inline sycl::float2 mul::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{a.x() * b.x(), a.y() * b.y()};
}
template <>
inline bf16 mul::op<bf16>(const bf16 &a, const bf16 &b) {
    return a * b;
}
template <>
inline bf16_2 mul::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return a * b;
}
template <>
inline sycl::half mul::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hmul(a, b);
}
template <>
inline half_2 mul::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::hmul2(a, b);
}
/**
 * @brief Division operation.
 *
 * This operation calculates the quotient of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The quotient of the input values.
 */
struct div {
    template<typename T> static inline T op(const T &a, const T &b) { return a/b; }
};
template <>
inline sycl::float2 div::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{a.x() / b.x(), a.y() / b.y()};
}
template <>
inline bf16 div::op<bf16>(const bf16 &a, const bf16 &b) {
    return a / b;
}
template <>
inline bf16_2 div::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return a / b;
} // this op is a special snowflake
template <>
inline sycl::half div::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hdiv(a, b);
}
template <>
inline half_2 div::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::h2div(a, b);
}
/**
 * @brief Maximum operation.
 *
 * This operation calculates the maximum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The maximum of the input values.
 */
struct max {
  template <typename T> static inline T op(const T &a, const T &b) {
    return dpct::max(a, b);
  }
  template <typename T> using sycl_op = sycl::maximum<T>;

#if defined(__SYCL_DEVICE_ONLY__)
  template <typename T> static T reduce(vector_t<T, 16> vec) {
    return inline_asm::__max_f_16x16(vec);
  }
  template <typename T>
  static T reduce(vector_t<T, 8> &vec1, const vector_t<T, 8> &vec2) {
    return inline_asm::__max_f_16x16(vec1, vec2);
  }
#endif
};
template <>
inline sycl::float2 max::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{sycl::max(a.x(), b.x()), sycl::max(a.y(), b.y())};
}
template <>
inline bf16 max::op<bf16>(const bf16 &a, const bf16 &b) {
    return sycl::ext::oneapi::experimental::fmax(a, b);
}
template <>
inline bf16_2 max::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return sycl::ext::oneapi::experimental::fmax(a, b);
}
template <>
inline sycl::half max::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hmax(a, b);
}
template <>
inline half_2 max::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::hmax2(a, b);
}
/**
 * @brief Minimum operation.
 *
 * This operation calculates the minimum of two input values.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @return The minimum of the input values.
 */
struct min {
    template <typename T>
    static inline T op(const T &a, const T &b) {
        return dpct::min(a, b);
    }
    template<typename T>
    using sycl_op = sycl::minimum<T>;
#if defined(__SYCL_DEVICE_ONLY__)
    template <typename T> static T reduce(vector_t<T, 16> &vec) {
        T result = vec[0];
        #pragma unroll
        for (int i = 1; i < 16; i++) result = op<T>(result, vec[i]);
        return result;
    }
    template <typename T>
    static T reduce(vector_t<T, 8> &vec1, const vector_t<T, 8> &vec2) {
        T result = vec1[0];
        #pragma unroll
        for (int i = 1; i < 8; i++) result = op<T>(result, vec1[i]);
        #pragma unroll
        for (int i = 0; i < 8; i++) result = op<T>(result, vec2[i]);
        return result;
    }
#endif
};
template <>
inline sycl::float2 min::op<sycl::float2>(const sycl::float2 &a,
                                          const sycl::float2 &b) {
    return sycl::float2{sycl::min(a.x(), b.x()), sycl::min(a.y(), b.y())};
}
template <>
inline bf16 min::op<bf16>(const bf16 &a, const bf16 &b) {
    return sycl::ext::oneapi::experimental::fmin(a, b);
}
template <>
inline bf16_2 min::op<bf16_2>(const bf16_2 &a, const bf16_2 &b) {
    return sycl::ext::oneapi::experimental::fmin(a, b);
}
template <>
inline sycl::half min::op<sycl::half>(const sycl::half &a,
                                      const sycl::half &b) {
    return sycl::ext::intel::math::hmin(a, b);
}
template <>
inline half_2 min::op<half_2>(const half_2 &a, const half_2 &b) {
    return sycl::ext::intel::math::hmin2(a, b);
}

/**
 * @brief Fused multiply-add operation A * B + C.
 *
 * This operation performs a fused multiply-add, computing (A * B) + C with only one rounding.
 *
 * @tparam T The data type of the input and output values.
 * @param a[in] The first input value.
 * @param b[in] The second input value.
 * @param c[in] The third input value to be added.
 * @return The result of the fused multiply-add operation.
 */
struct fused_madd {
    template <typename T>
    static inline T op(const T &a, const T &b, const T &c) {
        if constexpr (std::is_same_v<T, float>) {
            return sycl::fma(a, b, c);
        } else if constexpr (std::is_same_v<T, sycl::half>) {
            return sycl::fma(a, b, c);
        } else if constexpr (std::is_same_v<T, half_2>) {
            return sycl::ext::oneapi::experimental::fma(a, b, c);
        } else if constexpr (std::is_same_v<T, bf16> || std::is_same_v<T, bf16_2>) {
            return sycl::ext::oneapi::experimental::fma(a, b, c);
        } else {
            return sycl::fma(a, b, c);
        }
    }
};
/**
 * @brief Fused multiply-add operation A * B + C.
 */
struct fma_AxBtC {
    template<typename T> static inline T op(const T &a, const T &b, const T &c) {
        return fused_madd::op(a, b, c);
    }
};
/**
 * @brief Fused multiply-add operation A * C + B.
 *
 * This variant is useful for attention-style patterns where B is the addend.
 */
struct fma_AxCtB { // this is the one needed for attention
    template<typename T> static inline T op(const T &a, const T &b, const T &c) {
        return fused_madd::op(a, c, b);
    }
};

} // namespace base_ops

} // namespace kittens
