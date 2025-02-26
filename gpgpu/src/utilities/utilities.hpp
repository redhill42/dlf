
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides declarations for the common utility functions such as a command-line
// argument parser. On top of this, it serves as the 'common' header, including the C++ OpenCL
// wrapper.
//
// =================================================================================================

#ifndef GPGPU_BLAS_UTILITIES_H_
#define GPGPU_BLAS_UTILITIES_H_

#include <string>
#include <functional>
#include <complex>
#include <random>
#include <algorithm>
#include <iterator>
#include <cassert>

#include "gblas.h"
#include "gblas_half.h"
#include "utilities/exceptions.hpp"

namespace gpgpu { namespace blas {

// =================================================================================================

// Shorthands for complex data-types
using float2 = std::complex<float>;
using double2 = std::complex<double>;

// Khronos OpenCL extensions
const std::string kKhronosAttributesAMD = "cl_amd_device_attribute_query";
const std::string kKhronosAttributesNVIDIA = "cl_nv_device_attribute_query";
const std::string kKhronosIntelSubgroups = "cl_intel_subgroups";

// =================================================================================================

#ifdef VERBOSE
inline void log_debug(const std::string &log_string) {
  printf("[DEBUG] %s\n", log_string.c_str());
}
#else
inline void log_debug(const std::string&) { }
#endif

// =================================================================================================

// Precision traits

template <typename T>
struct PrecisionTraits {
  // Scalar of value 0, 1, and -1
  static constexpr T Zero{0}, One{1}, NegOne{static_cast<T>(-1)};

  // Converts a 'real' value to a 'real argument' value to be passed to kernel.
  // Normally there is no conversion, but half-precision is not supported as
  // kernel argument so it is converted to float.
  using RealArg = T;
  static RealArg GetRealArg(const T value) { return value; }

  // Convert the template argument into a precision value
  static const Precision precision;

  // Returns false if this precision is not supported by the device.
  static bool Supported(const Device&) { return true; }
};

template <> inline bool PrecisionTraits<double>::Supported(const Device& device) {
  return device.supportsFP64();
}
template <> inline bool PrecisionTraits<double2>::Supported(const Device& device) {
  return device.supportsFP64();
}

template <typename T>
constexpr T PrecisionTraits<T>::Zero;
template <typename T>
constexpr T PrecisionTraits<T>::One;
template <typename T>
constexpr T PrecisionTraits<T>::NegOne;

template <>
struct PrecisionTraits<half> {
  static const half Zero, One, NegOne;

  static constexpr Precision precision = Precision::Half;
  static bool Supported(const Device& device) { return device.supportsFP16(); }

  using RealArg = float;
  static RealArg GetRealArg(const half value) { return static_cast<RealArg>(value); }
};

template <typename T> inline T ConstantZero()   { return PrecisionTraits<T>::Zero; }
template <typename T> inline T ConstantOne()    { return PrecisionTraits<T>::One; }
template <typename T> inline T ConstantNegOne() { return PrecisionTraits<T>::NegOne; }

template <typename T>
inline auto GetRealArg(const T value) {
  return PrecisionTraits<T>::GetRealArg(value);
}

template <typename T>
inline Precision PrecisionValue() {
  return PrecisionTraits<T>::precision;
}

template <typename T>
inline bool PrecisionSupported(const Device& device) {
  return PrecisionTraits<T>::Supported(device);
}

inline bool IsIntegral(Precision precision) {
    return static_cast<int>(precision) > 10000;
}

inline Precision DatabasePrecision(Precision precision) {
    if (precision == Precision::Char)
        return Precision::Single;
    if (IsIntegral(precision))
        return static_cast<Precision>(static_cast<int>(precision) - 10000);
    return precision;
}

// =================================================================================================

// Converts a value (e.g. an integer) to a string. This also covers special cases for GPGPU
// data-types such as the Layout and Transpose data-types.
template <typename T>
std::string ToString(T value);

// =================================================================================================

// String splitting by a delimiter
template <typename Out>
void split(const std::string& s, char delimiter, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    *(result++) = item;
  }
}

// See above
inline std::vector<std::string> split(const std::string& s, char delimiter) {
  std::vector<std::string> elements;
  split(s, delimiter, std::back_inserter(elements));
  return elements;
}

// String character removal
inline void remove_character(std::string& str, char to_be_removed) {
  str.erase(std::remove(str.begin(), str.end(), to_be_removed), str.end());
}

// =================================================================================================

// Rounding functions
inline size_t CeilDiv(const size_t x, const size_t y) {
  return 1 + ((x - 1) / y);
}

inline size_t Ceil(const size_t x, const size_t y) {
  return CeilDiv(x, y) * y;
}

// Returns whether or not 'a' is a multiple of 'b'
inline bool IsMultiple(const size_t a, const size_t b) {
  return a % b == 0;
}

inline size_t NextPowerOfTwo(size_t x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return ++x;
}

// =================================================================================================

// Retrieves the squared difference, used for example for computing the L2 error
template <typename T>
double SquaredDifference(const T val1, const T val2);

// =================================================================================================

inline bool IsOpenCL(const Device& device) { return device.platform().api() == APIType::OpenCL; }
inline bool IsCUDA(const Device& device) { return device.platform().api() == APIType::CUDA; }

inline bool IsCPU(const Device& device) { return device.type() == DeviceType::CPU; }
inline bool IsGPU(const Device& device) { return device.type() == DeviceType::GPU; }

std::string GetDeviceType(const Device& device);
std::string GetDeviceVendor(const Device& device);
std::string GetDeviceArchitecture(const Device& device);
std::string GetDeviceName(const Device& device);

bool IsAMD(const Device& device);
bool IsNVIDIA(const Device& device);
bool IsIntel(const Device& device);
bool IsARM(const Device& device);

std::string AMDBoardName(const Device& device);
std::string NVIDIAComputeCapability(const Device& device);
bool IsPostNVIDIAVolta(const Device& device);

// =================================================================================================

// Solve Bezout's identity
// a * p + b * q = r = GCD(a, b)
void EuclidGCD(int a, int b, int &p, int &q, int &r);

// =================================================================================================

bool IsContiguous(const std::vector<size_t>& dim, const std::vector<size_t>& stride);

inline size_t GetSize(const std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
}

inline size_t GetBatchSize(const std::vector<size_t>& dims) {
    return std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
}

std::vector<size_t> MakeFlatShape(const std::vector<size_t>& dims);

Buffer<int> PackShape(const std::vector<size_t>& dim,
                      const std::vector<size_t>& stride,
                      const Context& context, const Queue& queue);
Buffer<int> PackShape(const std::vector<size_t>& dim,
                      const std::vector<size_t>& x_stride,
                      const std::vector<size_t>& y_stride,
                      const Context& context, const Queue& queue);
Buffer<int> PackShape(const std::vector<size_t>& dim,
                      const std::vector<size_t>& x_stride,
                      const std::vector<size_t>& y_stride,
                      const std::vector<size_t>& z_stride,
                      const Context& context, const Queue& queue);

}} // namespace gpgpu::blas

// =================================================================================================

/* Macro to facilitate debugging
 * Usage:
 *   Place CL_PROGRAM on the line before the first line of your source.
 *   The first line ends with:   CL_PROGRAM \"
 *   Each line thereafter of OpenCL C source must end with: \n\
 *   The last line ends in ";
 *
 *   Example:
 *
 *   const char *my_program = CL_PROGRAM "\
 *   kernel void foo( int a, float * b )             \n\
 *   {                                               \n\
 *      // my comment                                \n\
 *      *b[ get_global_id(0)] = a;                   \n\
 *   }                                               \n\
 *   ";
 *
 * This should correctly set up the line, (column) and file information for your source
 * string so you can do source level debugging.
 */
#ifndef NDEBUG
#define  __STRINGIFY( _x )  # _x
#define  _STRINGIFY( _x )   __STRINGIFY( _x )
#define  CL_PROGRAM         "#line "  _STRINGIFY(__LINE__) " \"" __FILE__ "\" \n\n"
#else
#define  CL_PROGRAM
#endif

#endif // GPGPU_BLAS_UTILITIES_H_
