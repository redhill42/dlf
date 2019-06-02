
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the common utility functions.
//
// =================================================================================================

#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

#include "utilities/utilities.hpp"
#include "utilities/device_mapping.hpp"

#if defined(__APPLE__) || defined(__MACOSX)
  #include "OpenCL/opencl.h"
#else
  #include "CL/cl.h"
#endif

#include <cuda.h>

using namespace gpgpu;

namespace gpgpu::blas {
// =================================================================================================

const half PrecisionTraits<half>::Zero{FloatToHalf(0.0f)};
const half PrecisionTraits<half>::One{FloatToHalf(1.0f)};
const half PrecisionTraits<half>::NegOne{FloatToHalf(-1.0f)};

template <> const Precision PrecisionTraits<float>::precision{Precision::Single};
template <> const Precision PrecisionTraits<double>::precision{Precision::Double};
template <> const Precision PrecisionTraits<float2>::precision{Precision::ComplexSingle};
template <> const Precision PrecisionTraits<double2>::precision{Precision::ComplexDouble};

// =================================================================================================

// Implements the string conversion using std::to_string if possible
template <typename T>
std::string ToString(T value) {
  return std::to_string(value);
}
template std::string ToString<int>(int value);
template std::string ToString<size_t>(size_t value);
template <>
std::string ToString(float value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}
template <>
std::string ToString(double value) {
  std::ostringstream result;
  result << std::fixed << std::setprecision(2) << value;
  return result.str();
}
template <> std::string ToString<std::string>(std::string value) { return value; }

// If not possible directly: special cases for complex data-types
template <>
std::string ToString(float2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
}
template <>
std::string ToString(double2 value) {
  return ToString(value.real())+"+"+ToString(value.imag())+"i";
}

// If not possible directly: special case for half-precision
template <>
std::string ToString(half value) {
  return std::to_string(HalfToFloat(value));
}

// If not possible directly: special cases for GPGPU data-types
template <>
std::string ToString(Layout value) {
  switch(value) {
    case Layout::RowMajor: return ToString(static_cast<int>(value))+" (row-major)";
    case Layout::ColMajor: return ToString(static_cast<int>(value))+" (col-major)";
  }
}
template <>
std::string ToString(Transpose value) {
  switch(value) {
    case Transpose::NoTrans: return ToString(static_cast<int>(value))+" (regular)";
    case Transpose::Trans: return ToString(static_cast<int>(value))+" (transposed)";
    case Transpose::ConjTrans: return ToString(static_cast<int>(value))+" (conjugate)";
  }
}
template <>
std::string ToString(Side value) {
  switch(value) {
    case Side::Left: return ToString(static_cast<int>(value))+" (left)";
    case Side::Right: return ToString(static_cast<int>(value))+" (right)";
  }
}
template <>
std::string ToString(Triangle value) {
  switch(value) {
    case Triangle::Upper: return ToString(static_cast<int>(value))+" (upper)";
    case Triangle::Lower: return ToString(static_cast<int>(value))+" (lower)";
  }
}
template <>
std::string ToString(Diagonal value) {
  switch(value) {
    case Diagonal::Unit: return ToString(static_cast<int>(value))+" (unit)";
    case Diagonal::NonUnit: return ToString(static_cast<int>(value))+" (non-unit)";
  }
}
template <>
std::string ToString(Precision value) {
  switch(value) {
    case Precision::Half: return ToString(static_cast<int>(value))+" (half)";
    case Precision::Single: return ToString(static_cast<int>(value))+" (single)";
    case Precision::Double: return ToString(static_cast<int>(value))+" (double)";
    case Precision::ComplexSingle: return ToString(static_cast<int>(value))+" (complex-single)";
    case Precision::ComplexDouble: return ToString(static_cast<int>(value))+" (complex-double)";
    case Precision::Any: return ToString(static_cast<int>(value))+" (any)";
  }
}
template <>
std::string ToString(KernelMode value) {
  switch(value) {
    case KernelMode::CrossCorrelation: return ToString(static_cast<int>(value))+" (cross-correlation)";
    case KernelMode::Convolution: return ToString(static_cast<int>(value))+" (convolution)";
  }
}
template <>
std::string ToString(StatusCode value) {
  return std::to_string(static_cast<int>(value));
}

// =================================================================================================

// Retrieves the squared difference, used for example for computing the L2 error
template <typename T>
double SquaredDifference(const T val1, const T val2) {
  const auto difference = (val1 - val2);
  return static_cast<double>(difference * difference);
}

// Compiles the default case for standard data-types
template double SquaredDifference<float>(const float, const float);
template double SquaredDifference<double>(const double, const double);

// Specialisations for non-standard data-types
template <>
double SquaredDifference(const float2 val1, const float2 val2) {
  const auto real = SquaredDifference(val1.real(), val2.real());
  const auto imag = SquaredDifference(val1.imag(), val2.imag());
  return real + imag;
}
template <>
double SquaredDifference(const double2 val1, const double2 val2) {
  const auto real = SquaredDifference(val1.real(), val2.real());
  const auto imag = SquaredDifference(val1.imag(), val2.imag());
  return real + imag;
}
template <>
double SquaredDifference(const half val1, const half val2) {
  return SquaredDifference(HalfToFloat(val1), HalfToFloat(val2));
}

// =================================================================================================

bool IsAMD(const Device& device) {
  std::string vendor = device.vendor();
  return vendor == "AMD"
      || vendor == "Advanced Micro Devices, Inc."
      || vendor == "AuthenticAMD";
}

bool IsNVIDIA(const Device& device) {
  std::string vendor = device.vendor();
  return vendor == "NVIDIA" || vendor == "NVIDIA Corporation";
}

bool IsIntel(const Device& device) {
  std::string vendor = device.vendor();
  return vendor == "INTEL"
      || vendor == "Intel"
      || vendor == "GenuineIntel"
      || vendor == "Intel(R) Corporation";
}

bool IsARM(const Device& device) {
  return device.vendor() == "ARM";
}

std::string AMDBoardName(const Device& device) {
  if (IsOpenCL(device)) {
    #ifndef CL_DEVICE_BOARD_NAME_AMD
      #define CL_DEVICE_BOARD_NAME_AMD 0x4038
    #endif
    auto device_id = reinterpret_cast<cl_device_id>(device.id());
    size_t bytes = 0;
    auto status = CL_SUCCESS;

    status = clGetDeviceInfo(device_id, CL_DEVICE_BOARD_NAME_AMD, 0, nullptr, &bytes);
    if (status != CL_SUCCESS)
      return {};

    std::string name;
    name.resize(bytes);
    clGetDeviceInfo(device_id, CL_DEVICE_BOARD_NAME_AMD, bytes, name.data(), nullptr);
    name.resize(strlen(name.c_str()));
    return name;
  }
  return {};
}

std::string NVIDIAComputeCapability(const Device& device) {
  if (IsCUDA(device))
    return device.capabilities();

  #ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
    #define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
  #endif
  #ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
    #define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
  #endif

  auto device_id = reinterpret_cast<cl_device_id>(device.id());
  cl_uint major, minor;
  auto status = CL_SUCCESS;

  status = clGetDeviceInfo(device_id, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(major), &major, nullptr);
  if (status != CL_SUCCESS)
    return {};
  status = clGetDeviceInfo(device_id, CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(minor), &minor, nullptr);
  if (status != CL_SUCCESS)
    return {};

  return "SM" + std::to_string(major) + "." + std::to_string(minor);
}

bool IsPostNVIDIAVolta(const Device& device) {
  if (IsCUDA(device)) {
    auto device_id = static_cast<CUdevice>(device.id());
    int info = 0;
    cuDeviceGetAttribute(&info, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_id);
    return info >= 7;
  }

  if (device.hasExtension("cl_nv_device_attribute_query")) {
    auto device_id = reinterpret_cast<cl_device_id>(device.id());
    cl_uint info;
    clGetDeviceInfo(device_id, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(info), &info, nullptr);
    return info >= 7;
  }

  return false;
}

// High-level info
std::string GetDeviceType(const Device& device) {
  return device.typeString();
}

std::string GetDeviceVendor(const Device& device) {
  auto device_vendor = device.vendor();
  for (auto& find_and_replace : device_mapping::kVendorNames) { // replacing to common names
    if (device_vendor == find_and_replace.first)
      device_vendor = find_and_replace.second;
  }
  return device_vendor;
}

// Mid-level info
std::string GetDeviceArchitecture(const Device& device) {
  std::string device_architecture;

  if (IsCUDA(device)) {
    device_architecture = NVIDIAComputeCapability(device);
  } else {
    if (device.hasExtension(kKhronosAttributesNVIDIA)) {
      device_architecture = NVIDIAComputeCapability(device);
    } else if (device.hasExtension(kKhronosAttributesAMD)) {
      device_architecture = device.name(); // Name is architecture for AMD APP and AMD ROCm
    } // Note: no else - 'device_architecture' might be the empty string
  }

  for (auto& find_and_replace : device_mapping::kArchitectureNames) { // replacing to common names
    if (device_architecture == find_and_replace.first) {
      device_architecture = find_and_replace.second;
    }
  }

  return device_architecture;
}

// Lowest-level
std::string GetDeviceName(const Device& device) {
  std::string device_name;

  if (device.hasExtension(kKhronosAttributesAMD)) {
    device_name = AMDBoardName(device);
  } else {
    device_name = device.name();
  }

  for (auto& find_and_replace : device_mapping::kDeviceNames) { // replacing to common names
    if (device_name == find_and_replace.first)
      device_name = find_and_replace.second;
  }

  for (auto& removal : device_mapping::kDeviceRemovals) { // removing certain things
    auto start_position_to_erase = device_name.find(removal);
    if (start_position_to_erase != std::string::npos) {
      device_name.erase(start_position_to_erase, removal.length());
    }
  }

  return device_name;
}

// =================================================================================================

// Solve Bezout's identity
// a * p + b * q = r = GCD(a, b)
void EuclidGCD(int a, int b, int &p, int &q, int &r) {
  p = 0;
  q = 1;
  int p_1 = 1;
  int q_1 = 0;
  for (;;) {
    const int c = a % b;
    if (c == 0) {
      break;
    }
    const int p_2 = p_1;
    const int q_2 = q_1;
    p_1 = p;
    q_1 = q;
    p = p_2 - p_1 * (a / b);
    q = q_2 - q_1 * (a / b);
    a = b;
    b = c;
  }
  r = b;
}

// =================================================================================================
} // namespace gpgpu::blas
