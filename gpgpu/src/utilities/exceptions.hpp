
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Ivan Shapovalov <intelfx@intelfx.name>
//
// This file implements the exception hierarchy for CLBlast. It contains classes for exceptions
// generated by different parts of CLBlast (e.g. OpenCL API calls, internal logic, semantic BLAS
// errors).
//
// =================================================================================================

#ifndef GPGPU_BLAS_EXCEPTIONS_H_
#define GPGPU_BLAS_EXCEPTIONS_H_

#include "utilities/utilities.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// Represents a semantic error in BLAS function arguments
class BLASError : public ErrorCode<Error<std::invalid_argument>, StatusCode> {
 public:
  explicit BLASError(StatusCode status, const std::string &subreason = std::string{});
};
// =================================================================================================

// Represents a runtime error generated by internal logic
class RuntimeErrorCode : public ErrorCode<RuntimeError, StatusCode> {
 public:
  explicit RuntimeErrorCode(StatusCode status, const std::string &subreason = std::string{});
};

// =================================================================================================

// Handles (most of the) runtime exceptions and converts them to StatusCode
StatusCode DispatchException(const bool silent = false);
StatusCode DispatchExceptionCatchAll(const bool silent = false);

// Handles remaining exceptions and converts them to StatusCode::kUnhandledError
StatusCode DispatchExceptionForC();

// =================================================================================================

}} // namespace gpgpu::blas

#endif // GPGPU_BLAS_EXCEPTIONS_H_
