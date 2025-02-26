
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains the CLBlast way to compile a kernel from source, used for the library and for
// the auto-tuners.
//
// =================================================================================================

#ifndef GPGPU_BLAS_UTILITIES_COMPILE_H_
#define GPGPU_BLAS_UTILITIES_COMPILE_H_

#include <string>
#include <vector>

#include "utilities/utilities.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// Compiles a program from source code
Program CompileFromSource(const std::string& source_string, const Precision precision,
                          const std::string& routine_name,
                          const Device& device, const Context& context,
                          std::vector<std::string>& options,
                          const size_t run_preprocessor); // 0: platform dependent, 1: always, 2: never

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_UTILITIES_COMPILE_H_
#endif
