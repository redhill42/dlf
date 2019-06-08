
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the kernel compilation functions (see the header for more information).
//
// =================================================================================================

#include <vector>
#include <chrono>

#include "routines/common.hpp"
#include "preprocessor.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// Compiles a program from source code
Program CompileFromSource(const std::string& source_string,
                          const Precision precision,
                          const std::string& routine_name,
                          const Device& device, const Context& context,
                          std::vector<std::string>& options,
                          const size_t run_preprocessor) { // 0: platform dependent, 1: always, 2: never
  auto header_string = std::string{""};

  header_string += "#define PRECISION " + ToString(static_cast<int>(precision)) + "\n";

  // Adds the name of the routine as a define
  header_string += "#define ROUTINE_" + routine_name + "\n";

  // Not all OpenCL compilers support the 'inline' keyword. The keyword is only used for devices on
  // which it is known to work with all OpenCL platforms.
  if (IsNVIDIA(device) || IsARM(device)) {
    header_string += "#define USE_INLINE_KEYWORD 1\n";
  }

  // For specific devices, use the non-IEE754 compliant OpenCL mad() instruction. This can improve
  // performance, but might result in a reduced accuracy.
  if (IsAMD(device) && IsGPU(device) && !IsIntegral(precision)) {
    header_string += "#define USE_CL_MAD 1\n";
  }

  // For specific devices, use staggered/shuffled workgroup indices.
  if (IsAMD(device) && IsGPU(device)) {
    header_string += "#define USE_STAGGERED_INDICES 1\n";
  }

  // For specific devices add a global synchronisation barrier to the GEMM kernel to optimize
  // performance through better cache behaviour
  if (IsARM(device) && IsGPU(device)) {
    header_string += "#define GLOBAL_MEM_FENCE 1\n";
  }

  // For Intel GPUs with subgroup support, use subgroup shuffling.
  if (IsGPU(device) && device.hasExtension(kKhronosIntelSubgroups) &&
      (precision == Precision::Single || precision == Precision::Half)) {
    header_string += "#define USE_SUBGROUP_SHUFFLING 1\n";
    header_string += "#define SUBGROUP_SHUFFLING_INTEL 1\n";
  }

  // For NVIDIA GPUs, inline PTX can provide subgroup support
  if (IsGPU(device) && IsNVIDIA(device) && precision == Precision::Single) {
    header_string += "#define USE_SUBGROUP_SHUFFLING 1\n";

    // Nvidia needs to check pre or post volta due to new shuffle commands
    if (IsPostNVIDIAVolta(device)) {
      header_string += "#define SUBGROUP_SHUFFLING_NVIDIA_POST_VOLTA 1\n";
    } else {
      header_string += "#define SUBGROUP_SHUFFLING_NVIDIA_PRE_VOLTA 1\n";
    }
  }
  
  // Optionally adds a translation header from OpenCL kernels to CUDA kernels
  if (IsCUDA(device)) {
    header_string +=
      #include "kernels/opencl_to_cuda.cl"
      ;
  }

  // Loads the common header (typedefs and defines and such)
  header_string +=
    #include "kernels/common.cl"
    ;

  // Prints details of the routine to compile in case of debugging in verbose mode
  #ifdef VERBOSE
    printf("[DEBUG] Compiling routine '%s-%s'\n",
           routine_name.c_str(), ToString(precision).c_str());
    const auto start_time = std::chrono::steady_clock::now();
  #endif

  // Runs a pre-processor to unroll loops and perform array-to-register promotion. Most OpenCL
  // compilers do this, but some don't.
  auto do_run_preprocessor = false;
  if (run_preprocessor == 0) { do_run_preprocessor = (IsARM(device) && IsGPU(device)); }
  if (run_preprocessor == 1) { do_run_preprocessor = true; }
  auto kernel_string = header_string + source_string;
  if (do_run_preprocessor) {
    log_debug("Running built-in pre-processor");
    kernel_string = PreprocessKernelSource(kernel_string);
  }

  // Compiles the kernel
  auto program = context.compileProgram(kernel_string.c_str(), options);

  // Prints the elapsed compilation time in case of debugging in verbose mode
  #ifdef VERBOSE
    const auto elapsed_time = std::chrono::steady_clock::now() - start_time;
    const auto timing = std::chrono::duration<double,std::milli>(elapsed_time).count();
    printf("[DEBUG] Completed compilation in %.2lf ms\n", timing);
  #endif

  return program;
}

// =================================================================================================
}} // namespace gpgpu::blas
