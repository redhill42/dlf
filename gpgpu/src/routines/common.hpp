
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the interfaces to common kernels, such as copying, padding, and
// transposing a matrix. These functions are templated and thus header-only. This file also contains
// other common functions to routines, such as a function to launch a kernel.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINES_COMMON_H_
#define GPGPU_BLAS_ROUTINES_COMMON_H_

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "utilities/compile.hpp"
#include "database/database.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// Enqueues a kernel, waits for completion, and checks for errors
void RunKernel(const Kernel& kernel, const Queue& queue, const Device& device,
               std::vector<size_t> global, const std::vector<size_t>& local,
               Event* event);

// =================================================================================================

// Sets all elements of a matrix to a constant value
template <typename T>
void FillMatrix(const Queue& queue, const Device& device,
                const Program& program, Event* event,
                const size_t m, const size_t n, const size_t ld, const size_t offset,
                Buffer<T> &dest, const T constant_value, const size_t local_size);

// Sets all elements of a vector to a constant value
template <typename T>
void FillVector(const Queue &queue, const Device &device,
                const Program& program, Event* event,
                const size_t n, const size_t inc, const size_t offset,
                Buffer<T>& dest, const T constant_value, const size_t local_size);

// =================================================================================================

// Copies or transposes a matrix and optionally pads/unpads it with zeros. This method is also able
// to write to symmetric and triangular matrices through optional arguments.
template <typename T>
void PadCopyTransposeMatrix(const Queue& queue, const Device& device,
                            const Databases& db, Event* event,
                            const size_t src_one, const size_t src_two,
                            const size_t src_ld, const size_t src_offset,
                            const Buffer<T>& src,
                            const size_t dest_one, const size_t dest_two,
                            const size_t dest_ld, const size_t dest_offset,
                            Buffer<T>& dest,
                            const T alpha,
                            const Program& program, const bool do_pad,
                            const bool do_transpose, const bool do_conjugate,
                            const bool upper = false, const bool lower = false,
                            const bool diagonal_imag_zero = false) {

  // Determines whether or not the fast-version could potentially be used
  auto use_fast_kernel = (src_offset == 0) && (dest_offset == 0) && (do_conjugate == false) &&
                         (src_one == dest_one) && (src_two == dest_two) && (src_ld == dest_ld) &&
                         (upper == false) && (lower == false) && (diagonal_imag_zero == false);

  // Determines the right kernel
  auto kernel_name = std::string{};
  auto pad_kernel = false;
  if (do_transpose) {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db["TRA_WPT"]) &&
        IsMultiple(src_one, db["TRA_WPT"]*db["TRA_DIM"]) &&
        IsMultiple(src_two, db["TRA_WPT"]*db["TRA_DIM"])) {
      kernel_name = "TransposeMatrixFast";
    }
    else {
      use_fast_kernel = false;
      pad_kernel = (do_pad || do_conjugate);
      kernel_name = (pad_kernel) ? "TransposePadMatrix" : "TransposeMatrix";
    }
  }
  else {
    if (use_fast_kernel &&
        IsMultiple(src_ld, db["COPY_VW"]) &&
        IsMultiple(src_one, db["COPY_VW"]*db["COPY_DIMX"]) &&
        IsMultiple(src_two, db["COPY_WPT"]*db["COPY_DIMY"])) {
      kernel_name = "CopyMatrixFast";
    }
    else {
      use_fast_kernel = false;
      pad_kernel = do_pad;
      kernel_name = (pad_kernel) ? "CopyPadMatrix" : "CopyMatrix";
    }
  }

  // Retrieves the kernel from the compiled binary
  auto kernel = program.getKernel(kernel_name.c_str());

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.setArgument(0, static_cast<int>(src_ld));
    kernel.setArgument(1, src);
    kernel.setArgument(2, dest);
    kernel.setArgument(3, GetRealArg(alpha));
  }
  else {
    kernel.setArgument(0, static_cast<int>(src_one));
    kernel.setArgument(1, static_cast<int>(src_two));
    kernel.setArgument(2, static_cast<int>(src_ld));
    kernel.setArgument(3, static_cast<int>(src_offset));
    kernel.setArgument(4, src);
    kernel.setArgument(5, static_cast<int>(dest_one));
    kernel.setArgument(6, static_cast<int>(dest_two));
    kernel.setArgument(7, static_cast<int>(dest_ld));
    kernel.setArgument(8, static_cast<int>(dest_offset));
    kernel.setArgument(9, dest);
    kernel.setArgument(10, GetRealArg(alpha));
    if (pad_kernel) {
      kernel.setArgument(11, static_cast<int>(do_conjugate));
    }
    else {
      kernel.setArgument(11, static_cast<int>(upper));
      kernel.setArgument(12, static_cast<int>(lower));
      kernel.setArgument(13, static_cast<int>(diagonal_imag_zero));
    }
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
        dest_one / db["TRA_WPT"],
        dest_two / db["TRA_WPT"]
      };
      const auto local = std::vector<size_t>{db["TRA_DIM"], db["TRA_DIM"]};
      RunKernel(kernel, queue, device, global, local, event);
    }
    else {
      const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
        Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"])
      };
      const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"]};
      RunKernel(kernel, queue, device, global, local, event);
    }
  }
  else {
    if (use_fast_kernel) {
      const auto global = std::vector<size_t>{
        dest_one / db["COPY_VW"],
        dest_two / db["COPY_WPT"]
      };
      const auto local = std::vector<size_t>{db["COPY_DIMX"], db["COPY_DIMY"]};
      RunKernel(kernel, queue, device, global, local, event);
    }
    else {
      const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
        Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"])
      };
      const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"]};
      RunKernel(kernel, queue, device, global, local, event);
    }
  }
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixBatched(const Queue& queue, const Device& device,
                                   const Databases& db, Event* event,
                                   const size_t src_one, const size_t src_two,
                                   const size_t src_ld, const Buffer<int>& src_offsets,
                                   const Buffer<T>& src,
                                   const size_t dest_one, const size_t dest_two,
                                   const size_t dest_ld, const Buffer<int>& dest_offsets,
                                   const Buffer<T>& dest,
                                   const Program& program, const bool do_pad,
                                   const bool do_transpose, const bool do_conjugate,
                                   const size_t batch_count) {

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    kernel_name = (do_pad) ? "TransposePadMatrixBatched" : "TransposeMatrixBatched";
  }
  else {
    kernel_name = (do_pad) ? "CopyPadMatrixBatched" : "CopyMatrixBatched";
  }

  // Retrieves the kernel from the compiled binary
  auto kernel = program.getKernel(kernel_name.c_str());

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(src_one));
  kernel.setArgument(1, static_cast<int>(src_two));
  kernel.setArgument(2, static_cast<int>(src_ld));
  kernel.setArgument(3, src_offsets);
  kernel.setArgument(4, src);
  kernel.setArgument(5, static_cast<int>(dest_one));
  kernel.setArgument(6, static_cast<int>(dest_two));
  kernel.setArgument(7, static_cast<int>(dest_ld));
  kernel.setArgument(8, dest_offsets);
  kernel.setArgument(9, dest);
  if (do_pad) {
    kernel.setArgument(10, static_cast<int>(do_conjugate));
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    const auto global = std::vector<size_t>{
      Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
      Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
      batch_count
    };
    const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"], 1};
    RunKernel(kernel, queue, device, global, local, event);
  }
  else {
    const auto global = std::vector<size_t>{
      Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
      Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"]),
      batch_count
    };
    const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"], 1};
    RunKernel(kernel, queue, device, global, local, event);
  }
}

// Batched version of the above
template <typename T>
void PadCopyTransposeMatrixStridedBatched(const Queue& queue, const Device& device,
                                          const Databases& db, Event* event,
                                          const size_t src_one, const size_t src_two,
                                          const size_t src_ld, const size_t src_offset,
                                          const size_t src_stride, const Buffer<T> &src,
                                          const size_t dest_one, const size_t dest_two,
                                          const size_t dest_ld, const size_t dest_offset,
                                          const size_t dest_stride, const Buffer<T> &dest,
                                          const Program& program, const bool do_pad,
                                          const bool do_transpose, const bool do_conjugate,
                                          const size_t batch_count) {

  // Determines the right kernel
  auto kernel_name = std::string{};
  if (do_transpose) {
    kernel_name = (do_pad) ? "TransposePadMatrixStridedBatched" : "TransposeMatrixStridedBatched";
  }
  else {
    kernel_name = (do_pad) ? "CopyPadMatrixStridedBatched" : "CopyMatrixStridedBatched";
  }

  // Retrieves the kernel from the compiled binary
  auto kernel = program.getKernel(kernel_name.c_str());

  // Sets the kernel arguments
  kernel.setArgument(0, static_cast<int>(src_one));
  kernel.setArgument(1, static_cast<int>(src_two));
  kernel.setArgument(2, static_cast<int>(src_ld));
  kernel.setArgument(3, static_cast<int>(src_offset));
  kernel.setArgument(4, static_cast<int>(src_stride));
  kernel.setArgument(5, src);
  kernel.setArgument(6, static_cast<int>(dest_one));
  kernel.setArgument(7, static_cast<int>(dest_two));
  kernel.setArgument(8, static_cast<int>(dest_ld));
  kernel.setArgument(9, static_cast<int>(dest_offset));
  kernel.setArgument(10, static_cast<int>(dest_stride));
  kernel.setArgument(11, dest);
  if (do_pad) {
    kernel.setArgument(12, static_cast<int>(do_conjugate));
  }

  // Launches the kernel and returns the error code. Uses global and local thread sizes based on
  // parameters in the database.
  if (do_transpose) {
    const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
        Ceil(CeilDiv(dest_two, db["PADTRA_WPT"]), db["PADTRA_TILE"]),
        batch_count
    };
    const auto local = std::vector<size_t>{db["PADTRA_TILE"], db["PADTRA_TILE"], 1};
    RunKernel(kernel, queue, device, global, local, event);
  }
  else {
    const auto global = std::vector<size_t>{
        Ceil(CeilDiv(dest_one, db["PAD_WPTX"]), db["PAD_DIMX"]),
        Ceil(CeilDiv(dest_two, db["PAD_WPTY"]), db["PAD_DIMY"]),
        batch_count
    };
    const auto local = std::vector<size_t>{db["PAD_DIMX"], db["PAD_DIMY"], 1};
    RunKernel(kernel, queue, device, global, local, event);
  }
}

// =================================================================================================
}} // namespace gpgpu::blas

#endif // GPGPU_BLAS_ROUTINES_COMMON_H_
