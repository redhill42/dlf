
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file contains all the common code to perform (partial) matrix inverting. This code is based
// on the TRSM implementation in the CUDA version of Magma version 2.2.0 and the poster "Triangular
// Linear System Solver for GPU with CUDA and OpenCL" by Peng Du, Stanimire Tomov, Piotr Luszczek,
// and Jack Dongarra.
//
// =================================================================================================

#include "routines/levelx/xinvert.hpp"

#include <string>
#include <vector>
#include <assert.h>

namespace gpgpu::blas {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xinvert<T>::Xinvert(const Queue &queue, Event* event, const std::string &name):
    Routine(queue, event, name, {"Invert"}, PrecisionValue<T>(), {}, {
      #include "../../kernels/level3/level3.cl"
      , // separated in multiple parts to prevent C1091 in MSVC 2013
      #include "../../kernels/level3/invert_diagonal_blocks_part1.cl"
      , // separated in multiple parts to prevent C1091 in MSVC 2013
      #include "../../kernels/level3/invert_diagonal_blocks_part2.cl"
    }) {
}

// =================================================================================================

// Inverts diagonal square blocks of a matrix
template <typename T>
void Xinvert<T>::InvertMatrixDiagonalBlocks(const Layout layout, const Triangle triangle, const Diagonal diag,
                                            const size_t n, const size_t block_size,
                                            const Buffer<T> &src, const size_t offset, const size_t ld_src,
                                            Buffer<T> &dest) {

  // Makes sure all dimensions are larger than zero
  if ((block_size == 0) || (n == 0)) {
    throw BLASError(StatusCode::kInvalidDimension);
  }

  // Some parts of this kernel are not tunable and thus require some minimal OpenCL properties
  if (device_.maxWorkGroupSize() < 16) { // minimum of total local work size of 16
    throw RuntimeErrorCode(StatusCode::kNotImplemented);
  }

  // Helper variables
  const auto internal_block_size = static_cast<size_t>(db_["INTERNAL_BLOCK_SIZE"]);
  if (internal_block_size != 16) {
    throw RuntimeErrorCode(StatusCode::kNotImplemented); // e.g. Apple CPU OpenCL with a WGS of 1
  }                                                      // when barriers are present
  const auto num_blocks = CeilDiv(n, block_size);
  const auto num_internal_blocks = CeilDiv(n, internal_block_size);
  const auto unit_diagonal = (diag == Diagonal::Unit) ? true : false;

  // This routine only supports block sizes which are a multiple of the internal block size and
  // block sizes up to and including 128
  if ((block_size % internal_block_size != 0) || (block_size > 128)) {
    throw BLASError(StatusCode::kUnknownError);
  }

  // Checks for validity of the source and destination matrices
  TestMatrixA(n, n, src, offset, ld_src);
  TestMatrixB(block_size, num_blocks * block_size, dest, 0, block_size);

  // Determines which kernels to run based on the layout (the kernels assume column-major as
  // default) and on whether we are dealing with an upper or lower triangle of the triangular matrix
  const bool is_upper = ((triangle == Triangle::Upper && layout != Layout::RowMajor) ||
                         (triangle == Triangle::Lower && layout == Layout::RowMajor));
  const auto name_postfix = (is_upper) ? "Upper" : "Lower";

  // Fills the output buffer with zeros
  auto event_wait_list = std::vector<Event>();
  auto fill_matrix_event = context_.createEvent();
  FillMatrix(queue_, device_, program_, &fill_matrix_event, event_wait_list,
             block_size, num_blocks * block_size, block_size, 0, dest, ConstantZero<T>(),
             16);
  event_wait_list.push_back(fill_matrix_event);

  // Inverts the diagonal IB by IB inner blocks of the matrix: one block per work-group
  auto kernel = program_.getKernel("InvertDiagonalBlock");
  kernel.setArgument(0, static_cast<int>(n));
  kernel.setArgument(1, src);
  kernel.setArgument(2, static_cast<int>(offset));
  kernel.setArgument(3, static_cast<int>(ld_src));
  kernel.setArgument(4, dest);
  kernel.setArgument(5, static_cast<int>(block_size));
  kernel.setArgument(6, static_cast<int>(unit_diagonal));
  kernel.setArgument(7, static_cast<int>(is_upper));
  const auto local_invert = std::vector<size_t>{internal_block_size};
  const auto global_invert = std::vector<size_t>{num_internal_blocks * internal_block_size};
  auto base_kernel_event = context_.createEvent();
  auto base_kernel_event_pointer = (internal_block_size == block_size) ? event_ : &base_kernel_event;
  RunKernel(kernel, queue_, device_, global_invert, local_invert, base_kernel_event_pointer, event_wait_list);
  if (internal_block_size == block_size) { event_wait_list.push_back(base_kernel_event); }

  // Builds up block_size x block_size blocks. For example, internal_block_size=16:
  // use   16 x 16  blocks to build  32 x 32  blocks,  1 x (1 x npages) grid,  4 x 4 threads;
  // then  32 x 32  blocks to build  64 x 64  blocks,  1 x (2 x npages) grid,  8 x 4 threads;
  // then  64 x 64  blocks to build 128 x 128 blocks,  1 x (4 x npages) grid, 16 x 4 threads;
  for (auto current_size = internal_block_size; current_size < block_size; current_size *= 2) {
    assert(current_size == 16 || current_size == 32 || current_size == 64);

    // Emulates a 3D grid: NX * (NY * npages)
    const auto npages = CeilDiv(n, current_size*2);
    const auto local0 = (current_size <= 32) ? current_size/4 : 16;
    const auto local = std::vector<size_t>{local0, 4};
    const auto global = std::vector<size_t>{Ceil(current_size/local[1], local[0]),
                                            Ceil(npages*(current_size/16)*local[1], local[1])};

    // Part 1
    auto kernel1 = program_.getKernel("TripleMatMul" + ToString(current_size) + "Part1" + name_postfix);
    kernel1.setArgument(0, static_cast<int>(n));
    kernel1.setArgument(1, src);
    kernel1.setArgument(2, static_cast<int>(offset));
    kernel1.setArgument(3, static_cast<int>(ld_src));
    kernel1.setArgument(4, dest);
    kernel1.setArgument(5, static_cast<int>(current_size));
    kernel1.setArgument(6, static_cast<int>(npages));
    kernel1.setArgument(7, static_cast<int>(block_size));
    auto kernel1_event = context_.createEvent();
    RunKernel(kernel1, queue_, device_, global, local, &kernel1_event, event_wait_list);
    event_wait_list.push_back(kernel1_event);

    // Part 2
    const bool is_last_kernel = (current_size * 2 >= block_size);
    auto kernel2 = program_.getKernel("TripleMatMul" + ToString(current_size) + "Part2" + name_postfix);
    kernel2.setArgument(0, static_cast<int>(n));
    kernel2.setArgument(1, dest);
    kernel2.setArgument(2, static_cast<int>(current_size));
    kernel2.setArgument(3, static_cast<int>(npages));
    kernel2.setArgument(4, static_cast<int>(block_size));
    auto kernel2_event = context_.createEvent();
    auto kernel2_event_pointer = (is_last_kernel) ? event_ : &kernel2_event;
    RunKernel(kernel2, queue_, device_, global, local, kernel2_event_pointer, event_wait_list);
    if (!is_last_kernel) { event_wait_list.push_back(kernel2_event); }

    // Exit in case we reach beyond the bounds of the input matrix
    if (current_size*2 >= n) { break; }
  }
}

// =================================================================================================

// Compiles the templated class
template class Xinvert<half>;
template class Xinvert<float>;
template class Xinvert<double>;
template class Xinvert<float2>;
template class Xinvert<double2>;

// =================================================================================================
} // namespace gpgpu::blas
