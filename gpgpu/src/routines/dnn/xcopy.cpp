#include "xcopy.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xcopy<T>::Xcopy(const Queue& queue, Event* event, const std::string& name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xcopy.cl"
    }) {
}

// The main routine
template <typename T>
void Xcopy<T>::DoCopy(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
                      const size_t y_size, Buffer<T>& y_buffer, const size_t y_offset)
{
  // Makes sure all dimensions are larger than zero
  if (x_size == 0 || y_size == 0)
      throw BLASError(StatusCode::kInvalidDimension);

  // Tests the vectors for validity
  TestVectorX(x_size, x_buffer, x_offset, 1);
  TestVectorY(y_size, y_buffer, y_offset, 1);

  // Determines whether or not the fast-version can be used
  bool use_fast_kernel = (x_size == y_size) && (x_offset == 0) && (y_offset == 0) &&
                         IsMultiple(y_size, db_["WGS"]*db_["WPT"]*db_["VW"]);

  // If possible, run the fast-version of the kernel
  auto kernel_name = use_fast_kernel ? "XcopyFast" : "Xcopy";

  // Retrieves the Xcopy kernel from the compiled binary
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.setArguments(static_cast<int>(y_size), x_buffer, y_buffer);
  } else {
    kernel.setArguments(static_cast<int>(x_size), x_buffer, static_cast<int>(x_offset),
                        static_cast<int>(y_size), y_buffer, static_cast<int>(y_offset));
  }

  // Launches the kernel
  if (use_fast_kernel) {
    auto global = std::vector<size_t>{CeilDiv(y_size, db_["WPT"]*db_["VW"])};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
  else {
    auto n_ceiled = Ceil(y_size, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
  }
}

static bool is_contiguous(const std::vector<size_t>& dim, const std::vector<size_t>& stride) {
  size_t size = 1;
  for (int i = dim.size(); --i >= 0; ) {
      if (stride[i] == 0 && dim[i] == 1)
          continue;
      if (stride[i] != size)
          return false;
      size *= dim[i];
  }
  return true;
}

// The main routine
template <typename T>
void Xcopy<T>::DoCopyStrided(const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride)
{
    if (is_contiguous(dims, x_stride) && is_contiguous(dims, y_stride)) {
        DoCopy(n, x_buffer, x_offset, n, y_buffer, y_offset);
        return;
    }

    // Create compact buffer to hold strides and dims
    auto rank = dims.size();
    assert(x_stride.size() == rank && y_stride.size() == rank);
    std::vector<int> shape_data(rank * 3);
    std::copy(dims.begin(), dims.end(), shape_data.begin());
    std::copy(x_stride.begin(), x_stride.end(), shape_data.begin() + rank);
    std::copy(y_stride.begin(), y_stride.end(), shape_data.begin() + rank*2);
    auto shape_buffer = context_.getSharedBuffer<int>(shape_data.data(), shape_data.size(), queue_);

    auto kernel = program_.getKernel("XcopyStrided");
    kernel.setArguments(static_cast<int>(n), static_cast<int>(rank),
                        shape_buffer,
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset));

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xcopy<T>::DoConcatCopy(const size_t n,
                            const size_t offset, const size_t block, const size_t stride,
                            const Buffer<T>& x_buffer, Buffer<T>& y_buffer)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    TestVectorX(n, x_buffer, 0, 1);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xconcat_copy");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n),
                        static_cast<int>(offset),
                        static_cast<int>(block),
                        static_cast<int>(stride),
                        x_buffer, y_buffer);

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xcopy<T>::DoSplitCopy(const size_t n,
                           const size_t offset, const size_t block, const size_t stride,
                           const Buffer<T>& x_buffer, Buffer<T>& y_buffer)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    TestVectorX(n, x_buffer, 0, 1);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xsplit_copy");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n),
                        static_cast<int>(offset),
                        static_cast<int>(block),
                        static_cast<int>(stride),
                        x_buffer, y_buffer);

    // Launches the kernel
    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xcopy<int16_t>;
template class Xcopy<int32_t>;
template class Xcopy<int64_t>;
template class Xcopy<half>;
template class Xcopy<float>;
template class Xcopy<double>;
template class Xcopy<float2>;
template class Xcopy<double2>;

}} // namespace gpgpu::dnn
