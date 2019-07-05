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
void Xcopy<T>::DoCopy(const size_t x_size, const Buffer<T>& x_buffer,
                      const size_t y_size, Buffer<T>& y_buffer)
{
  // Makes sure all dimensions are larger than zero
  if (x_size == 0 || y_size == 0)
      throw BLASError(StatusCode::kInvalidDimension);

  // Tests the vectors for validity
  TestVectorX(x_size, x_buffer, 0, 1);
  TestVectorY(y_size, y_buffer, 0, 1);

  // Determines whether or not the fast-version can be used
  bool use_fast_kernel = (x_size == y_size) &&
                         IsMultiple(y_size, db_["WGS"]*db_["WPT"]*db_["VW"]);

  // If possible, run the fast-version of the kernel
  auto kernel_name = use_fast_kernel ? "XcopyFast" : "Xcopy";

  // Retrieves the Xcopy kernel from the compiled binary
  auto kernel = program_.getKernel(kernel_name);

  // Sets the kernel arguments
  if (use_fast_kernel) {
    kernel.setArguments(static_cast<int>(y_size), x_buffer, y_buffer);
  } else {
    kernel.setArguments(static_cast<int>(x_size), x_buffer, static_cast<int>(y_size), y_buffer);
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

// The main routine
template <typename T>
void Xcopy<T>::DoCopyStrided(const size_t n,
    const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
    const std::vector<size_t>& stride, const std::vector<size_t>& shape)
{
    // Makes sure all dimensions are larger than zero
    if (n == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Create compact buffer to hold stride and shape
    auto rank = shape.size();
    assert(stride.size() == rank);
    std::vector<int> shape_data(rank * 2);
    std::copy(shape.begin(), shape.end(), shape_data.begin());
    std::copy(stride.begin(), stride.end(), shape_data.begin() + rank);
    Buffer<int> shape_buffer = context_.createBuffer<int>(rank*2, BufferAccess::WriteOnly);
    shape_buffer.write(queue_, shape_data.data(), shape_data.size());

    // Retrieves the Xcopy kernel from the compiled binary
    auto kernel = program_.getKernel("XcopyStrided");

    // Sets the kernel arguments
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, static_cast<int>(rank));
    kernel.setArgument(2, shape_buffer);
    kernel.setArgument(3, x_buffer);
    kernel.setArgument(4, y_buffer);

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

template class Xcopy<int16_t>;
template class Xcopy<int32_t>;
template class Xcopy<int64_t>;
template class Xcopy<half>;
template class Xcopy<float>;
template class Xcopy<double>;
template class Xcopy<float2>;
template class Xcopy<double2>;

}} // namespace gpgpu::dnn
