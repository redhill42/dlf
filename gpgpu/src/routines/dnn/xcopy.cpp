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

static Buffer<int> pack_shape(
    const Context& context, const Queue& queue,
    const std::vector<size_t>& dim, const std::vector<size_t>& stride)
{
    auto rank = dim.size();
    assert(stride.size() == rank);

    std::vector<int> shape(rank * 2);
    std::copy(dim.begin(), dim.end(), shape.begin());
    std::copy(stride.begin(), stride.end(), shape.begin() + rank);

    Buffer<int> shape_buffer = context.createBuffer<int>(rank*2, BufferAccess::WriteOnly);
    shape_buffer.write(queue, shape.data(), shape.size());
    return shape_buffer;
}

// The main routine
template <typename T>
void Xcopy<T>::DoCopyStrided(const size_t n,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset,
    const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride)
{
    if (is_contiguous(x_dim, x_stride) && is_contiguous(y_dim, y_stride)) {
        DoCopy(n, x_buffer, x_offset, n, y_buffer, y_offset);
        return;
    }

    Buffer<int> x_shape, y_shape;
    Kernel kernel;

    if (is_contiguous(y_dim, y_stride)) {
        x_shape = pack_shape(context_, queue_, x_dim, x_stride);
        kernel = program_.getKernel("XcopyStridedL");
        kernel.setArguments(static_cast<int>(n),
                            x_buffer, static_cast<int>(x_offset),
                            static_cast<int>(x_dim.size()), x_shape,
                            y_buffer, static_cast<int>(y_offset));
    } else if (is_contiguous(x_dim, x_stride)) {
        y_shape = pack_shape(context_, queue_, y_dim, y_stride);
        kernel = program_.getKernel("XcopyStridedR");
        kernel.setArguments(static_cast<int>(n),
                            x_buffer, static_cast<int>(x_offset),
                            static_cast<int>(y_dim.size()), y_shape,
                            y_buffer, static_cast<int>(y_offset));
    } else {
        x_shape = pack_shape(context_, queue_, x_dim, x_stride);
        y_shape = pack_shape(context_, queue_, y_dim, y_stride);
        kernel = program_.getKernel("XcopyStridedLR");
        kernel.setArguments(static_cast<int>(n),
                            static_cast<int>(x_dim.size()), x_shape,
                            x_buffer, static_cast<int>(x_offset),
                            static_cast<int>(y_dim.size()), y_shape,
                            y_buffer, static_cast<int>(y_offset));
    }

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
