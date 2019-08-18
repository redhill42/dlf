#include "xreduce.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xreduce<T>::Xreduce(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xreduce.cl"
    }) {
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

template <typename T>
void Xreduce<T>::DoReduce(const std::string& name, const size_t m, const size_t n,
                          const std::vector<size_t>& dims, const std::vector<size_t>& strides,
                          const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
                          gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    Kernel kernel;

    if (is_contiguous(dims, strides)) {
        kernel = program_.getKernel("X" + name);
        kernel.setArguments(static_cast<int>(n),
                            x_buffer, static_cast<int>(x_offset),
                            y_buffer, static_cast<int>(y_offset));
    } else {
        auto rank = dims.size();
        assert(strides.size() == rank);
        std::vector<int> shape_data(rank * 2);
        std::copy(dims.begin(), dims.end(), shape_data.begin());
        std::copy(strides.begin(), strides.end(), shape_data.begin() + rank);
        auto shape_buffer = context_.getSharedBuffer<int>(shape_data.data(), shape_data.size(), queue_);

        kernel = program_.getKernel("X" + name + "Strided");
        kernel.setArguments(static_cast<int>(n), static_cast<int>(rank), shape_buffer,
                            x_buffer, static_cast<int>(x_offset),
                            y_buffer, static_cast<int>(y_offset));
    }

    auto global = std::vector<size_t>{m};
    auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xreduce<half>;
template class Xreduce<float>;
template class Xreduce<double>;

}} // namespace gpgpu::dnn
