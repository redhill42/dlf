#pragma once

#include "routine.hpp"

namespace gpgpu { namespace dnn {

template <typename T>
class Xcopy: public blas::Routine {
 public:
  Xcopy(const Queue &queue, Event* event, const std::string &name = "DNN_COPY");

  void DoCopy(const size_t x_size, const Buffer<T> &x_buffer,
              const size_t y_size, Buffer<T> &y_buffer);

  void DoCopyStrided(
              const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
              const std::vector<size_t>& stride, const std::vector<size_t>& shape);
};

}} // namespace gpgpu::dnn
