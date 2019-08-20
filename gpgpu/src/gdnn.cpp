#include <cassert>
#include "gdnn.h"
#include "routines/routines.hpp"
#include "cudnn.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
void copy(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
          const size_t y_size, Buffer<T>& y_buffer, const size_t y_offset,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopy(x_size, x_buffer, x_offset, y_size, y_buffer, y_offset);
}

template void PUBLIC_API copy<int16_t>(const size_t, const Buffer<int16_t>&, const size_t,
                                       const size_t, Buffer<int16_t>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const Buffer<int32_t>&, const size_t,
                                       const size_t, Buffer<int32_t>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const Buffer<int64_t>&, const size_t,
                                       const size_t, Buffer<int64_t>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const Buffer<float>&, const size_t,
                                       const size_t, Buffer<float>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const Buffer<double>&, const size_t,
                                       const size_t, Buffer<double>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const Buffer<float2>&, const size_t,
                                       const size_t, Buffer<float2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const Buffer<double2>&, const size_t,
                                       const size_t, Buffer<double2>&, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const Buffer<half>&, const size_t,
                                       const size_t, Buffer<half>&, const size_t,
                                       const Queue&, Event*);

template <typename T>
void copy(const size_t n, const std::vector<size_t>& dims,
          const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopyStrided(
        n, dims,
        x_buffer, x_offset, x_stride,
        y_buffer, y_offset, y_stride);
}

template void PUBLIC_API copy<int16_t>(const size_t, const std::vector<size_t>&,
                                       const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
                                       Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const std::vector<size_t>&,
                                       const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
                                       Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const std::vector<size_t>&,
                                       const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
                                       Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const std::vector<size_t>&,
                                       const Buffer<half>&, const size_t, const std::vector<size_t>&,
                                       Buffer<half>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const std::vector<size_t>&,
                                       const Buffer<float>&, const size_t, const std::vector<size_t>&,
                                       Buffer<float>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const std::vector<size_t>&,
                                       const Buffer<double>&, const size_t, const std::vector<size_t>&,
                                       Buffer<double>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const std::vector<size_t>&,
                                       const Buffer<float2>&, const size_t, const std::vector<size_t>&,
                                       Buffer<float2>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const std::vector<size_t>&,
                                       const Buffer<double2>&, const size_t, const std::vector<size_t>&,
                                       Buffer<double2>&, const size_t, const std::vector<size_t>&,
                                       const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset,
               Buffer<T>& y_buffer, const size_t y_offset,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform<T>(queue, event);
    routine.DoTransform(name, n, x_buffer, x_offset, y_buffer, y_offset);
}

template void PUBLIC_API transform<int16_t>(const std::string&, const size_t,
                                            const Buffer<int16_t>&, const size_t,
                                            Buffer<int16_t>&, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&, const size_t,
                                            const Buffer<int32_t>&, const size_t,
                                            Buffer<int32_t>&, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&, const size_t,
                                            const Buffer<int64_t>&, const size_t,
                                            Buffer<int64_t>&, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&, const size_t,
                                            const Buffer<half>&, const size_t,
                                            Buffer<half>&, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&, const size_t,
                                            const Buffer<float>&, const size_t,
                                            Buffer<float>&, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&, const size_t,
                                            const Buffer<double>&, const size_t,
                                            Buffer<double>&, const size_t,
                                            const Queue&, Event*);

template <typename T>
void transform(const std::string& name, size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
               Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform<T>(queue, event);
    routine.DoTransform(name, n, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
}

template void PUBLIC_API transform<int16_t>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<half>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    Buffer<half>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<float>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    Buffer<float>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<double>(
    const std::string&, const size_t, const std::vector<size_t>&,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    Buffer<double>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);

template <typename T>
void transform(const std::string& name,const T alpha, const T beta, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset,
               Buffer<T>& y_buffer, const size_t y_offset,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_p<T>(queue, event);
    routine.DoTransform(name, alpha, beta, n, x_buffer, x_offset, y_buffer, y_offset);
}

template void PUBLIC_API transform<half>  (const std::string&, const half, const half, const size_t,
                                           const Buffer<half>&, const size_t,
                                           Buffer<half>&, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<float> (const std::string&, const float, const float, const size_t,
                                           const Buffer<float>&, const size_t,
                                           Buffer<float>&, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<double>(const std::string&, const double, const double, const size_t,
                                           const Buffer<double>&, const size_t,
                                           Buffer<double>&, const size_t,
                                           const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const T alpha, const T beta,
               size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
               Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_p<T>(queue, event);
    routine.DoTransform(name, alpha, beta, n, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
}

template void PUBLIC_API transform<half>(
    const std::string&, const half, const half, const size_t, const std::vector<size_t>&,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    Buffer<half>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<float>(
    const std::string&, const float, const float, const size_t, const std::vector<size_t>&,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    Buffer<float>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<double>(
    const std::string&, const double, const double, const size_t, const std::vector<size_t>&,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    Buffer<double>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);

template <typename T, typename R>
void transform(const std::string& name,
               const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
               const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset,
               Buffer<R>& z_buffer, const size_t z_offset,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_b<T,R>(queue, event);
    routine.DoTransform(name, x_size, x_buffer, x_offset, y_size, y_buffer, y_offset, z_buffer, z_offset);
}

template void PUBLIC_API transform<int16_t, int16_t>(
    const std::string&,
    const size_t, const Buffer<int16_t>&, const size_t,
    const size_t, const Buffer<int16_t>&, const size_t,
    Buffer<int16_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, int32_t>(
    const std::string&,
    const size_t, const Buffer<int32_t>&, const size_t,
    const size_t, const Buffer<int32_t>&, const size_t,
    Buffer<int32_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, int64_t>(
    const std::string&,
    const size_t, const Buffer<int64_t>&, const size_t,
    const size_t, const Buffer<int64_t>&, const size_t,
    Buffer<int64_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<half, half>(
    const std::string&,
    const size_t, const Buffer<half>&, const size_t,
    const size_t, const Buffer<half>&, const size_t,
    Buffer<half>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float, float>(
    const std::string&,
    const size_t, const Buffer<float>&, const size_t,
    const size_t, const Buffer<float>&, const size_t,
    Buffer<float>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double, double>(
    const std::string&,
    const size_t, const Buffer<double>&, const size_t,
    const size_t, const Buffer<double>&, const size_t,
    Buffer<double>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float2, float2>(
    const std::string&,
    const size_t, const Buffer<float2>&, const size_t,
    const size_t, const Buffer<float2>&, const size_t,
    Buffer<float2>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double2, double2>(
    const std::string&,
    const size_t, const Buffer<double2>&, const size_t,
    const size_t, const Buffer<double2>&, const size_t,
    Buffer<double2>&, const size_t,
    const Queue&, Event*);

template void PUBLIC_API transform<int16_t, bool>(
    const std::string&,
    const size_t, const Buffer<int16_t>&, const size_t,
    const size_t, const Buffer<int16_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, bool>(
    const std::string&,
    const size_t, const Buffer<int32_t>&, const size_t,
    const size_t, const Buffer<int32_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, bool>(
    const std::string&,
    const size_t, const Buffer<int64_t>&, const size_t,
    const size_t, const Buffer<int64_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<half, bool>(
    const std::string&,
    const size_t, const Buffer<half>&, const size_t,
    const size_t, const Buffer<half>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float, bool>(
    const std::string&,
    const size_t, const Buffer<float>&, const size_t,
    const size_t, const Buffer<float>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double, bool>(
    const std::string&,
    const size_t, const Buffer<double>&, const size_t,
    const size_t, const Buffer<double>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);

template <typename T, typename R>
void transform(const std::string& name, const size_t n, const std::vector<size_t>& dims,
               const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
               const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
               Buffer<R>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_stride,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_b<T,R>(queue, event);
    routine.DoTransform(name, n, dims,
        x_buffer, x_offset, x_stride,
        y_buffer, y_offset, y_stride,
        z_buffer, z_offset, z_stride);
}

template void PUBLIC_API transform<int16_t, int16_t>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, int32_t>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, int64_t>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<half, half>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    Buffer<half>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<float, float>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    Buffer<float>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<double, double>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    Buffer<double>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<float2, float2>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<float2>&, const size_t, const std::vector<size_t>&,
    const Buffer<float2>&, const size_t, const std::vector<size_t>&,
    Buffer<float2>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<double2, double2>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<double2>&, const size_t, const std::vector<size_t>&,
    const Buffer<double2>&, const size_t, const std::vector<size_t>&,
    Buffer<double2>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);

template void PUBLIC_API transform<int16_t, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int16_t>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int32_t>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    const Buffer<int64_t>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<half, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    const Buffer<half>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<float, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    const Buffer<float>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);
template void PUBLIC_API transform<double, bool>(
    const std::string&, const size_t, const std::vector<size_t>& dims,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    const Buffer<double>&, const size_t, const std::vector<size_t>&,
    Buffer<bool>&, const size_t, const std::vector<size_t>&,
    const Queue&, Event*);

template <typename T, typename R>
void transform(const std::string& name,
               const size_t m, const size_t n, const size_t channels,
               const Buffer<T>& x_buffer, const size_t x_offset,
               const Buffer<T>& y_buffer, const size_t y_offset,
               Buffer<R>& z_buffer, const size_t z_offset,
               const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (name == "add_v" && IsCUDA(queue.context().device()) &&
        x_buffer.handle() == z_buffer.handle() &&
        x_offset == 0 && y_offset == 0 && z_offset == 0) {
        auto y_desc = TensorDescriptor<T>(1, channels, 1, 1);
        auto z_desc = TensorDescriptor<T>(m/channels, channels, 1, n);
        T alpha = 1, beta = 1;
        checkCUDNN(cudnnAddTensor(
            cudnn_handle(queue),
            &alpha, y_desc, cu::cuBuffer::unwrap(y_buffer),
            &beta,  z_desc, cu::cuBuffer::unwrap(z_buffer)));
        return;
    }
#endif

    auto routine = Xtransform_c<T,R>(queue, event);
    routine.DoTransform(name, m, n, channels, x_buffer, x_offset, y_buffer, y_offset, z_buffer, z_offset);
}

template void PUBLIC_API transform<int16_t, int16_t>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int16_t>&, const size_t,
    const Buffer<int16_t>&, const size_t,
    Buffer<int16_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, int32_t>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int32_t>&, const size_t,
    const Buffer<int32_t>&, const size_t,
    Buffer<int32_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, int64_t>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int64_t>&, const size_t,
    const Buffer<int64_t>&, const size_t,
    Buffer<int64_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<half, half>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<half>&, const size_t,
    const Buffer<half>&, const size_t,
    Buffer<half>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float, float>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<float>&, const size_t,
    const Buffer<float>&, const size_t,
    Buffer<float>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double, double>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<double>&, const size_t,
    const Buffer<double>&, const size_t,
    Buffer<double>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float2, float2>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<float2>&, const size_t,
    const Buffer<float2>&, const size_t,
    Buffer<float2>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double2, double2>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<double2>&, const size_t,
    const Buffer<double2>&, const size_t,
    Buffer<double2>&, const size_t,
    const Queue&, Event*);

template void PUBLIC_API transform<int16_t, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int16_t>&, const size_t,
    const Buffer<int16_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int32_t, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int32_t>&, const size_t,
    const Buffer<int32_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<int64_t, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<int64_t>&, const size_t,
    const Buffer<int64_t>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<half, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<half>&, const size_t,
    const Buffer<half>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<float, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<float>&, const size_t,
    const Buffer<float>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API transform<double, bool>(
    const std::string&,
    const size_t, const size_t, const size_t,
    const Buffer<double>&, const size_t,
    const Buffer<double>&, const size_t,
    Buffer<bool>&, const size_t,
    const Queue&, Event*);

template <typename T>
void reduce(const std::string& name, const size_t m, const size_t n,
            const std::vector<size_t>& dims, const std::vector<size_t>& strides,
            const Buffer<T>& x_buffer, const size_t x_offset,
            Buffer<T>& y_buffer, const size_t y_offset,
            const Queue& queue, Event* event)
{
    auto routine = Xreduce<T>(queue, event);
    routine.DoReduce(name, m, n, dims, strides, x_buffer, x_offset, y_buffer, y_offset);
}

template void PUBLIC_API reduce<half>  (const std::string&, const size_t, const size_t,
                                        const std::vector<size_t>&, const std::vector<size_t>&,
                                        const Buffer<half>&, const size_t,
                                        Buffer<half>&, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API reduce<float> (const std::string&, const size_t, const size_t,
                                        const std::vector<size_t>&, const std::vector<size_t>&,
                                        const Buffer<float>&, const size_t,
                                        Buffer<float>&, const size_t,
                                        const Queue&, Event*);
template void PUBLIC_API reduce<double>(const std::string&, const size_t, const size_t,
                                        const std::vector<size_t>&, const std::vector<size_t>&,
                                        const Buffer<double>&, const size_t,
                                        Buffer<double>&, const size_t,
                                        const Queue&, Event*);

template <typename T>
void batch_norm(const std::vector<size_t>& dims,
                const Buffer<T>& x_buffer,
                      Buffer<T>& y_buffer,
                const Buffer<T>& scale_buffer,
                const Buffer<T>& bias_buffer,
                const Buffer<T>& mean_buffer,
                const Buffer<T>& var_buffer,
                const T epsilon,
                const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device()) && (dims.size() == 4 || dims.size() == 5)) {
        TensorDescriptor<T> xy_desc(dims);
        TensorDescriptor<T> sbmv_desc;
        checkCUDNN(cudnnDeriveBNTensorDescriptor(sbmv_desc, xy_desc,
            cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL));

        const T alpha = 1, beta = 0;
        checkCUDNN(cudnnBatchNormalizationForwardInference(
            cudnn_handle(queue), cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            xy_desc, cu::cuBuffer::unwrap(x_buffer),
            xy_desc, cu::cuBuffer::unwrap(y_buffer),
            sbmv_desc,
            cu::cuBuffer::unwrap(scale_buffer),
            cu::cuBuffer::unwrap(bias_buffer),
            cu::cuBuffer::unwrap(mean_buffer),
            cu::cuBuffer::unwrap(var_buffer),
            0.00001)); // FIXME
        return;
    }
#endif

    auto batches = dims[0];
    auto channels = dims[1];
    auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

    auto routine = Xnormalization<T>(queue, event);
    routine.DoBatchNorm(batches, channels, spatial, x_buffer, y_buffer,
                        scale_buffer, bias_buffer, mean_buffer, var_buffer, epsilon);
}

template void PUBLIC_API batch_norm<half>  (const std::vector<size_t>&,
                                            const Buffer<half>&, Buffer<half>&,
                                            const Buffer<half>&, const Buffer<half>&,
                                            const Buffer<half>&, const Buffer<half>&,
                                            const half, const Queue&, Event*);
template void PUBLIC_API batch_norm<float> (const std::vector<size_t>&,
                                            const Buffer<float>&, Buffer<float>&,
                                            const Buffer<float>&, const Buffer<float>&,
                                            const Buffer<float>&, const Buffer<float>&,
                                            const float, const Queue&, Event*);
template void PUBLIC_API batch_norm<double>(const std::vector<size_t>&,
                                            const Buffer<double>&, Buffer<double>&,
                                            const Buffer<double>&, const Buffer<double>&,
                                            const Buffer<double>&, const Buffer<double>&,
                                            const double, const Queue&, Event*);

template <typename T>
void lrn(const std::vector<size_t>& dims, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
         const int nsize, const T alpha, const T beta, const T bias,
         const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device()) && (dims.size() == 4 || dims.size() == 5)) {
        TensorDescriptor<T> xy_desc(dims);
        LRNDescriptor desc(nsize, alpha, beta, bias);

        T blend_alpha = 1, blend_beta = 0;
        checkCUDNN(cudnnLRNCrossChannelForward(
            cudnn_handle(queue), desc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &blend_alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &blend_beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
        return;
    }
#endif

    auto batches = dims[0];
    auto channels = dims[1];
    auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

    auto routine = Xnormalization<T>(queue, event);
    routine.DoLRN(batches, channels, spatial, x_buffer, y_buffer, nsize, alpha, beta, bias);
}

template void PUBLIC_API lrn<half>  (const std::vector<size_t>&,
                                     const Buffer<half>&, Buffer<half>&,
                                     const int, const half, const half, const half,
                                     const Queue&, Event*);
template void PUBLIC_API lrn<float> (const std::vector<size_t>&,
                                     const Buffer<float>&, Buffer<float>&,
                                     const int, const float, const float, const float,
                                     const Queue&, Event*);
template void PUBLIC_API lrn<double>(const std::vector<size_t>&,
                                     const Buffer<double>&, Buffer<double>&,
                                     const int, const double, const double, const double,
                                     const Queue&, Event*);

namespace {

#if HAS_CUDA

#define CUDNN_CONV2D_PROLOGUE                                               \
    auto cudnn = cudnn_handle(queue);                                       \
                                                                            \
    auto conv_desc = ConvolutionDescriptor<T>(                              \
        pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w);     \
    auto x_desc = TensorDescriptor<T>(                                      \
        batches, channels, height, width);                                  \
    auto w_desc = FilterDescriptor<T>(                                      \
        num_kernels, channels/group, kernel_h, kernel_w);                   \
    auto y_desc = TensorDescriptor<T>(                                      \
        batches, num_kernels, output_h, output_w);                          \
                                                                            \
    checkCUDNN(cudnnSetConvolutionGroupCount(conv_desc, group));            \
                                                                            \
    cudnnConvolutionFwdAlgo_t conv_algo;                                    \
    if (group != 1) {                                                       \
        conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;               \
    } else {                                                                \
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(                     \
            cudnn, x_desc, w_desc, conv_desc, y_desc,                       \
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,                           \
            /*memoryLimitInBytes=*/0,                                       \
            &conv_algo));                                                   \
    }                                                                       \
                                                                            \
    size_t workspace_size = 0;                                              \
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(                     \
        cudnn, x_desc, w_desc, conv_desc, y_desc, conv_algo, &workspace_size));

template <typename T>
void cudnnConv(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_top, const size_t pad_left,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Buffer<T>& im_buffer, const Buffer<T>& kernel_buffer,
    Buffer<T>& result_buffer, Buffer<T>* work, const Queue& queue)
{
    CUDNN_CONV2D_PROLOGUE

    Buffer<T> temp_buffer;
    if (workspace_size > 0) {
        if (work == nullptr) {
            temp_buffer = queue.context().createBuffer<T>(workspace_size / sizeof(T));
            work = &temp_buffer;
        } else {
            assert(work->size() * sizeof(T) >= workspace_size);
        }
    }

    const T alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(
        cudnn, &alpha,
        x_desc, cu::cuBuffer::unwrap(im_buffer),
        w_desc, cu::cuBuffer::unwrap(kernel_buffer),
        conv_desc, conv_algo,
        workspace_size == 0 ? nullptr : cu::cuBuffer::unwrap(*work),
        workspace_size,
        &beta,
        y_desc, cu::cuBuffer::unwrap((result_buffer))));
}

#endif //!HAS_CUDA

template <typename T>
void convWithIm2Col(
    const bool is_1x1_kernel,
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_h, const size_t pad_w,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Buffer<T>& im_buffer, const Buffer<T>& kernel_buffer,
    Buffer<T>& result_buffer, Buffer<T>* work_buffer,
    const Queue& queue, Event* event)
{
    auto m = num_kernels / group;
    auto k = channels * kernel_h * kernel_w / group;
    auto n = output_h * output_w;

    const Buffer<T>* x_buffer;
    Buffer<T> temp_buffer;

    if (is_1x1_kernel) {
        assert(height == output_h && width == output_w);
        x_buffer = &im_buffer;
    } else {
        if (work_buffer == nullptr) {
            temp_buffer = queue.context().createBuffer<T>(k * n * batches * group);
            work_buffer = &temp_buffer;
        } else {
            assert(work_buffer->size() >= k * n * batches * group);
        }

        im2col(KernelMode::CrossCorrelation,
               batches*group, channels/group,
               height, width, output_h, output_w,
               kernel_h, kernel_w, pad_h, pad_w,
               stride_h, stride_w, dilation_h, dilation_w,
               im_buffer, 0, *work_buffer, 0,
               queue, event);

        x_buffer = work_buffer;
    }

    if (group == 1) {
        gemmStridedBatched(gpgpu::blas::Layout::RowMajor,
                           gpgpu::blas::Transpose::NoTrans,
                           gpgpu::blas::Transpose::NoTrans,
                           m, n, k, T{1},
                           kernel_buffer, 0, k, 0,
                           *x_buffer, 0, n, k*n,
                           T{0}, result_buffer, 0, n, m*n,
                           batches,
                           queue, event);
    } else {
        for (size_t b = 0; b < batches; b++) {
            gemmStridedBatched(gpgpu::blas::Layout::RowMajor,
                               gpgpu::blas::Transpose::NoTrans,
                               gpgpu::blas::Transpose::NoTrans,
                               m, n, k, T{1},
                               kernel_buffer, 0, k, m*k,
                               *x_buffer, b*group*k*n, n, k*n,
                               T{0}, result_buffer, b*group*m*n, n, m*n,
                               group,
                               queue, event);
        }
    }
}

} // anonymous namespace

template <typename T>
void conv2d(const size_t batches, const size_t channels,
            const size_t height, const size_t width,
            const size_t output_h, const size_t output_w,
            const size_t num_kernels, const size_t group,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_top, const size_t pad_left,
            const size_t pad_bottom, const size_t pad_right,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const Buffer<T>& im_buffer, const Buffer<T>& kernel_buffer,
            Buffer<T>& result_buffer, Buffer<T>* work_buffer,
            const Queue& queue, Event* event)
{
    bool is_1x1_kernel =
        kernel_h == 1 && kernel_w == 1 &&
        stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 &&
        pad_top + pad_bottom == 0  && pad_left + pad_right == 0;

#if HAS_CUDA
    bool is_cuda_applicable =
        IsCUDA(queue.context().device()) &&
        pad_top == pad_bottom &&
        pad_left == pad_right;

    if (is_cuda_applicable) {
        cudnnConv(
            batches, channels, height, width, output_h, output_w,
            num_kernels, group, kernel_h, kernel_w,
            pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w,
            im_buffer, kernel_buffer, result_buffer, work_buffer, queue);
        return;
    }
#endif

    if (is_1x1_kernel || group != 1) {
        convWithIm2Col(
            is_1x1_kernel,
            batches, channels, height, width, output_h, output_w,
            num_kernels, group, kernel_h, kernel_w,
            pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w,
            im_buffer, kernel_buffer, result_buffer, work_buffer,
            queue, event);
    } else {
        convgemm(KernelMode::CrossCorrelation,
                 batches, channels,
                 height, width, output_h, output_w,
                 num_kernels, kernel_h, kernel_w,
                 pad_top, pad_left, stride_h, stride_w,
                 dilation_h, dilation_w,
                 im_buffer, 0, kernel_buffer, 0, result_buffer, 0,
                 work_buffer, queue, event);
    }
}

template void PUBLIC_API conv2d<float> (const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<float>&, const Buffer<float>&,
                                        Buffer<float>&, Buffer<float>*,
                                        const Queue&, Event*);
template void PUBLIC_API conv2d<double>(const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<double>&, const Buffer<double>&,
                                        Buffer<double>&, Buffer<double>*,
                                        const Queue&, Event*);
template void PUBLIC_API conv2d<half>  (const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<half>&, const Buffer<half>&,
                                        Buffer<half>&, Buffer<half>*,
                                        const Queue&, Event*);

template <typename T>
size_t conv2dWorkspaceSize(const size_t batches, const size_t channels,
                           const size_t height, const size_t width,
                           const size_t output_h, const size_t output_w,
                           const size_t num_kernels, const size_t group,
                           const size_t kernel_h, const size_t kernel_w,
                           const size_t pad_top, const size_t pad_left,
                           const size_t pad_bottom, const size_t pad_right,
                           const size_t stride_h, const size_t stride_w,
                           const size_t dilation_h, const size_t dilation_w,
                           const Queue& queue)
{
    bool is_1x1_kernel =
        kernel_h == 1 && kernel_w == 1 &&
        stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 &&
        pad_top + pad_bottom == 0 && pad_left + pad_right == 0;

#if HAS_CUDA
    bool is_cuda_applicable =
        IsCUDA(queue.context().device()) &&
        pad_top == pad_bottom &&
        pad_left == pad_right;

    if (is_cuda_applicable) {
        CUDNN_CONV2D_PROLOGUE
        return workspace_size / sizeof(T);
    }
#else
    (void)height, (void)width, (void)num_kernels, (void)group;
    (void)pad_bottom, (void)pad_right;
#endif

    if (is_1x1_kernel)
        return 0;

    if (group != 1) {
        auto k = channels * kernel_h * kernel_w / group;
        auto n = output_h * output_w;
        return k * n * batches * group;
    }

    return convgemmTempBufferSize<T>(batches, channels, output_h, output_w, kernel_h, kernel_w, queue);
}

template size_t PUBLIC_API conv2dWorkspaceSize<float>(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_top, const size_t pad_left,
    const size_t pad_bottom, const size_t pad_right,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Queue& queue);
template size_t PUBLIC_API conv2dWorkspaceSize<double>(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_top, const size_t pad_left,
    const size_t pad_bottom, const size_t pad_right,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Queue& queue);
template size_t PUBLIC_API conv2dWorkspaceSize<half>(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_top, const size_t pad_left,
    const size_t pad_bottom, const size_t pad_right,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Queue& queue);

template <typename T>
void maxpool(const size_t batches, const size_t channels,
             const size_t height, const size_t width,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_top, const size_t pad_left,
             const size_t pad_bottom, const size_t pad_right,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device()) && (pad_top == pad_bottom && pad_left == pad_right)
                                         && (dilation_h == 1 && dilation_w == 1)) {
        PoolingDescriptor desc(cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            kernel_h, kernel_w, pad_top, pad_left, stride_h, stride_w);
        TensorDescriptor<T> x_desc(batches, channels, height, width);
        TensorDescriptor<T> y_desc(batches, channels, output_h, output_w);

        T alpha = 1, beta = 0;
        cudnnPoolingForward(
            cudnn_handle(queue), desc,
            &alpha, x_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta, y_desc, cu::cuBuffer::unwrap(y_buffer));
        return;
    }
#else
    (void)pad_bottom, (void)pad_right;
#endif

    auto routine = Xpool<T>(queue, event);
    routine.DoMaxPool(batches, channels,
                      height, width,
                      output_h, output_w,
                      kernel_h, kernel_w,
                      pad_top, pad_left,
                      stride_h, stride_w,
                      dilation_h, dilation_w,
                      x_buffer, 0, y_buffer, 0);
}

template void PUBLIC_API maxpool<half>  (const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const Buffer<half>&, Buffer<half>&,
                                         const Queue&, Event*);
template void PUBLIC_API maxpool<float> (const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const Buffer<float>&, Buffer<float>&,
                                         const Queue&, Event*);
template void PUBLIC_API maxpool<double>(const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const Buffer<double>&, Buffer<double>&,
                                         const Queue&, Event*);

template <typename T>
void avgpool(const size_t batches, const size_t channels,
             const size_t height, const size_t width,
             const size_t output_h, const size_t output_w,
             const size_t kernel_h, const size_t kernel_w,
             const size_t pad_top, const size_t pad_left,
             const size_t pad_bottom, const size_t pad_right,
             const size_t stride_h, const size_t stride_w,
             const size_t dilation_h, const size_t dilation_w,
             const bool count_include_pad,
             const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device()) && (pad_top == pad_bottom && pad_left == pad_right)
                                         && (dilation_h == 1 && dilation_w == 1)) {
        PoolingDescriptor desc(
            count_include_pad ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                              : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
            kernel_h, kernel_w, pad_top, pad_left, stride_h, stride_w);
        TensorDescriptor<T> x_desc(batches, channels, height, width);
        TensorDescriptor<T> y_desc(batches, channels, output_h, output_w);

        T alpha = 1, beta = 0;
        cudnnPoolingForward(
            cudnn_handle(queue), desc,
            &alpha, x_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta, y_desc, cu::cuBuffer::unwrap(y_buffer));
        return;
    }
#else
    (void)pad_bottom, (void)pad_right;
#endif

    auto routine = Xpool<T>(queue, event);
    routine.DoAvgPool(batches, channels,
                      height, width,
                      output_h, output_w,
                      kernel_h, kernel_w,
                      pad_top, pad_left,
                      stride_h, stride_w,
                      dilation_h, dilation_w,
                      count_include_pad,
                      x_buffer, 0, y_buffer, 0);
}

template void PUBLIC_API avgpool<half>  (const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const bool,
                                         const Buffer<half>&, Buffer<half>&,
                                         const Queue&, Event*);
template void PUBLIC_API avgpool<float> (const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const bool,
                                         const Buffer<float>&, Buffer<float>&,
                                         const Queue&, Event*);
template void PUBLIC_API avgpool<double>(const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const size_t, const size_t, const size_t, const size_t,
                                         const bool,
                                         const Buffer<double>&, Buffer<double>&,
                                         const Queue&, Event*);

template <typename T>
void lppool(const size_t batches, const size_t channels,
            const size_t height, const size_t width,
            const size_t output_h, const size_t output_w,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_top, const size_t pad_left,
            const size_t /*pad_bottom*/, const size_t /*pad_right*/,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const int p,
            const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
            const Queue& queue, Event* event)
{
    auto routine = Xpool<T>(queue, event);
    routine.DoLpPool(batches, channels,
                     height, width,
                     output_h, output_w,
                     kernel_h, kernel_w,
                     pad_top, pad_left,
                     stride_h, stride_w,
                     dilation_h, dilation_w,
                     p,
                     x_buffer, 0, y_buffer, 0);
}

template void PUBLIC_API lppool<half>  (const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const int,
                                        const Buffer<half>&, Buffer<half>&,
                                        const Queue&, Event*);
template void PUBLIC_API lppool<float> (const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const int,
                                        const Buffer<float>&, Buffer<float>&,
                                        const Queue&, Event*);
template void PUBLIC_API lppool<double>(const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const int,
                                        const Buffer<double>&, Buffer<double>&,
                                        const Queue&, Event*);

template <typename T>
void softmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device())) {
        TensorDescriptor<T> xy_desc(m, 1, 1, n);
        T alpha = 1, beta = 0;
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle(queue),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
        return;
    }
#endif

    auto routine = Xsoftmax<T>(queue, event);
    routine.DoSoftmax(m, n, x_buffer, y_buffer);
}

template void PUBLIC_API softmax<half>  (const size_t, const size_t,
                                         const Buffer<half>&, Buffer<half>&,
                                         const Queue& queue, Event*);
template void PUBLIC_API softmax<float> (const size_t, const size_t,
                                         const Buffer<float>&, Buffer<float>&,
                                         const Queue& queue, Event*);
template void PUBLIC_API softmax<double>(const size_t, const size_t,
                                         const Buffer<double>&, Buffer<double>&,
                                         const Queue& queue, Event*);

template <typename T>
void logsoftmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue, Event* event)
{
#if HAS_CUDA
    if (IsCUDA(queue.context().device())) {
        TensorDescriptor<T> xy_desc(m, 1, 1, n);
        T alpha = 1, beta = 0;
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle(queue),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
        return;
    }
#endif

    auto routine = Xsoftmax<T>(queue, event);
    routine.DoLogSoftmax(m, n, x_buffer, y_buffer);
}

template void PUBLIC_API logsoftmax<half>  (const size_t, const size_t,
                                            const Buffer<half>&, Buffer<half>&,
                                            const Queue& queue, Event*);
template void PUBLIC_API logsoftmax<float> (const size_t, const size_t,
                                            const Buffer<float>&, Buffer<float>&,
                                            const Queue& queue, Event*);
template void PUBLIC_API logsoftmax<double>(const size_t, const size_t,
                                            const Buffer<double>&, Buffer<double>&,
                                            const Queue& queue, Event*);

template <typename T>
void hardmax(const size_t m, const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
             const Queue& queue, Event* event)
{
    auto routine = Xsoftmax<T>(queue, event);
    routine.DoHardmax(m, n, x_buffer, y_buffer);
}

template void PUBLIC_API hardmax<half>  (const size_t, const size_t,
                                         const Buffer<half>&, Buffer<half>&,
                                         const Queue& queue, Event*);
template void PUBLIC_API hardmax<float> (const size_t, const size_t,
                                         const Buffer<float>&, Buffer<float>&,
                                         const Queue& queue, Event*);
template void PUBLIC_API hardmax<double>(const size_t, const size_t,
                                         const Buffer<double>&, Buffer<double>&,
                                         const Queue& queue, Event*);

template <typename T>
void argmax(const size_t m, const size_t k, const size_t n,
            const Buffer<T>& x_buffer, Buffer<int>& y_buffer,
            const Queue& queue, Event* event)
{
    auto routine = Xargmax<T>(queue, event);
    routine.DoArgMax(m, k, n, x_buffer, y_buffer);
}

template void PUBLIC_API argmax<int16_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int16_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmax<int32_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int32_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmax<int64_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int64_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmax<half>   (const size_t, const size_t, const size_t,
                                         const Buffer<half>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmax<float>  (const size_t, const size_t, const size_t,
                                         const Buffer<float>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmax<double> (const size_t, const size_t, const size_t,
                                         const Buffer<double>&, Buffer<int>&,
                                         const Queue&, Event*);

template <typename T>
void argmin(const size_t m, const size_t k, const size_t n,
            const Buffer<T>& x_buffer, Buffer<int>& y_buffer,
            const Queue& queue, Event* event)
{
    auto routine = Xargmax<T>(queue, event);
    routine.DoArgMin(m, k, n, x_buffer, y_buffer);
}

template void PUBLIC_API argmin<int16_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int16_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmin<int32_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int32_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmin<int64_t>(const size_t, const size_t, const size_t,
                                         const Buffer<int64_t>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmin<half>   (const size_t, const size_t, const size_t,
                                         const Buffer<half>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmin<float>  (const size_t, const size_t, const size_t,
                                         const Buffer<float>&, Buffer<int>&,
                                         const Queue&, Event*);
template void PUBLIC_API argmin<double> (const size_t, const size_t, const size_t,
                                         const Buffer<double>&, Buffer<int>&,
                                         const Queue&, Event*);

template <typename T>
void where(const size_t n, const size_t rank,
           const Buffer<bool>& c_buffer, const size_t c_offset,
           const std::vector<size_t>& c_dim, const std::vector<size_t>& c_stride,
           const Buffer<T>& x_buffer, const size_t x_offset,
           const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
           const Buffer<T>& y_buffer, const size_t y_offset,
           const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
           Buffer<T>& z_buffer, const size_t z_offset,
           const Queue& queue, Event* event)
{
    auto routine = Xwhere<T>(queue, event);
    routine.DoWhere(
        n, rank,
        c_buffer, c_offset, c_dim, c_stride,
        x_buffer, x_offset, x_dim, x_stride,
        y_buffer, y_offset, y_dim, y_stride,
        z_buffer, z_offset);
}

template void PUBLIC_API where<int16_t>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int16_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int16_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<int16_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API where<int32_t>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int32_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int32_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<int32_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API where<int64_t>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int64_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<int64_t>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<int64_t>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API where<half>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<half>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<half>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<half>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API where<float>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<float>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<float>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<float>&, const size_t,
    const Queue&, Event*);
template void PUBLIC_API where<double>(
    const size_t, const size_t,
    const Buffer<bool>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<double>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    const Buffer<double>&, const size_t,
    const std::vector<size_t>&, const std::vector<size_t>&,
    Buffer<double>&, const size_t,
    const Queue&, Event*);

}} // namespace gpgpu::dnn
