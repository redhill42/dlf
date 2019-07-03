#include "gdnn.h"
#include "routines/routines.hpp"

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
void copy(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          const size_t y_size, Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopy(x_size, x_buffer, x_offset, x_inc,
                   y_size, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API copy<int16_t>(const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                       const size_t, Buffer<int16_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                       const size_t, Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                       const size_t, Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const Buffer<float>&, const size_t, const size_t,
                                       const size_t, Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const Buffer<double>&, const size_t, const size_t,
                                       const size_t, Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const Buffer<float2>&, const size_t, const size_t,
                                       const size_t, Buffer<float2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const Buffer<double2>&, const size_t, const size_t,
                                       const size_t, Buffer<double2>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const Buffer<half>&, const size_t, const size_t,
                                       const size_t, Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);

template <typename T>
void copy(const size_t n, const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
          const std::vector<size_t>& stride, const std::vector<size_t>& shape,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopyStrided(n, x_buffer, y_buffer, stride, shape);
}

template void PUBLIC_API copy<int16_t>(const size_t, const Buffer<int16_t>&, Buffer<int16_t>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const Buffer<int32_t>&, Buffer<int32_t>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const Buffer<int64_t>&, Buffer<int64_t>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const Buffer<half>&, Buffer<half>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const Buffer<float>&, Buffer<float>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const Buffer<double>&, Buffer<double>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const Buffer<float2>&, Buffer<float2>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const Buffer<double2>&, Buffer<double2>&,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform<T>(queue, event);
    routine.DoTransform(name, n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API transform<int16_t>(const std::string&, const size_t,
                                            const Buffer<int16_t>&, const size_t, const size_t,
                                            Buffer<int16_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&, const size_t,
                                            const Buffer<int32_t>&, const size_t, const size_t,
                                            Buffer<int32_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&, const size_t,
                                            const Buffer<int64_t>&, const size_t, const size_t,
                                            Buffer<int64_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&, const size_t,
                                            const Buffer<half>&, const size_t, const size_t,
                                            Buffer<half>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&, const size_t,
                                            const Buffer<float>&, const size_t, const size_t,
                                            Buffer<float>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&, const size_t,
                                            const Buffer<double>&, const size_t, const size_t,
                                            Buffer<double>&, const size_t, const size_t,
                                            const Queue&, Event*);

template <typename T>
void transform(const std::string& name,
                const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc,
                const Queue& queue, Event* event)
{
    auto routine = Xtransform_b<T>(queue, event);
    routine.DoTransform(name,
        x_size, x_buffer, x_offset, x_inc,
        y_size, y_buffer, y_offset, y_inc,
        z_buffer, z_offset, z_inc);
}

template void PUBLIC_API transform<int16_t>(const std::string&,
                                            const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                            const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                            Buffer<int16_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&,
                                            const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                            const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                            Buffer<int32_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&,
                                            const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                            const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                            Buffer<int64_t>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&,
                                            const size_t, const Buffer<half>&, const size_t, const size_t,
                                            const size_t, const Buffer<half>&, const size_t, const size_t,
                                            Buffer<half>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&,
                                            const size_t, const Buffer<float>&, const size_t, const size_t,
                                            const size_t, const Buffer<float>&, const size_t, const size_t,
                                            Buffer<float>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&,
                                            const size_t, const Buffer<double>&, const size_t, const size_t,
                                            const size_t, const Buffer<double>&, const size_t, const size_t,
                                            Buffer<double>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float2> (const std::string&,
                                            const size_t, const Buffer<float2>&, const size_t, const size_t,
                                            const size_t, const Buffer<float2>&, const size_t, const size_t,
                                            Buffer<float2>&, const size_t, const size_t,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double2>(const std::string&,
                                            const size_t, const Buffer<double2>&, const size_t, const size_t,
                                            const size_t, const Buffer<double2>&, const size_t, const size_t,
                                            Buffer<double2>&, const size_t, const size_t,
                                            const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
               const std::vector<size_t>& lstride, const std::vector<size_t>& rstride,
               const std::vector<size_t>& oshape,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_b<T>(queue, event);
    routine.DoTransform(name, n, x_buffer, y_buffer, z_buffer, lstride, rstride, oshape);
}

template void PUBLIC_API transform<int16_t>(const std::string&, const size_t,
                                            const Buffer<int16_t>&, const Buffer<int16_t>&, Buffer<int16_t>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&, const size_t,
                                            const Buffer<int32_t>&, const Buffer<int32_t>&, Buffer<int32_t>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&, const size_t,
                                            const Buffer<int64_t>&, const Buffer<int64_t>&, Buffer<int64_t>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&, const size_t,
                                            const Buffer<half>&, const Buffer<half>&, Buffer<half>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&, const size_t,
                                            const Buffer<float>&, const Buffer<float>&, Buffer<float>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&, const size_t,
                                            const Buffer<double>&, const Buffer<double>&, Buffer<double>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float2> (const std::string&, const size_t,
                                            const Buffer<float2>&, const Buffer<float2>&, Buffer<float2>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double2>(const std::string&, const size_t,
                                            const Buffer<double2>&, const Buffer<double2>&, Buffer<double2>&,
                                            const std::vector<size_t>&, const std::vector<size_t>&,
                                            const std::vector<size_t>&,
                                            const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const size_t n, const T alpha, const T beta,
               const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_p<T>(queue, event);
    routine.DoTransform(name, n, alpha, beta, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API transform<half>  (const std::string&, const size_t, const half, const half,
                                           const Buffer<half>&, const size_t, const size_t,
                                           Buffer<half>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<float> (const std::string&, const size_t, const float, const float,
                                           const Buffer<float>&, const size_t, const size_t,
                                           Buffer<float>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<double>(const std::string&, const size_t, const double, const double,
                                           const Buffer<double>&, const size_t, const size_t,
                                           Buffer<double>&, const size_t, const size_t,
                                           const Queue&, Event*);


}} // namespace gpgpu::dnn
