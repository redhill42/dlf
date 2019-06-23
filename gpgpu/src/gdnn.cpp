#include "gdnn.h"
#include "routines/routines.hpp"

namespace gpgpu { namespace dnn {

using namespace gpgpu::blas;

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
               Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform<T>(queue, event);
    routine.DoTransform(name, n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API transform<half>  (const std::string&, const size_t,
                                           const Buffer<half>&, const size_t, const size_t,
                                           Buffer<half>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<float> (const std::string&, const size_t,
                                           const Buffer<float>&, const size_t, const size_t,
                                           Buffer<float>&, const size_t, const size_t,
                                           const Queue&, Event*);
template void PUBLIC_API transform<double>(const std::string&, const size_t,
                                           const Buffer<double>&, const size_t, const size_t,
                                           Buffer<double>&, const size_t, const size_t,
                                           const Queue&, Event*);

template <typename T>
void abs(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue, Event* event)
{
    auto routine = Xabs<T>(queue, event);
    routine.DoAbs(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API abs<int16_t>(const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                      Buffer<int16_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API abs<int32_t>(const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API abs<int64_t>(const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<int64_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API abs<half>   (const size_t, const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API abs<float>  (const size_t, const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API abs<double> (const size_t, const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);

template <typename T>
void neg(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
         Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
         const Queue& queue, Event* event)
{
    auto routine = Xneg<T>(queue, event);
    routine.DoNeg(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API neg<int16_t>(const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                      Buffer<int16_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API neg<int32_t>(const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                      Buffer<int32_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API neg<int64_t>(const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                      Buffer<int64_t>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API neg<half>   (const size_t, const Buffer<half>&, const size_t, const size_t,
                                      Buffer<half>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API neg<float>  (const size_t, const Buffer<float>&, const size_t, const size_t,
                                      Buffer<float>&, const size_t, const size_t,
                                      const Queue&, Event*);
template void PUBLIC_API neg<double> (const size_t, const Buffer<double>&, const size_t, const size_t,
                                      Buffer<double>&, const size_t, const size_t,
                                      const Queue&, Event*);

template <typename T>
void sign(const size_t n, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
          Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
          const Queue& queue, Event* event)
{
    auto routine = Xsign<T>(queue, event);
    routine.DoSign(n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc);
}

template void PUBLIC_API sign<int16_t>(const size_t, const Buffer<int16_t>&, const size_t, const size_t,
                                       Buffer<int16_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API sign<int32_t>(const size_t, const Buffer<int32_t>&, const size_t, const size_t,
                                       Buffer<int32_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API sign<int64_t>(const size_t, const Buffer<int64_t>&, const size_t, const size_t,
                                       Buffer<int64_t>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API sign<half>   (const size_t, const Buffer<half>&, const size_t, const size_t,
                                       Buffer<half>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API sign<float>  (const size_t, const Buffer<float>&, const size_t, const size_t,
                                       Buffer<float>&, const size_t, const size_t,
                                       const Queue&, Event*);
template void PUBLIC_API sign<double> (const size_t, const Buffer<double>&, const size_t, const size_t,
                                       Buffer<double>&, const size_t, const size_t,
                                       const Queue&, Event*);

}} // namespace gpgpu::dnn
