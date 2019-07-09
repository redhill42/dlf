#include <cassert>
#include "gdnn.h"
#include "routines/routines.hpp"
#include "gpgpu_cu.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
class TensorDescriptor {
    cudnnTensorDescriptor_t desc;

public:
    TensorDescriptor() {
        cudnnCreateTensorDescriptor(&desc);
    }

    explicit TensorDescriptor(const std::vector<size_t> dims) {
        cudnnDataType_t dtype;
        switch (PrecisionValue<T>()) {
        case Precision::Half:
            dtype = cudnnDataType_t::CUDNN_DATA_HALF;
            break;
        case Precision::Single:
            dtype = cudnnDataType_t::CUDNN_DATA_FLOAT;
            break;
        case Precision::Double:
            dtype = cudnnDataType_t::CUDNN_DATA_DOUBLE;
            break;
        case Precision::Int:
            dtype = cudnnDataType_t::CUDNN_DATA_INT32;
            break;
        default:
            throw std::runtime_error("cudnn: unsupported data type");
        }

        int  rank = dims.size();
        int* i_dims = reinterpret_cast<int*>(alloca(rank * sizeof(int)));
        int* i_strides = reinterpret_cast<int*>(alloca(rank * sizeof(int)));
        int  size = 1;
        for (int i = 0; i < rank; i++) {
            i_dims[i] = static_cast<int>(dims[i]);
            i_strides[i] = size;
            size *= dims[i];
        }

        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensorNdDescriptor(desc, dtype, rank, i_dims, i_strides);
    }

    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc);
    }

    operator cudnnTensorDescriptor_t() { return desc; }
    operator const cudnnTensorDescriptor_t() const { return desc; }
};

template <typename T>
void copy(const size_t x_size, const Buffer<T>& x_buffer,
          const size_t y_size, Buffer<T>& y_buffer,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopy(x_size, x_buffer, y_size, y_buffer);
}

template void PUBLIC_API copy<int16_t>(const size_t, const Buffer<int16_t>&,
                                       const size_t, Buffer<int16_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const Buffer<int32_t>&,
                                       const size_t, Buffer<int32_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const Buffer<int64_t>&,
                                       const size_t, Buffer<int64_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const Buffer<float>&,
                                       const size_t, Buffer<float>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const Buffer<double>&,
                                       const size_t, Buffer<double>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const Buffer<float2>&,
                                       const size_t, Buffer<float2>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const Buffer<double2>&,
                                       const size_t, Buffer<double2>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const Buffer<half>&,
                                       const size_t, Buffer<half>&,
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
void concat_copy(const size_t n, const size_t offset, const size_t block, const size_t stride,
                 const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                 const Queue& queue, Event* event)
{
    auto routine = Xcopy<T>(queue, event);
    routine.DoConcatCopy(n, offset, block, stride, x_buffer, y_buffer);
}

template void PUBLIC_API concat_copy<int16_t>(const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<int16_t>&, Buffer<int16_t>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<int32_t>(const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<int32_t>&, Buffer<int32_t>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<int64_t>(const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<int64_t>&, Buffer<int64_t>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<half>   (const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<half>&, Buffer<half>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<float>  (const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<float>&, Buffer<float>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<double> (const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<double>&, Buffer<double>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<float2> (const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<float2>&, Buffer<float2>&,
                                              const Queue&, Event*);
template void PUBLIC_API concat_copy<double2>(const size_t, const size_t, const size_t, const size_t,
                                              const Buffer<double2>&, Buffer<double2>&,
                                              const Queue&, Event*);

template <typename T>
void split_copy(const size_t n, const size_t offset, const size_t block, const size_t stride,
                 const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
                 const Queue& queue, Event* event)
{
    auto routine = Xcopy<T>(queue, event);
    routine.DoSplitCopy(n, offset, block, stride, x_buffer, y_buffer);
}

template void PUBLIC_API split_copy<int16_t>(const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<int16_t>&, Buffer<int16_t>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<int32_t>(const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<int32_t>&, Buffer<int32_t>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<int64_t>(const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<int64_t>&, Buffer<int64_t>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<half>   (const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<half>&, Buffer<half>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<float>  (const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<float>&, Buffer<float>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<double> (const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<double>&, Buffer<double>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<float2> (const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<float2>&, Buffer<float2>&,
                                             const Queue&, Event*);
template void PUBLIC_API split_copy<double2>(const size_t, const size_t, const size_t, const size_t,
                                             const Buffer<double2>&, Buffer<double2>&,
                                             const Queue&, Event*);

template <typename T>
void transform(const std::string& name, const size_t n,
               const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform<T>(queue, event);
    routine.DoTransform(name, n, x_buffer, y_buffer);
}

template void PUBLIC_API transform<int16_t>(const std::string&, const size_t,
                                            const Buffer<int16_t>&, Buffer<int16_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&, const size_t,
                                            const Buffer<int32_t>&, Buffer<int32_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&, const size_t,
                                            const Buffer<int64_t>&, Buffer<int64_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&, const size_t,
                                            const Buffer<half>&, Buffer<half>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&, const size_t,
                                            const Buffer<float>&, Buffer<float>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&, const size_t,
                                            const Buffer<double>&, Buffer<double>&,
                                            const Queue&, Event*);

template <typename T>
void transform(const std::string& name,
                const size_t x_size, const Buffer<T>& x_buffer,
                const size_t y_size, const Buffer<T>& y_buffer,
                Buffer<T>& z_buffer,
                const Queue& queue, Event* event)
{
    auto routine = Xtransform_b<T>(queue, event);
    routine.DoTransform(name, x_size, x_buffer, y_size, y_buffer, z_buffer);
}

template void PUBLIC_API transform<int16_t>(const std::string&,
                                            const size_t, const Buffer<int16_t>&,
                                            const size_t, const Buffer<int16_t>&,
                                            Buffer<int16_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&,
                                            const size_t, const Buffer<int32_t>&,
                                            const size_t, const Buffer<int32_t>&,
                                            Buffer<int32_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&,
                                            const size_t, const Buffer<int64_t>&,
                                            const size_t, const Buffer<int64_t>&,
                                            Buffer<int64_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&,
                                            const size_t, const Buffer<half>&,
                                            const size_t, const Buffer<half>&,
                                            Buffer<half>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&,
                                            const size_t, const Buffer<float>&,
                                            const size_t, const Buffer<float>&,
                                            Buffer<float>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&,
                                            const size_t, const Buffer<double>&,
                                            const size_t, const Buffer<double>&,
                                            Buffer<double>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float2> (const std::string&,
                                            const size_t, const Buffer<float2>&,
                                            const size_t, const Buffer<float2>&,
                                            Buffer<float2>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double2>(const std::string&,
                                            const size_t, const Buffer<double2>&,
                                            const size_t, const Buffer<double2>&,
                                            Buffer<double2>&,
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
               const Buffer<T>& x_buffer, Buffer<T>& y_buffer,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_p<T>(queue, event);
    routine.DoTransform(name, n, alpha, beta, x_buffer, y_buffer);
}

template void PUBLIC_API transform<half>  (const std::string&, const size_t, const half, const half,
                                           const Buffer<half>&, Buffer<half>&,
                                           const Queue&, Event*);
template void PUBLIC_API transform<float> (const std::string&, const size_t, const float, const float,
                                           const Buffer<float>&, Buffer<float>&,
                                           const Queue&, Event*);
template void PUBLIC_API transform<double>(const std::string&, const size_t, const double, const double,
                                           const Buffer<double>&, Buffer<double>&,
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
    auto batches = dims[0];
    auto channels = dims[1];
    auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

    auto routine = Xbatch_norm<T>(queue, event);
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

}} // namespace gpgpu::dnn
