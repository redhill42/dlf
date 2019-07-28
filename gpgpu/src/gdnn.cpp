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
void copy(const size_t n,
          const Buffer<T>& x_buffer, const size_t x_offset,
          const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
          Buffer<T>& y_buffer, const size_t y_offset,
          const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
          const Queue& queue, Event* event) {
    auto routine = Xcopy<T>(queue, event);
    routine.DoCopyStrided(n, x_buffer, x_offset, x_dim, x_stride,
                             y_buffer, y_offset, y_dim, y_stride);
}

template void PUBLIC_API copy<int16_t>(const size_t, const Buffer<int16_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<int16_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int32_t>(const size_t, const Buffer<int32_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<int32_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<int64_t>(const size_t, const Buffer<int64_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<int64_t>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<half>   (const size_t, const Buffer<half>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<half>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float>  (const size_t, const Buffer<float>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<float>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double> (const size_t, const Buffer<double>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<double>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<float2> (const size_t, const Buffer<float2>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<float2>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       const Queue&, Event*);
template void PUBLIC_API copy<double2>(const size_t, const Buffer<double2>&, const size_t,
                                       const std::vector<size_t>&, const std::vector<size_t>&,
                                       Buffer<double2>&, const size_t,
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
void transform(const std::string& name,
               const size_t m, const size_t n, const size_t channels,
               const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
               const Queue& queue, Event* event)
{
    auto routine = Xtransform_c<T>(queue, event);
    routine.DoTransform(name, m, n, channels, x_buffer, y_buffer, z_buffer);
}

template void PUBLIC_API transform<int16_t>(const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<int16_t>&, const Buffer<int16_t>&,
                                            Buffer<int16_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int32_t>(const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<int32_t>&, const Buffer<int32_t>&,
                                            Buffer<int32_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<int64_t>(const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<int64_t>&, const Buffer<int64_t>&,
                                            Buffer<int64_t>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<half>   (const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<half>&, const Buffer<half>&,
                                            Buffer<half>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float>  (const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<float>&, const Buffer<float>&,
                                            Buffer<float>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double> (const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<double>&, const Buffer<double>&,
                                            Buffer<double>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<float2> (const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<float2>&, const Buffer<float2>&,
                                            Buffer<float2>&,
                                            const Queue&, Event*);
template void PUBLIC_API transform<double2>(const std::string&,
                                            const size_t, const size_t, const size_t,
                                            const Buffer<double2>&, const Buffer<double2>&,
                                            Buffer<double2>&,
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
    if (IsOpenCL(queue.context().device()) || (dims.size() != 4 && dims.size() != 5)) {
        auto batches = dims[0];
        auto channels = dims[1];
        auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

        auto routine = Xnormalization<T>(queue, event);
        routine.DoBatchNorm(batches, channels, spatial, x_buffer, y_buffer,
                            scale_buffer, bias_buffer, mean_buffer, var_buffer, epsilon);
    } else {
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
    }
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
    if (IsOpenCL(queue.context().device()) || (dims.size() != 4 && dims.size() != 5)) {
        auto batches = dims[0];
        auto channels = dims[1];
        auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

        auto routine = Xnormalization<T>(queue, event);
        routine.DoLRN(batches, channels, spatial, x_buffer, y_buffer, nsize, alpha, beta, bias);
    } else {
        TensorDescriptor<T> xy_desc(dims);
        LRNDescriptor desc(nsize, alpha, beta, bias);

        T blend_alpha = 1, blend_beta = 0;
        checkCUDNN(cudnnLRNCrossChannelForward(
            cudnn_handle(queue), desc, CUDNN_LRN_CROSS_CHANNEL_DIM1,
            &blend_alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &blend_beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
    }
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
    auto cudnn = cudnn_handle(queue);

    auto desc = ConvolutionDescriptor<T>(pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w);
    auto x_desc = TensorDescriptor<T>(batches, channels, height, width);
    auto w_desc = FilterDescriptor<T>(num_kernels, channels/group, kernel_h, kernel_w);
    auto y_desc = TensorDescriptor<T>(batches, num_kernels, output_h, output_w);

    checkCUDNN(cudnnSetConvolutionGroupCount(desc, group));

    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn, x_desc, w_desc, desc, y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        &algo));

    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, desc, y_desc, algo, &workspace_size));

    Buffer<T> temp_buffer;
    if (work == nullptr) {
        temp_buffer = queue.context().createBuffer<T>(workspace_size / sizeof(T));
        work = &temp_buffer;
    } else {
        assert(work->size() * sizeof(T) >= workspace_size);
    }

    const T alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(
        cudnn, &alpha,
        x_desc, cu::cuBuffer::unwrap(im_buffer),
        w_desc, cu::cuBuffer::unwrap(kernel_buffer),
        desc, algo,
        cu::cuBuffer::unwrap(*work), workspace_size,
        &beta,
        y_desc, cu::cuBuffer::unwrap((result_buffer))));
}

template <typename T>
size_t cudnnConv2dWorkspaceSize(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t num_kernels, const size_t group,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_top, const size_t pad_left,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const Queue& queue)
{
    auto cudnn = cudnn_handle(queue);

    auto desc = ConvolutionDescriptor<T>(pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w);
    auto x_desc = TensorDescriptor<T>(batches, channels, height, width);
    auto w_desc = FilterDescriptor<T>(num_kernels, channels/group, kernel_h, kernel_w);
    auto y_desc = TensorDescriptor<T>(batches, num_kernels, output_h, output_w);

    checkCUDNN(cudnnSetConvolutionGroupCount(desc, group));

    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn, x_desc, w_desc, desc, y_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        &algo));

    size_t workspace_size = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, x_desc, w_desc, desc, y_desc, algo, &workspace_size));
    return workspace_size / sizeof(T);
}

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

    std::vector<size_t> w_offsets(batches*group);
    std::vector<size_t> x_offsets(batches*group);
    std::vector<size_t> y_offsets(batches*group);
    std::vector<T> alpha(batches*group);
    std::vector<T> beta(batches*group);

    for (size_t i = 0; i < batches; i++) {
        for (size_t j = 0; j < group; j++) {
            auto b = i * group + j;
            w_offsets[b] = j * m * k;
            x_offsets[b] = b * k * n;
            y_offsets[b] = b * m * n;
            alpha[b] = T{1};
            beta[b] = T{0};
        }
    }

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

    gemmBatched(gpgpu::blas::Layout::RowMajor,
                gpgpu::blas::Transpose::NoTrans,
                gpgpu::blas::Transpose::NoTrans,
                m, n, k, &alpha[0],
                kernel_buffer, &w_offsets[0], k,
                *x_buffer, &x_offsets[0], n,
                &beta[0], result_buffer, &y_offsets[0], n,
                batches * group);
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
    if (IsCUDA(queue.context().device()) && pad_top == pad_bottom && pad_left == pad_right) {
        cudnnConv(
            batches, channels, height, width, output_h, output_w,
            num_kernels, group, kernel_h, kernel_w,
            pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w,
            im_buffer, kernel_buffer, result_buffer, work_buffer, queue);
        return;
    }

    bool is_1x1_kernel =
        kernel_h == 1 && kernel_w == 1 &&
        stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 &&
        pad_top + pad_bottom == 0  && pad_left + pad_right == 0;

    if (is_1x1_kernel || group != 1) {
        convWithIm2Col(
            is_1x1_kernel,
            batches, channels, height, width, output_h, output_w,
            num_kernels, group, kernel_h, kernel_w,
            pad_top, pad_left, stride_h, stride_w, dilation_h, dilation_w,
            im_buffer, kernel_buffer, result_buffer, work_buffer,
            queue, event);
        return;
    }

    convgemm(KernelMode::CrossCorrelation,
             batches, channels,
             height, width, output_h, output_w,
             num_kernels, kernel_h, kernel_w,
             pad_top, pad_left, stride_h, stride_w,
             dilation_h, dilation_w,
             im_buffer, 0, kernel_buffer, 0, result_buffer, 0,
             work_buffer, queue, event);
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
    if (IsCUDA(queue.context().device()) && pad_top == pad_bottom && pad_left == pad_right) {
        return cudnnConv2dWorkspaceSize<T>(
            batches, channels, height, width, output_h, output_w,
            num_kernels, group, kernel_h, kernel_w, pad_top, pad_left,
            stride_h, stride_w, dilation_h, dilation_w, queue);
    }

    bool is_1x1_kernel =
        kernel_h == 1 && kernel_w == 1 &&
        stride_h == 1 && stride_w == 1 &&
        dilation_h == 1 && dilation_w == 1 &&
        pad_top + pad_bottom == 0 && pad_left + pad_right == 0;

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
    if (IsOpenCL(queue.context().device()) || (pad_top != pad_bottom || pad_left != pad_right)
                                           || (dilation_h != 1 || dilation_w != 1)) {
        auto routine = Xpool<T>(queue, event);
        routine.DoMaxPool(batches, channels,
                          height, width,
                          output_h, output_w,
                          kernel_h, kernel_w,
                          pad_top, pad_left,
                          stride_h, stride_w,
                          dilation_h, dilation_w,
                          x_buffer, 0, y_buffer, 0);
    } else {
        PoolingDescriptor desc(cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            kernel_h, kernel_w, pad_top, pad_left, stride_h, stride_w);
        TensorDescriptor<T> x_desc(batches, channels, height, width);
        TensorDescriptor<T> y_desc(batches, channels, output_h, output_w);

        T alpha = 1, beta = 0;
        cudnnPoolingForward(
            cudnn_handle(queue), desc,
            &alpha, x_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta, y_desc, cu::cuBuffer::unwrap(y_buffer));
    }
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
    if (IsOpenCL(queue.context().device()) || (pad_top != pad_bottom || pad_left != pad_right)
                                           || (dilation_h != 1 || dilation_w != 1)) {
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
    } else {
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
    }
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
    if (IsOpenCL(queue.context().device())) {
        auto routine = Xsoftmax<T>(queue, event);
        routine.DoSoftmax(m, n, x_buffer, y_buffer);
    } else {
        TensorDescriptor<T> xy_desc(m, 1, 1, n);
        T alpha = 1, beta = 0;
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle(queue),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
    }
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
    if (IsOpenCL(queue.context().device())) {
        auto routine = Xsoftmax<T>(queue, event);
        routine.DoLogSoftmax(m, n, x_buffer, y_buffer);
    } else {
        TensorDescriptor<T> xy_desc(m, 1, 1, n);
        T alpha = 1, beta = 0;
        checkCUDNN(cudnnSoftmaxForward(
            cudnn_handle(queue),
            cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG,
            cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            &alpha, xy_desc, cu::cuBuffer::unwrap(x_buffer),
            &beta,  xy_desc, cu::cuBuffer::unwrap(y_buffer)));
    }
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

}} // namespace gpgpu::dnn
