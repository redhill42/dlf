#include <cassert>
#include "gdnn.h"
#include "routines/routines.hpp"
#include "cudnn.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

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
    if (IsOpenCL(queue.context().device()) || (dims.size() != 4 && dims.size() != 5)) {
        auto batches = dims[0];
        auto channels = dims[1];
        auto spatial = std::accumulate(dims.begin()+2, dims.end(), size_t{1}, std::multiplies<>());

        auto routine = Xbatch_norm<T>(queue, event);
        routine.DoBatchNorm(batches, channels, spatial, x_buffer, y_buffer,
                            scale_buffer, bias_buffer, mean_buffer, var_buffer, epsilon);
    } else {
        TensorDescriptor<T> xy_desc(dims);
        TensorDescriptor<T> sbmv_desc;
        cudnnDeriveBNTensorDescriptor(sbmv_desc, xy_desc,
            cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL);

        const T alpha = 1, beta = 0;
        cudnnBatchNormalizationForwardInference(
            cudnn_handle(queue), cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta,
            xy_desc, reinterpret_cast<void*>(*cu::cuBuffer::unwrap(x_buffer)),
            xy_desc, reinterpret_cast<void*>(*cu::cuBuffer::unwrap(y_buffer)),
            sbmv_desc,
            reinterpret_cast<void*>(*cu::cuBuffer::unwrap(scale_buffer)),
            reinterpret_cast<void*>(*cu::cuBuffer::unwrap(bias_buffer)),
            reinterpret_cast<void*>(*cu::cuBuffer::unwrap(mean_buffer)),
            reinterpret_cast<void*>(*cu::cuBuffer::unwrap(var_buffer)),
            epsilon >= CUDNN_BN_MIN_EPSILON ? epsilon : 0.00001); // FIXME
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
void conv2d(const size_t channels, const size_t height, const size_t width,
            const size_t kernel_h, const size_t kernel_w,
            const size_t pad_t, const size_t pad_l, const size_t pad_b, const size_t pad_r,
            const size_t stride_h, const size_t stride_w,
            const size_t dilation_h, const size_t dilation_w,
            const size_t num_kernels, const size_t batch_count,
            const Buffer<T>& im_buffer, const Buffer<T>& kernel_buffer,
            Buffer<T>& result_buffer, const Queue& queue, Event* event)
{
    if (IsOpenCL(queue.context().device()) || (pad_t != pad_b || pad_l != pad_r)) {
        auto routine = Xconvgemm<T>(queue, event);
        routine.DoConvgemm(KernelMode::Convolution,
                           channels, height, width,
                           kernel_h, kernel_w,
                           pad_t, pad_l, pad_b, pad_r,
                           stride_h, stride_w,
                           dilation_h, dilation_w,
                           num_kernels, batch_count,
                           im_buffer, 0,
                           kernel_buffer, 0,
                           result_buffer, 0);
    } else {
        auto cudnn = cudnn_handle(queue);

        const auto size_h = height + pad_t + pad_b;
        const auto padding_h = dilation_h * (kernel_h - 1) + 1;
        const auto output_h = (size_h >= padding_h) ? (size_h - padding_h) / stride_h + 1 : 1;
        const auto size_w = width + pad_l + pad_r;
        const auto padding_w = dilation_w * (kernel_w - 1) + 1;
        const auto output_w = (size_w >= padding_w) ? (size_w - padding_w) / stride_w + 1 : 1;

        auto x_desc = TensorDescriptor<T>(batch_count, channels, height, width);
        auto w_desc = FilterDescriptor<T>(num_kernels, channels, kernel_h, kernel_w);
        auto y_desc = TensorDescriptor<T>(batch_count, num_kernels, output_h, output_w);
        auto conv_desc = ConvolutionDescriptor<T>(pad_t, pad_l, stride_h, stride_w, dilation_h, dilation_w);

        cudnnConvolutionFwdAlgo_t algo;
        checkCUDNN(cudnnGetConvolutionForwardAlgorithm(
            cudnn, x_desc, w_desc, conv_desc, y_desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            /*memoryLimitInBytes=*/0,
            &algo));

        size_t workspace_size = 0;
        checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));
        auto workspace = queue.context().createBuffer<char>(workspace_size);

        const T alpha = 1, beta = 0;
        checkCUDNN(cudnnConvolutionForward(
            cudnn, &alpha,
            x_desc, reinterpret_cast<void*>(*cu::cuBuffer::unwrap(im_buffer)),
            w_desc, reinterpret_cast<void*>(*cu::cuBuffer::unwrap(kernel_buffer)),
            conv_desc, algo,
            reinterpret_cast<void*>(*cu::cuBuffer::unwrap(workspace)), workspace_size,
            &beta,
            y_desc, reinterpret_cast<void*>(*cu::cuBuffer::unwrap((result_buffer)))));
    }
}

template void PUBLIC_API conv2d<float> (const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<float>&, const Buffer<float>&, Buffer<float>&,
                                        const Queue&, Event*);
template void PUBLIC_API conv2d<double>(const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<double>&, const Buffer<double>&, Buffer<double>&,
                                        const Queue&, Event*);
template void PUBLIC_API conv2d<half>  (const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t, const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const size_t, const size_t,
                                        const Buffer<half>&, const Buffer<half>&, Buffer<half>&,
                                        const Queue&, Event*);

}} // namespace gpgpu::dnn
