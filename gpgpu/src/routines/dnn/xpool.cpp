#include "xpool.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xpool<T>::Xpool(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xpool.cl"
    }){
}

template <typename T>
void Xpool<T>::DoMaxPool(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_h, const size_t pad_w,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
    gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    if (channels == 0 || height == 0 || width == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xmaxpool");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(channels),
                        static_cast<int>(height),
                        static_cast<int>(width),
                        static_cast<int>(output_h),
                        static_cast<int>(output_w),
                        static_cast<int>(kernel_h),
                        static_cast<int>(kernel_w),
                        static_cast<int>(pad_h),
                        static_cast<int>(pad_w),
                        static_cast<int>(stride_h),
                        static_cast<int>(stride_w),
                        static_cast<int>(dilation_h),
                        static_cast<int>(dilation_w),
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset));

    // Launches the kernel
    const auto w_ceiled = Ceil(output_w, db_["COPY_DIMX"]);
    const auto h_ceiled = Ceil(output_h, db_["COPY_DIMY"]);
    const auto global = std::vector<size_t>{w_ceiled, h_ceiled * channels, batches};
    const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"], 1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xpool<T>::DoAvgPool(
    const size_t batches, const size_t channels,
    const size_t height, const size_t width,
    const size_t output_h, const size_t output_w,
    const size_t kernel_h, const size_t kernel_w,
    const size_t pad_h, const size_t pad_w,
    const size_t stride_h, const size_t stride_w,
    const size_t dilation_h, const size_t dilation_w,
    const bool count_include_pad,
    const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
    gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    if (channels == 0 || height == 0 || width == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xavgpool");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(channels),
                        static_cast<int>(height),
                        static_cast<int>(width),
                        static_cast<int>(output_h),
                        static_cast<int>(output_w),
                        static_cast<int>(kernel_h),
                        static_cast<int>(kernel_w),
                        static_cast<int>(pad_h),
                        static_cast<int>(pad_w),
                        static_cast<int>(stride_h),
                        static_cast<int>(stride_w),
                        static_cast<int>(dilation_h),
                        static_cast<int>(dilation_w),
                        static_cast<int>(count_include_pad),
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset));

    // Launches the kernel
    const auto w_ceiled = Ceil(output_w, db_["COPY_DIMX"]);
    const auto h_ceiled = Ceil(output_h, db_["COPY_DIMY"]);
    const auto global = std::vector<size_t>{w_ceiled, h_ceiled * channels, batches};
    const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"], 1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xpool<half>;
template class Xpool<float>;
template class Xpool<double>;

}} // namespace gpgpu::dnn
