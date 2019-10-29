#include "xcopy.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xcopy<T>::Xcopy(const Queue& queue, Event* event, const std::string& name):
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xcopy.cl"
}) {}

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
    } else {
        auto n_ceiled = Ceil(y_size, db_["WGS"]*db_["WPT"]);
        auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

// The main routine
template <typename T>
void Xcopy<T>::DoCopyStrided(const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride)
{
    if (IsContiguous(dims, x_stride) && IsContiguous(dims, y_stride)) {
        DoCopy(n, x_buffer, x_offset, n, y_buffer, y_offset);
        return;
    }

    // Create compact buffer to hold strides and dims
    auto shape_buffer = PackShape(dims, x_stride, y_stride, context_, queue_);
    auto kernel = program_.getKernel("XcopyStrided");
    kernel.setArguments(static_cast<int>(n),
                        static_cast<int>(dims.size()), shape_buffer,
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset));

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
