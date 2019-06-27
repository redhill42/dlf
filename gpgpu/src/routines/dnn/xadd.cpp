#include "routines/dnn/xadd.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xadd<T>::Xadd(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xadd.cl"
    #include "../../kernels/dnn/xtransform2.cl"
    }) {
}

template <typename T>
void Xadd<T>::DoAdd(const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                    const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                    Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc)
{
    const size_t n = std::max(x_size, y_size);

    // Make sure all dimensions are larger than zero
    if (x_size == 0 || y_size == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    TestVectorX(x_size, x_buffer, x_offset, x_inc);
    TestVectorY(y_size, y_buffer, y_offset, y_inc);
    TestVectorY(n, z_buffer, z_offset, z_inc); // TODO: Make a TestVectorZ function with error codes

    // Determines whether or not the fast-version can be used
    const auto use_faster_kernel = (x_size == y_size) &&
                                   (x_offset == 0) && (x_inc == 1) &&
                                   (y_offset == 0) && (y_inc == 1) &&
                                   (z_offset == 0) && (z_inc == 1) &&
                                   IsMultiple(n, db_["WPT"]*db_["VW"]);
    const auto use_fastest_kernel = use_faster_kernel &&
                                    IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

    // If possible, run the fast-version of the kernel
    const auto kernel_name = (use_fastest_kernel) ? "XaddFastest" :
                             (use_faster_kernel)  ? "XaddFaster" : "Xadd";

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel(kernel_name);

    // Sets the kernel arguments
    if (use_faster_kernel || use_fastest_kernel) {
        kernel.setArgument(0, static_cast<int>(n));
        kernel.setArgument(1, x_buffer);
        kernel.setArgument(2, y_buffer);
        kernel.setArgument(3, z_buffer);
    } else {
        kernel.setArgument(0, static_cast<int>(x_size));
        kernel.setArgument(1, x_buffer);
        kernel.setArgument(2, static_cast<int>(x_offset));
        kernel.setArgument(3, static_cast<int>(x_inc));
        kernel.setArgument(4, static_cast<int>(y_size));
        kernel.setArgument(5, y_buffer);
        kernel.setArgument(6, static_cast<int>(y_offset));
        kernel.setArgument(7, static_cast<int>(y_inc));
        kernel.setArgument(8, z_buffer);
        kernel.setArgument(9, static_cast<int>(z_offset));
        kernel.setArgument(10, static_cast<int>(z_inc));
    }

    // Launches the kernel
    if (use_fastest_kernel) {
        auto global = std::vector<size_t>{CeilDiv(n, db_["WPT"]*db_["VW"])};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else if (use_faster_kernel) {
        auto global = std::vector<size_t>{Ceil(CeilDiv(n, db_["WPT"] * db_["VW"]), db_["WGS"])};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    } else {
        const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
        auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
        auto local = std::vector<size_t>{db_["WGS"]};
        RunKernel(kernel, queue_, device_, global, local, event_);
    }
}

template class Xadd<int16_t>;
template class Xadd<int32_t>;
template class Xadd<int64_t>;
template class Xadd<half>;
template class Xadd<float>;
template class Xadd<double>;
template class Xadd<float2>;
template class Xadd<double2>;

}} // namespace gpgpu::dnn
