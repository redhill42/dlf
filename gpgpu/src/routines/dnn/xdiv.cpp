#include "routines/dnn/xdiv.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xdiv<T>::Xdiv(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xdiv.cl"
    }) {
}

template <typename T>
void Xdiv<T>::DoDiv(const size_t n,
                    const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
                    const Buffer<T>& y_buffer, const size_t y_offset, const size_t y_inc,
                    Buffer<T>& z_buffer, const size_t z_offset, const size_t z_inc)
{
    // Make sure all dimensions are larger than zero
    if (n == 0) throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    if (x_inc != 0)
        TestVectorX(n, x_buffer, x_offset, x_inc);
    if (y_inc != 0)
        TestVectorY(n, y_buffer, y_offset, y_inc);
    TestVectorY(n, z_buffer, z_offset, z_inc); // TODO: Make a TestVectorZ function with error codes

    // Determines whether or not the fast-version can be used
    const auto use_faster_kernel = (x_offset == 0) && (x_inc == 1) &&
                                   (y_offset == 0) && (y_inc == 1) &&
                                   (z_offset == 0) && (z_inc == 1) &&
                                   IsMultiple(n, db_["WPT"]*db_["VW"]);
    const auto use_fastest_kernel = use_faster_kernel &&
                                    IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

    // If possible, run the fast-version of the kernel
    const auto kernel_name = (use_fastest_kernel) ? "XdivFastest" :
                             (use_faster_kernel)  ? "XdivFaster" : "Xdiv";

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel(kernel_name);

    // Sets the kernel arguments
    if (use_faster_kernel || use_fastest_kernel) {
        kernel.setArgument(0, static_cast<int>(n));
        kernel.setArgument(1, x_buffer);
        kernel.setArgument(2, y_buffer);
        kernel.setArgument(3, z_buffer);
    } else {
        kernel.setArgument(0, static_cast<int>(n));
        kernel.setArgument(1, x_buffer);
        kernel.setArgument(2, static_cast<int>(x_offset));
        kernel.setArgument(3, static_cast<int>(x_inc));
        kernel.setArgument(4, y_buffer);
        kernel.setArgument(5, static_cast<int>(y_offset));
        kernel.setArgument(6, static_cast<int>(y_inc));
        kernel.setArgument(7, z_buffer);
        kernel.setArgument(8, static_cast<int>(z_offset));
        kernel.setArgument(9, static_cast<int>(z_inc));
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

template class Xdiv<int16_t>;
template class Xdiv<int32_t>;
template class Xdiv<int64_t>;
template class Xdiv<half>;
template class Xdiv<float>;
template class Xdiv<double>;
template class Xdiv<float2>;
template class Xdiv<double2>;

}} // namespace gpgpu::dnn
