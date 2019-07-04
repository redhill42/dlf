#include "xtransform_b.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xtransform_b<T>::Xtransform_b(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "kernels/dnn/xtransform_b.cl"
}) {
}

static inline bool ends_with(const std::string& str, const std::string& suffix) {
    return str.length() >= suffix.length() &&
        0 == str.compare(str.length()-suffix.length(), suffix.length(), suffix);
}

template <typename T>
void Xtransform_b<T>::DoTransform(
    const std::string& name,
    const size_t x_size, const Buffer<T>& x_buffer,
    const size_t y_size, const Buffer<T>& y_buffer,
    Buffer<T>& z_buffer)
{
    const size_t n = std::max(x_size, y_size);

    // Make sure all dimensions are larger than zero
    if (x_size == 0 || y_size == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    TestVectorX(x_size, x_buffer, 0, 1);
    TestVectorY(y_size, y_buffer, 0, 1);
    TestVectorY(n, z_buffer, 0, 1); // TODO: Make a TestVectorZ function with error codes

    // Determines whether or not the fast-version can be used
    const auto use_faster_kernel = (ends_with(name, "_v")) &&
                                   (x_size == y_size) &&
                                   IsMultiple(n, db_["WPT"]*db_["VW"]);
    const auto use_fastest_kernel = use_faster_kernel &&
                                    IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

    // If possible, run the fast-version of the kernel
    auto kernel_name = "X" + name;
    kernel_name = use_fastest_kernel ? kernel_name + "Fastest" :
                  use_faster_kernel  ? kernel_name + "Faster" : kernel_name;

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel(kernel_name);

    // Sets the kernel arguments
    if (use_faster_kernel || use_fastest_kernel) {
        kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer, z_buffer);
    } else {
        kernel.setArguments(static_cast<int>(x_size), x_buffer,
                            static_cast<int>(y_size), y_buffer,
                            z_buffer);
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

template <typename T>
void Xtransform_b<T>::DoTransform(const std::string& name, const size_t n,
    const Buffer<T>& x_buffer, const Buffer<T>& y_buffer, Buffer<T>& z_buffer,
    const std::vector<size_t>& lstride, const std::vector<size_t>& rstride,
    const std::vector<size_t>& oshape)
{
    // Make sure all dimensions are larger than zero
    if (n == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Create compact buffer to hold strides and shapes
    auto rank = oshape.size();
    assert(lstride.size() == rank && rstride.size() == rank);
    std::vector<int> shape_data(rank * 3);
    std::copy(oshape.begin(), oshape.end(), shape_data.begin());
    std::copy(lstride.begin(), lstride.end(), shape_data.begin() + rank);
    std::copy(rstride.begin(), rstride.end(), shape_data.begin() + rank*2);
    Buffer<int> shape_buffer = context_.createBuffer<int>(rank*3, BufferAccess::WriteOnly);
    shape_buffer.write(queue_, shape_data.data(), shape_data.size());

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name + "Strided");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n), static_cast<int>(rank),
                        shape_buffer, x_buffer, y_buffer, z_buffer);

    // Launches the kernel
    const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtransform_b<int16_t>;
template class Xtransform_b<int32_t>;
template class Xtransform_b<int64_t>;
template class Xtransform_b<half>;
template class Xtransform_b<float>;
template class Xtransform_b<double>;
template class Xtransform_b<float2>;
template class Xtransform_b<double2>;

}}
