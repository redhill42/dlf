#include "xbinary.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xbinary<T>::Xbinary(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "kernels/dnn/xbinary.cl"
}) {
}

template <typename T>
void Xbinary<T>::DoBinary(
    const std::string& name,
    const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset, const size_t x_inc,
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
    auto kernel_name = "X" + name;
    kernel_name = use_fastest_kernel ? kernel_name + "Fastest" :
                  use_faster_kernel  ? kernel_name + "Faster" : kernel_name;

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

template <typename T>
void Xbinary<T>::DoBinaryStrided(const std::string& name, const size_t n,
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
    kernel.setArgument(0, static_cast<int>(n));
    kernel.setArgument(1, static_cast<int>(rank));
    kernel.setArgument(2, shape_buffer);
    kernel.setArgument(3, x_buffer);
    kernel.setArgument(4, y_buffer);
    kernel.setArgument(5, z_buffer);

    // Launches the kernel
    const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xbinary<int16_t>;
template class Xbinary<int32_t>;
template class Xbinary<int64_t>;
template class Xbinary<half>;
template class Xbinary<float>;
template class Xbinary<double>;
template class Xbinary<float2>;
template class Xbinary<double2>;

}}