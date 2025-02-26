#include "xtransform_b.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T, typename R>
Xtransform_b<T, R>::Xtransform_b(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Xaxpy", "Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/level1/level1.cl"
    #include "../../kernels/dnn/xtransform_b.cl"
}) {}

template <typename T, typename R>
void Xtransform_b<T, R>::DoTransform(
    const size_t x_size, const Buffer<T>& x_buffer, const size_t x_offset,
    const size_t y_size, const Buffer<T>& y_buffer, const size_t y_offset,
    Buffer<R>& z_buffer, const size_t z_offset)
{
    const size_t n = std::max(x_size, y_size);

    // Make sure all dimensions are larger than zero
    if (x_size == 0 || y_size == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Tests the vectors for validity
    TestVectorX(x_size, x_buffer, x_offset, 1);
    TestVectorY(y_size, y_buffer, y_offset, 1);
    TestVectorY(n, z_buffer, z_offset, 1); // TODO: Make a TestVectorZ function with error codes

    // Determines whether or not the fast-version can be used
    const auto use_faster_kernel =
        (routine_name_ == "add" || routine_name_ == "sub" ||
         routine_name_ == "mul" || routine_name_ == "div") &&
        (x_size == y_size) &&
        (x_offset == 0 && y_offset == 0 && z_offset == 0) &&
        IsMultiple(n, db_["WPT"]*db_["VW"]);
    const auto use_fastest_kernel =
        use_faster_kernel && IsMultiple(n, db_["WGS"]*db_["WPT"]*db_["VW"]);

    // If possible, run the fast-version of the kernel
    std::string kernel_name = "Xtransform";
    kernel_name = use_fastest_kernel ? kernel_name + "Fastest" :
                  use_faster_kernel  ? kernel_name + "Faster"  :
                  x_size == 1        ? kernel_name + "ExpandL" :
                  y_size == 1        ? kernel_name + "ExpandR" :
                  x_size < y_size    ? kernel_name + "RepeatL" :
                  y_size < x_size    ? kernel_name + "RepeatR"
                                     : kernel_name;

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel(kernel_name);

    // Sets the kernel arguments
    if (use_faster_kernel || use_fastest_kernel) {
        kernel.setArguments(static_cast<int>(n), x_buffer, y_buffer, z_buffer);
    } else {
        kernel.setArguments(static_cast<int>(x_size), x_buffer, static_cast<int>(x_offset),
                            static_cast<int>(y_size), y_buffer, static_cast<int>(y_offset),
                            z_buffer, static_cast<int>(z_offset));
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

template <typename T, typename R>
void Xtransform_b<T, R>::DoTransformStrided(
    const size_t n, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_stride,
    const Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_stride,
    Buffer<R>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_stride)
{
    // Make sure all dimensions are larger than zero
    if (n == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Create compact buffer to hold strides and shapes
    auto shape_buffer = PackShape(dims, x_stride, y_stride, z_stride, context_, queue_);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("XtransformStrided");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(n),
                        static_cast<int>(dims.size()),
                        shape_buffer,
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset),
                        z_buffer, static_cast<int>(z_offset));

    // Launches the kernel
    const auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T, typename R>
void Xtransform_b<T, R>::DoTransformChannel(
    const size_t m, const size_t n, const size_t channels,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const Buffer<T>& y_buffer, const size_t y_offset,
    Buffer<R>& z_buffer, const size_t z_offset)
{
    // Make sure all dimensions are larger than zero
    if (m == 0 || n == 0 || channels == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("XtransformChannel");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(m),
                        static_cast<int>(n),
                        static_cast<int>(channels),
                        x_buffer, static_cast<int>(x_offset),
                        y_buffer, static_cast<int>(y_offset),
                        z_buffer, static_cast<int>(z_offset));

    // Launches the kernel
    auto m_ceiled = Ceil(m, db_["COPY_DIMX"]);
    auto n_ceiled = Ceil(n, db_["COPY_DIMY"]*db_["COPY_WPT"]);
    auto global = std::vector<size_t>{m_ceiled, n_ceiled/db_["COPY_WPT"]};
    auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtransform_b<int16_t, int16_t>;
template class Xtransform_b<int32_t, int32_t>;
template class Xtransform_b<int64_t, int64_t>;
template class Xtransform_b<half,    half>;
template class Xtransform_b<float,   float>;
template class Xtransform_b<double,  double>;
template class Xtransform_b<float2,  float2>;
template class Xtransform_b<double2, double2>;

template class Xtransform_b<int16_t, bool>;
template class Xtransform_b<int32_t, bool>;
template class Xtransform_b<int64_t, bool>;
template class Xtransform_b<half,    bool>;
template class Xtransform_b<float,   bool>;
template class Xtransform_b<double,  bool>;

}} // namespace gpgpu::dnn
