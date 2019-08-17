#include "xtransform_c.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T, typename R>
Xtransform_c<T, R>::Xtransform_c(const Queue& queue, Event* event, const std::string& name)
    : Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
    #include "kernels/dnn/xtransform_c.cl"
    }) {
}

template <typename T, typename R>
void Xtransform_c<T, R>::DoTransform(
    const std::string& name,
    const size_t m, const size_t n, const size_t channels,
    const Buffer<T>& x_buffer, const Buffer<T>& y_buffer,
    gpgpu::Buffer<R>& z_buffer)
{
    // Make sure all dimensions are larger than zero
    if (m == 0 || n == 0 || channels == 0)
        throw BLASError(StatusCode::kInvalidDimension);

    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("X" + name);

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(m),
                        static_cast<int>(n),
                        static_cast<int>(channels),
                        x_buffer, y_buffer, z_buffer);

    // Launches the kernel
    auto m_ceiled = Ceil(m, db_["COPY_DIMX"]);
    auto n_ceiled = Ceil(m, db_["COPY_DIMY"]*db_["COPY_WPT"]);
    auto global = std::vector<size_t>{m_ceiled, n_ceiled/db_["COPY_WPT"]};
    auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtransform_c<int16_t, int16_t>;
template class Xtransform_c<int32_t, int32_t>;
template class Xtransform_c<int64_t, int64_t>;
template class Xtransform_c<half,    half>;
template class Xtransform_c<float,   float>;
template class Xtransform_c<double,  double>;
template class Xtransform_c<float2,  float2>;
template class Xtransform_c<double2, double2>;

template class Xtransform_c<int16_t, bool>;
template class Xtransform_c<int32_t, bool>;
template class Xtransform_c<int64_t, bool>;
template class Xtransform_c<half,    bool>;
template class Xtransform_c<float,   bool>;
template class Xtransform_c<double,  bool>;

}}