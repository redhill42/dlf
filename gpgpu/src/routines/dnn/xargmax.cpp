#include "xargmax.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xargmax<T>::Xargmax(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xargmax.cl"
    }){
}

template <typename T>
void Xargmax<T>::DoArgMax(const size_t m, const size_t k, const size_t n,
                          const Buffer<T>& x_buffer, Buffer<int>& y_buffer)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xargmax");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(m),
                        static_cast<int>(k),
                        static_cast<int>(n),
                        x_buffer, y_buffer);

    // Launches the kernel
    const auto m_ceiled = Ceil(m, db_["COPY_DIMX"]);
    const auto n_ceiled = Ceil(n, db_["COPY_DIMY"]);
    const auto global = std::vector<size_t>{m_ceiled, n_ceiled};
    const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xargmax<T>::DoArgMin(const size_t m, const size_t k, const size_t n,
                          const Buffer<T>& x_buffer, Buffer<int>& y_buffer)
{
    // Retrieves the kernel from the compiled binary
    auto kernel = program_.getKernel("Xargmin");

    // Sets the kernel arguments
    kernel.setArguments(static_cast<int>(m),
                        static_cast<int>(k),
                        static_cast<int>(n),
                        x_buffer, y_buffer);

    // Launches the kernel
    const auto m_ceiled = Ceil(m, db_["COPY_DIMX"]);
    const auto n_ceiled = Ceil(n, db_["COPY_DIMY"]);
    const auto global = std::vector<size_t>{m_ceiled, n_ceiled};
    const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xargmax<int16_t>;
template class Xargmax<int32_t>;
template class Xargmax<int64_t>;
template class Xargmax<half>;
template class Xargmax<float>;
template class Xargmax<double>;

}} // namespace gpgpu::dnn
