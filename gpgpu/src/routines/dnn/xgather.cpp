#include "xgather.hpp"

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xgather<T>::Xgather(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy", "Copy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xgather.cl"
}) {}

template <typename T>
void Xgather<T>::DoGather(
    const size_t m, const size_t n, const size_t chunk, const size_t max_item,
    const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
    const Buffer<int>& i_buffer, const size_t i_offset,
    const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
    Buffer<T>& y_buffer, const size_t y_offset)
{
    Kernel kernel;

    if (IsContiguous(x_dim, x_stride) &&
        IsContiguous(i_dim, i_stride) &&
        IsContiguous(y_dim, y_stride))
    {
        kernel = program_.getKernel("Xgather");
        kernel.setArguments(
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(chunk),
            static_cast<int>(max_item),
            x_buffer, static_cast<int>(x_offset),
            i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto x_shape_buffer = PackShape(x_dim, x_stride, context_, queue_);
        auto i_shape_buffer = PackShape(i_dim, i_stride, context_, queue_);
        auto y_shape_buffer = PackShape(y_dim, y_stride, context_, queue_);

        kernel = program_.getKernel("XgatherStrided");
        kernel.setArguments(
            static_cast<int>(m),
            static_cast<int>(n),
            static_cast<int>(chunk),
            static_cast<int>(max_item),
            static_cast<int>(x_dim.size()), x_shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            static_cast<int>(i_dim.size()), i_shape_buffer,
            i_buffer, static_cast<int>(i_offset),
            static_cast<int>(y_dim.size()), y_shape_buffer,
            y_buffer, static_cast<int>(y_offset));
    }

    const auto m_ceiled = Ceil(m, db_["COPY_DIMX"]);
    const auto n_ceiled = Ceil(n, db_["COPY_DIMY"]);
    const auto global = std::vector<size_t>{m_ceiled, n_ceiled};
    const auto local = std::vector<size_t>{db_["COPY_DIMX"], db_["COPY_DIMY"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xgather<T>::DoGatherElements(
    const size_t n, const int axis,
    const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
    const gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
    const gpgpu::Buffer<int>& i_buffer, const size_t i_offset,
    const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
    gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    assert(x_shape.size() == x_strides.size());
    assert(i_shape.size() == i_strides.size() && i_shape.size() == x_shape.size());
    assert(y_shape.size() == y_strides.size() && y_shape.size() == x_shape.size());

    int i_stride1 = 1, x_stride1 = 1;
    for (size_t i = axis+1; i < x_shape.size(); i++) {
        i_stride1 *= i_shape[i];
        x_stride1 *= x_shape[i];
    }
    int i_stride2 = i_stride1 * i_shape[axis];
    int x_stride2 = x_stride1 * x_shape[axis];

    Kernel kernel;

    if (IsContiguous(x_shape, x_strides) &&
        IsContiguous(i_shape, i_strides) &&
        IsContiguous(y_shape, y_strides))
    {
        kernel = program_.getKernel("Xgather_elements");
        kernel.setArguments(
            static_cast<int>(n),
            static_cast<int>(x_shape[axis]),
            i_stride1, i_stride2,
            x_stride1, x_stride2,
            x_buffer, static_cast<int>(x_offset),
            i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto rank = x_shape.size();
        auto x_shape_buffer = PackShape(x_shape, x_strides, context_, queue_);
        auto i_shape_buffer = PackShape(i_shape, i_strides, y_strides, context_, queue_);

        kernel = program_.getKernel("Xgather_elementsStrided");
        kernel.setArguments(
            static_cast<int>(n),
            static_cast<int>(x_shape[axis]),
            static_cast<int>(rank),
            i_stride1, i_stride2,
            x_stride1, x_stride2,
            x_shape_buffer, x_buffer, static_cast<int>(x_offset),
            i_shape_buffer, i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    }

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xgather<T>::DoScatterElements(
    const size_t n, const int axis,
    const std::vector<size_t>& x_shape, const std::vector<size_t>& x_strides,
    gpgpu::Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& i_shape, const std::vector<size_t>& i_strides,
    const gpgpu::Buffer<int>& i_buffer, const size_t i_offset,
    const std::vector<size_t>& y_shape, const std::vector<size_t>& y_strides,
    const gpgpu::Buffer<T>& y_buffer, const size_t y_offset)
{
    assert(x_shape.size() == x_strides.size());
    assert(i_shape.size() == i_strides.size() && i_shape.size() == x_shape.size());
    assert(y_shape.size() == y_strides.size() && y_shape.size() == x_shape.size());

    int i_stride1 = 1, x_stride1 = 1;
    for (size_t i = axis+1; i < x_shape.size(); i++) {
        i_stride1 *= i_shape[i];
        x_stride1 *= x_shape[i];
    }
    int i_stride2 = i_stride1 * i_shape[axis];
    int x_stride2 = x_stride1 * x_shape[axis];

    Kernel kernel;

    if (IsContiguous(x_shape, x_strides) &&
        IsContiguous(i_shape, i_strides) &&
        IsContiguous(y_shape, y_strides))
    {
        kernel = program_.getKernel("Xscatter_elements");
        kernel.setArguments(
            static_cast<int>(n),
            static_cast<int>(x_shape[axis]),
            i_stride1, i_stride2,
            x_stride1, x_stride2,
            x_buffer, static_cast<int>(x_offset),
            i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto rank = x_shape.size();
        auto x_shape_buffer = PackShape(x_shape, x_strides, context_, queue_);
        auto i_shape_buffer = PackShape(i_shape, i_strides, y_strides, context_, queue_);

        kernel = program_.getKernel("Xscatter_elementsStrided");
        kernel.setArguments(
            static_cast<int>(n),
            static_cast<int>(x_shape[axis]),
            static_cast<int>(rank),
            i_stride1, i_stride2,
            x_stride1, x_stride2,
            x_shape_buffer, x_buffer, static_cast<int>(x_offset),
            i_shape_buffer, i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    }

    auto n_ceiled = Ceil(n, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{n_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xgather<T>::DoGatherND(
    const size_t n, const size_t k, const size_t chunk,
    const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
    const Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
    const Buffer<int>& i_buffer, const size_t i_offset,
    const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
    Buffer<T>& y_buffer, const size_t y_offset)
{
    Kernel kernel;

    if (IsContiguous(x_dim, x_stride) &&
        IsContiguous(i_dim, i_stride) &&
        IsContiguous(y_dim, y_stride))
    {
        auto shape_buffer = PackShape(x_dim, x_stride, context_, queue_);
        kernel = program_.getKernel("Xgather_nd");
        kernel.setArguments(
            static_cast<int>(k), static_cast<int>(chunk),
            static_cast<int>(x_dim.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto x_shape_buffer = PackShape(x_dim, x_stride, context_, queue_);
        auto i_shape_buffer = PackShape(i_dim, i_stride, context_, queue_);
        auto y_shape_buffer = PackShape(y_dim, y_stride, context_, queue_);
        kernel = program_.getKernel("Xgather_ndStrided");
        kernel.setArguments(
            static_cast<int>(k), static_cast<int>(chunk),
            static_cast<int>(x_dim.size()), x_shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            static_cast<int>(i_dim.size()), i_shape_buffer,
            i_buffer, static_cast<int>(i_offset),
            static_cast<int>(y_dim.size()), y_shape_buffer,
            y_buffer, static_cast<int>(y_offset));
    }

    const auto global = std::vector<size_t>{n};
    const auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xgather<T>::DoScatterND(
    const size_t n, const size_t k, const size_t chunk,
    const std::vector<size_t>& x_dim, const std::vector<size_t>& x_stride,
    Buffer<T>& x_buffer, const size_t x_offset,
    const std::vector<size_t>& i_dim, const std::vector<size_t>& i_stride,
    const Buffer<int>& i_buffer, const size_t i_offset,
    const std::vector<size_t>& y_dim, const std::vector<size_t>& y_stride,
    const Buffer<T>& y_buffer, const size_t y_offset)
{
    Kernel kernel;

    if (IsContiguous(x_dim, x_stride) &&
        IsContiguous(i_dim, i_stride) &&
        IsContiguous(y_dim, y_stride))
    {
        auto shape_buffer = PackShape(x_dim, x_stride, context_, queue_);
        kernel = program_.getKernel("Xscatter_nd");
        kernel.setArguments(
            static_cast<int>(k), static_cast<int>(chunk),
            static_cast<int>(x_dim.size()), shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            i_buffer, static_cast<int>(i_offset),
            y_buffer, static_cast<int>(y_offset));
    } else {
        auto x_shape_buffer = PackShape(x_dim, x_stride, context_, queue_);
        auto i_shape_buffer = PackShape(i_dim, i_stride, context_, queue_);
        auto y_shape_buffer = PackShape(y_dim, y_stride, context_, queue_);
        kernel = program_.getKernel("Xscatter_ndStrided");
        kernel.setArguments(
            static_cast<int>(k), static_cast<int>(chunk),
            static_cast<int>(x_dim.size()), x_shape_buffer,
            x_buffer, static_cast<int>(x_offset),
            static_cast<int>(i_dim.size()), i_shape_buffer,
            i_buffer, static_cast<int>(i_offset),
            static_cast<int>(y_dim.size()), y_shape_buffer,
            y_buffer, static_cast<int>(y_offset));
    }

    const auto global = std::vector<size_t>{n};
    const auto local = std::vector<size_t>{1};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xgather<T>::DoGatherIndices(
    const size_t m, const size_t n, const bool row_major,
    const std::vector<size_t>& dims,
    const Buffer<int>& indices, const size_t indices_offset,
    Buffer<int>& output, const size_t output_offset)
{
    std::vector<int> int_dims(dims.begin(), dims.end());
    auto dims_buffer = context_.getSharedBuffer<int>(int_dims.data(), int_dims.size(), queue_);

    auto kernel = program_.getKernel("Xgather_indices");
    kernel.setArguments(
        static_cast<int>(m), static_cast<int>(n), static_cast<int>(row_major),
        static_cast<int>(dims.size()), dims_buffer,
        indices, static_cast<int>(indices_offset),
        output, static_cast<int>(output_offset));

    auto m_ceiled = Ceil(m, db_["WGS"]*db_["WPT"]);
    auto global = std::vector<size_t>{m_ceiled/db_["WPT"]};
    auto local = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xgather<half>;
template class Xgather<float>;
template class Xgather<double>;
template class Xgather<float2>;
template class Xgather<double2>;
template class Xgather<int32_t>;
template class Xgather<int64_t>;

}} // namespace gpgpu::dnn
