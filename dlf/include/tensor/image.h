#pragma once

namespace dlf { namespace im {

namespace detail {
template <typename T>
void resize1d(const size_t batch_count,
              const T* input, const Shape& input_shape,
              T* output, const Shape& output_shape)
{
    const auto output_length = output_shape.extent(-1);

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, batch_count, 1, 0, output_length, GRAINSIZE),
        [&](const auto& r) {
            const int input_length = input_shape.extent(-1);
            const int input_inc    = input_shape.stride(-1);
            const int output_inc   = output_shape.stride(-1);
            const int scale        = static_cast<float>(input_length) / output_length;

            for (int batch = r.rows().begin(); batch < r.rows().end(); ++batch) {
                const auto p_in  = input + input_shape.linear_offset(batch * input_length);
                const auto p_out = output + output_shape.linear_offset(batch * output_length);

                for (int idx = r.cols().begin(); idx < r.cols().end(); ++idx) {
                    const float x = (idx + 0.5f) * scale - 0.5f;
                    const int   i = static_cast<int>(std::floor(x));
                    const float a = x == static_cast<float>(i) ? 1.f : x - i;

                    auto q0 = p_in[std::max(0, i) * input_inc];
                    auto q1 = p_in[std::min(i+1, input_length-1) * input_inc];
                    p_out[idx * output_inc] = q0 + a*(q1 - q0);
                }
            }
        });
}

template <typename T>
void resize2d(const size_t batch_count,
              const T* input, const Shape& input_shape,
              T* output, const Shape& output_shape)
{
    const auto output_height = output_shape.extent(-2);
    const auto output_width  = output_shape.extent(-1);

    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, batch_count, 1, 0, output_height*output_width, GRAINSIZE),
        [&](const auto& r) {
            const int input_height  = input_shape.extent(-2);
            const int input_width   = input_shape.extent(-1);
            const int input_stride  = input_shape.stride(-2);
            const int input_inc     = input_shape.stride(-1);
            const int output_stride = output_shape.stride(-2);
            const int output_inc    = output_shape.stride(-1);

            const auto x_scale = static_cast<float>(input_width) / output_width;
            const auto y_scale = static_cast<float>(input_height) / output_height;

            for (int batch = r.rows().begin(); batch < r.rows().end(); ++batch) {
                const auto p_in  = input + input_shape.linear_offset(batch * input_height*input_width);
                const auto p_out = output + output_shape.linear_offset(batch * output_height*output_width);

                for (int id = r.cols().begin(); id < r.cols().end(); ++id) {
                    const auto idy = id / output_width;
                    const auto idx = id % output_width;

                    const float x = (idx + 0.5f) * x_scale - 0.5f;
                    const int   i = static_cast<int>(std::floor(x));
                    const float a = x == static_cast<float>(i) ? 1.f : x - i;

                    const float y = (idy + 0.5f) * y_scale - 0.5f;
                    const int   j = static_cast<int>(std::floor(y));
                    const float b = y == static_cast<float>(j) ? 1.f : y - j;

                    auto i0 = std::max(0, i) * input_inc;
                    auto i1 = std::min(i+1, input_width-1) * input_inc;
                    auto j0 = std::max(0, j) * input_stride;
                    auto j1 = std::min(j+1, input_height-1) * input_stride;

                    auto q00 = p_in[i0 + j0];
                    auto q10 = p_in[i1 + j0];
                    auto q01 = p_in[i0 + j1];
                    auto q11 = p_in[i1 + j1];

                    auto r0 = q00 + a*(q10 - q00);
                    auto r1 = q01 + a*(q11 - q01);
                    p_out[idy*output_stride + idx*output_inc] = r0 + b*(r1 - r0);
                }
            }
        });
}

template <typename T>
void resize1d(const size_t batch_count,
              const gpgpu::Buffer<T>& input, const Shape& input_shape,
              gpgpu::Buffer<T>& output, const Shape& output_shape)
{
    gpgpu::dnn::resize1d(batch_count,
        input, input_shape.offset(), input_shape.extents(), input_shape.strides(),
        output, output_shape.offset(), output_shape.extents(), output_shape.strides());
}

template <typename T>
void resize2d(const size_t batch_count,
              const gpgpu::Buffer<T>& input, const Shape& input_shape,
              gpgpu::Buffer<T>& output, const Shape& output_shape)
{
    gpgpu::dnn::resize2d(batch_count,
        input, input_shape.offset(), input_shape.extents(), input_shape.strides(),
        output, output_shape.offset(), output_shape.extents(), output_shape.strides());
}
} // namespace detail

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
resize(const TensorT& X, TensorR&& Y, const std::vector<float> scales) {
    if (X.rank() != scales.size())
        throw shape_error("resize: number of elements of `scales' must be same as rank of input");

    auto output_dims = std::vector<size_t>();
    auto transpose_perm = std::vector<size_t>();
    int  resize_rank = 0;

    for (size_t i = 0; i < scales.size(); ++i) {
        if (scales[i] <= 0)
            throw shape_error("resize: scale value must be greater than 0");
        output_dims.push_back(static_cast<size_t>(std::floor(X.extent(i) * scales[i])));
        if (scales[i] == 1)
            transpose_perm.push_back(i);
    }
    for (size_t i = 0; i < scales.size(); ++i) {
        if (scales[i] != 1) {
            transpose_perm.push_back(i);
            resize_rank++;
        }
    }

    Y.resize(Shape(output_dims));

    auto input_shape = X.shape().transpose(transpose_perm);
    auto output_shape = Y.shape().transpose(transpose_perm);
    auto batch_count = input_shape.partial_size(0, X.rank() - resize_rank);

    switch (resize_rank) {
    case 0:
        reorder(X, std::forward<TensorR>(Y));
        break;
    case 1:
        detail::resize1d(batch_count, X.data(), input_shape, Y.data(), output_shape);
        break;
    case 2:
        detail::resize2d(batch_count, X.data(), input_shape, Y.data(), output_shape);
        break;
    default:
        throw shape_error("resize: unsupported spatial rank");
    }
}

template <typename TensorT>
enable_if_tensor<TensorT>
resize(const TensorT& X, const std::vector<float>& scales) {
    tensor_type<TensorT> Y{};
    resize(X, Y, scales);
    return Y;
}

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
resize(const TensorT& X, TensorR&& Y, const Shape& shape) {
    if (X.rank() != shape.rank())
        throw shape_error("resize: rank of output shape must be same as input shape");

    auto transpose_perm = std::vector<size_t>();
    int  resize_rank = 0;

    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape.extent(i) == X.extent(i))
            transpose_perm.push_back(i);
    }
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape.extent(i) != X.extent(i)) {
            transpose_perm.push_back(i);
            resize_rank++;
        }
    }

    Y.resize(shape);

    auto input_shape = X.shape().transpose(transpose_perm);
    auto output_shape = Y.shape().transpose(transpose_perm);
    auto batch_count = input_shape.partial_size(0, X.rank() - resize_rank);

    switch (resize_rank) {
    case 0:
        reorder(X, std::forward<TensorR>(Y));
        break;
    case 1:
        detail::resize1d(batch_count, X.data(), input_shape, Y.data(), output_shape);
        break;
    case 2:
        detail::resize2d(batch_count, X.data(), input_shape, Y.data(), output_shape);
        break;
    default:
        throw shape_error("resize: unsupported spatial rank");
    }
}

template <typename TensorT>
enable_if_tensor<TensorT>
resize(const TensorT& X, const Shape& shape) {
    tensor_type<TensorT> Y{};
    resize(X, Y, shape);
    return Y;
}

}} // namespace dlf::im
