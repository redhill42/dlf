#pragma once

namespace dlf { namespace dnn {

template <typename T>
void batch_norm(const Tensor<T>& X, Tensor<T>& Y,
                const Tensor<T>& scale, const Tensor<T>& bias,
                const Tensor<T>& mean, const Tensor<T>& var,
                const T epsilon = T(1e-5))
{
    assert(X.shape() == Y.shape());
    auto batches  = X.extent(0);
    auto channels = X.extent(1);
    auto spatial  = X.size() / (batches * channels);

    assert(scale.is_vector() && scale.extent(0) == channels);
    assert(bias.is_vector() && bias.extent(0) == channels);
    assert(mean.is_vector() && mean.extent(0) == channels);
    assert(var.is_vector() && var.extent(0) == channels);

    const T* x = X.data();
          T* y = Y.data();
    const T* s = scale.data();
    const T* b = bias.data();
    const T* m = mean.data();

    T* v = reinterpret_cast<T*>(alloca(channels * sizeof(T)));
    std::transform(var.begin(), var.end(), v, [=](auto x) {
        return std::sqrt(x + epsilon);
    });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, spatial, 256), [=](auto& r) {
        for (size_t bat = 0; bat < batches; bat++) {
            for (size_t c = 0; c < channels; c++) {
                auto offset = (bat * channels + c) * spatial + r.begin();
                auto px = x + offset;
                auto py = y + offset;
                for (auto n = r.size(); n--; ) {
                    *py++ = s[c] * (*px++ - m[c]) / v[c] + b[c];
                }
            }
        }
    });
}

template <typename T>
void batch_norm(const DevTensor<T>& X, DevTensor<T>& Y,
                const DevTensor<T>& scale, const DevTensor<T>& bias,
                const DevTensor<T>& mean, const DevTensor<T>& var,
                const T epsilon = T(1e-5))
{
    assert(X.shape() == Y.shape());
    assert(scale.is_vector() && scale.extent(0) == X.extent(1));
    assert(bias.is_vector() && bias.extent(0) == X.extent(1));
    assert(mean.is_vector() && mean.extent(0) == X.extent(1));
    assert(var.is_vector() && var.extent(0) == X.extent(1));

    gpgpu::dnn::batch_norm(X.shape().extents(), X.data(), Y.data(),
                           scale.data(), bias.data(), mean.data(),
                           var.data(), epsilon);
}

template <typename T>
void lrn(const Tensor<T>& X, Tensor<T>& Y, const int n,
         const T alpha = 0.00001, const T beta = 0.75, const T bias = 1.0)
{
    assert(X.shape() == Y.shape());
    assert(n > 0);

    tbb::parallel_for(tbb::blocked_range<int>(0, X.stride(1), 256), [=, &X, &Y](auto r) {
        const int B = X.extent(0);
        const int N = X.extent(1);
        const int S = X.stride(1);

        auto x_buffer = X.data();
        auto y_buffer = Y.data();

        for (int b = 0; b < B; b++) {
            for (int i = 0; i < N; i++) {
                auto offset = (b * N + i) * S + r.begin();
                auto px = x_buffer + offset;
                auto py = y_buffer + offset;
                const int L = std::max(0, i - (n-1)/2);
                const int H = std::min(N-1, i + n/2);
                for (int count = r.size(); --count >= 0; ++px, ++py) {
                    T val{};
                    for (int j = L; j <= H; j++) {
                        auto x = px[(j-i)*S];
                        val += x*x;
                    }
                    *py = *px * std::pow(alpha*val/n + bias, -beta);
                }
            }
        }
    });
}

template <typename T>
void lrn(const DevTensor<T>& X, DevTensor<T>& Y, const int nsize,
         const T alpha = 0.00001, const T beta = 0.75, const T bias = 1.0)
{
    assert(X.shape() == Y.shape());
    assert(nsize > 0);
    gpgpu::dnn::lrn(X.shape().extents(), X.data(), Y.data(), nsize, alpha, beta, bias);
}

template <typename T>
void im2col(const T* x_buffer, T* y_buffer, const FilterShape2D& filter) {
    tbb::parallel_for(
        tbb::blocked_range2d<int>(
            0, filter.output_w(), 32,
            0, filter.output_h()*filter.channels()/filter.group(), 32),
        [=, &filter](auto r) {
            const auto input_h    = filter.height();
            const auto input_w    = filter.width();
            const auto output_h   = filter.output_h();
            const auto output_w   = filter.output_w();
            const auto kernel_h   = filter.kernel_h();
            const auto kernel_w   = filter.kernel_w();
            const auto pad_h      = filter.pad_h();
            const auto pad_w      = filter.pad_w();
            const auto stride_h   = filter.stride_h();
            const auto stride_w   = filter.stride_w();
            const auto dilation_h = filter.dilation_h();
            const auto dilation_w = filter.dilation_w();

            for (int w_id = r.rows().begin(); w_id < r.rows().end(); w_id++)
            for (int hc_id = r.cols().begin(); hc_id < r.cols().end(); hc_id++) {
                int c_id = hc_id / output_h;
                int h_id = hc_id - c_id * output_h;

                for (int kh_id = 0; kh_id < kernel_h; kh_id++)
                for (int kw_id = 0; kw_id < kernel_w; kw_id++) {
                    // Retrieves the input value.
                    int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
                    int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
                    T val{};
                    if (h_index >= 0 && h_index < input_h &&
                        w_index >= 0 && w_index < input_w) {
                        int input_index = (c_id * input_h + h_index) * input_w + w_index;
                        val = x_buffer[input_index];
                    }

                    // Sets the output value
                    int kernel_index = kw_id + kernel_w * kh_id;
                    int output_index = ((c_id*kernel_h*kernel_w + kernel_index)*output_h + h_id)*output_w + w_id;
                    y_buffer[output_index] = val;
                }
            }
        });
}

template <typename T>
void conv2d(const Tensor<T>& X, const Tensor<T>& W, Tensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    assert(Y.shape() == filter.output_shape());

    const auto group = filter.group();
    const auto m = filter.num_kernels() / group;
    const auto k = filter.channels() * filter.kernel_h() * filter.kernel_w() / group;
    const auto n = filter.output_h() * filter.output_w();
    Tensor<T> work({k, n});

    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    for (size_t b = 0; b < filter.batches(); b++) {
        auto w_buffer = W.data();
        for (size_t c = 0; c < group; c++) {
            im2col(x_buffer, work.data(), filter);

            cblas::gemm(cblas::Layout::RowMajor,
                        cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
                        m, n, k,
                        T{1}, w_buffer, W.stride(0),
                        work.data(), work.stride(0),
                        T{0}, y_buffer, Y.stride(1));

            x_buffer += X.stride(0) / group;
            y_buffer += Y.stride(0) / group;
            w_buffer += W.size() / group;
        }
    }
}

template <typename T>
void conv2d(const DevTensor<T>& X, const DevTensor<T>& W, DevTensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    assert(Y.shape() == filter.output_shape());
    gpgpu::dnn::conv2d(filter.batches(), filter.channels(),
                       filter.height(), filter.width(),
                       filter.output_h(), filter.output_w(),
                       filter.num_kernels(), filter.group(),
                       filter.kernel_h(), filter.kernel_w(),
                       filter.pad_top(), filter.pad_left(),
                       filter.pad_bottom(), filter.pad_right(),
                       filter.stride_h(), filter.stride_w(),
                       filter.dilation_h(), filter.dilation_w(),
                       X.data(), W.data(), Y.data());
}

namespace detail {
template <typename T, typename Accum, typename Reduce>
void pooling(const T* input, T* output,
             const FilterShape2D& filter,
             const bool count_include_pad,
             const T identity, Accum accum, Reduce reduce)
{
    tbb::parallel_for(
        tbb::blocked_range3d<int>(
            0, filter.batches(), 1,
            0, filter.output_w(), 32,
            0, filter.output_h()*filter.channels(), 32
        ), [=, &filter](auto r) {
        auto channels   = filter.channels();
        auto input_h    = filter.height();
        auto input_w    = filter.width();
        auto output_h   = filter.output_h();
        auto output_w   = filter.output_w();
        auto kernel_h   = filter.kernel_h();
        auto kernel_w   = filter.kernel_w();
        auto pad_h      = filter.pad_h();
        auto pad_w      = filter.pad_w();
        auto stride_h   = filter.stride_h();
        auto stride_w   = filter.stride_w();
        auto dilation_h = filter.dilation_h();
        auto dilation_w = filter.dilation_w();

        for (int b_id = r.pages().begin(); b_id < r.pages().end(); b_id++) {
            for (int w_id = r.rows().begin(); w_id < r.rows().end(); w_id++) {
                for (int hc_id = r.cols().begin(); hc_id < r.cols().end(); hc_id++) {
                    int c_id = hc_id / output_h;
                    int h_id = hc_id - c_id * output_h;

                    T acc = identity; int count = 0;
                    for (int kh_id = 0; kh_id < kernel_h; kh_id++) {
                        int h_index = kh_id * dilation_h + stride_h * h_id - pad_h;
                        if (h_index >= 0 && h_index < input_h) {
                            for (int kw_id = 0; kw_id < kernel_w; kw_id++) {
                                int w_index = kw_id * dilation_w + stride_w * w_id - pad_w;
                                if (w_index >= 0 && w_index < input_w) {
                                    int input_index =
                                        ((b_id * channels + c_id) * input_h + h_index) * input_w + w_index;
                                    acc = accum(acc, input[input_index]);
                                    count++;
                                }
                            }
                        }
                    }

                    int output_index = ((b_id * channels + c_id) * output_h + h_id) * output_w + w_id;
                    if (count_include_pad) count = kernel_h*kernel_w;
                    output[output_index] = reduce(acc, count);
                }
            }
        }
    });
}

template <typename T, typename Accum, typename Join, typename Reduce>
void global_pooling(const Tensor<T>& X, Tensor<T>& Y, const T identity,
                    Accum accum, Join join, Reduce reduce)
{
    assert(X.rank() >= 3);
    assert(X.rank() == Y.rank());
    auto M = X.extent(0) * X.extent(1);
    auto N = X.size() / M;
    assert(Y.size() == M);

    size_t grainsize = std::max(size_t(1), GRAINSIZE / N);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, M, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto val = tbb::parallel_reduce(
                tbb::blocked_range<int>(0, N, grainsize),
                identity,
                [=](auto r, T acc) {
                    auto px = x_buffer + b*N + r.begin();
                    for (size_t k = r.size(); k-- != 0; )
                        acc = accum(acc, *px++);
                    return acc;
                },
                join);
            y_buffer[b] = reduce(val, N);
        }
    });
}
} // namespace detail

template <typename T>
void maxpool(const Tensor<T>& X, Tensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, false,
                    std::numeric_limits<T>::lowest(),
                    [](auto acc, auto x) { return std::max(acc, x); },
                    [](auto acc, auto)   { return acc; });
}

template <typename T>
void maxpool(const DevTensor<T>& X, DevTensor<T>& Y, const FilterShape2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    gpgpu::dnn::maxpool(filter.batches(), filter.channels(),
                        filter.height(), filter.width(),
                        filter.output_h(), filter.output_w(),
                        filter.kernel_h(), filter.kernel_w(),
                        filter.pad_top(), filter.pad_left(),
                        filter.pad_bottom(), filter.pad_right(),
                        filter.stride_h(), filter.stride_w(),
                        filter.dilation_h(), filter.dilation_w(),
                        X.data(), Y.data());
}

template <typename T>
void avgpool(const Tensor<T>& X, Tensor<T>& Y, const FilterShape2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, count_include_pad,
                    T{}, std::plus<T>(), std::divides<>());
}

template <typename T>
void avgpool(const DevTensor<T>& X, DevTensor<T>& Y, const FilterShape2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    gpgpu::dnn::avgpool(filter.batches(), filter.channels(),
                        filter.height(), filter.width(),
                        filter.output_h(), filter.output_w(),
                        filter.kernel_h(), filter.kernel_w(),
                        filter.pad_top(), filter.pad_left(),
                        filter.pad_bottom(), filter.pad_right(),
                        filter.stride_h(), filter.stride_w(),
                        filter.dilation_h(), filter.dilation_w(),
                        count_include_pad,
                        X.data(), Y.data());
}

template <typename T>
void lppool(const Tensor<T>& X, Tensor<T>& Y, const FilterShape2D& filter, const int p) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    assert(p > 0);

    switch (p) {
    case 1:
        detail::pooling(
            X.data(), Y.data(), filter, false,
            T{},
            [](auto acc, auto x) { return acc + std::abs(x); },
            [](auto acc, auto  ) { return acc; });
        break;

    case 2:
        detail::pooling(
            X.data(), Y.data(), filter, false,
            T{},
            [](auto acc, auto x) { return acc + x*x; },
            [](auto acc, auto  ) { return std::sqrt(acc); });
        break;

    case 3:
        detail::pooling(
            X.data(), Y.data(), filter, false,
            T{},
            [](auto acc, auto x) { return acc + std::abs(x*x*x); },
            [](auto acc, auto  ) { return std::cbrt(acc); });
        break;

    default:
        detail::pooling(
            X.data(), Y.data(), filter, false,
            T{},
            [p](auto acc, auto x) { return acc + std::pow(std::abs(x), p); },
            [p](auto acc, auto  ) { return std::pow(acc, T{1}/p); });
        break;
    }
}

template <typename T>
void lppool(const DevTensor<T>& X, DevTensor<T>& Y, const FilterShape2D& filter, int p) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    gpgpu::dnn::lppool(filter.batches(), filter.channels(),
                       filter.height(), filter.width(),
                       filter.output_h(), filter.output_w(),
                       filter.kernel_h(), filter.kernel_w(),
                       filter.pad_top(), filter.pad_left(),
                       filter.pad_bottom(), filter.pad_right(),
                       filter.stride_h(), filter.stride_w(),
                       filter.dilation_h(), filter.dilation_w(),
                       p, X.data(), Y.data());
}

template <typename T>
void global_maxpool(const Tensor<T>& X, Tensor<T>& Y) {
    detail::global_pooling(
        X, Y, std::numeric_limits<T>::lowest(),
        [](auto acc, auto x){ return std::max(acc, x); },
        [](auto x, auto y)  { return std::max(x, y); },
        [](auto acc, auto)  { return acc; });
}

template <typename T>
void global_maxpool(const DevTensor<T>& input, DevTensor<T>& output) {
    auto h = input.extent(2), w = input.extent(3);
    auto filter = FilterShape2D(input.shape(), h, w).strides(h, w);
    maxpool(input, output, filter);
}

template <typename T>
void global_avgpool(const Tensor<T>& X, Tensor<T>& Y) {
    detail::global_pooling(
        X, Y, T{},
        std::plus<T>(),
        std::plus<T>(),
        [](auto acc, auto n){ return acc / n; });
}

template <typename T>
void global_avgpool(const DevTensor<T>& input, DevTensor<T>& output) {
    auto h = input.extent(2), w = input.extent(3);
    auto filter = FilterShape2D(input.shape(), h, w).strides(h, w);
    avgpool(input, output, filter, false);
}

template <typename T>
void global_lppool(const Tensor<T>& X, Tensor<T>& Y, const int p) {
    assert(p > 0);

    switch (p) {
    case 1:
        detail::global_pooling(
            X, Y, T{},
            [](auto acc, auto x) { return acc + std::abs(x); },
            std::plus<T>(),
            [](auto acc, auto  ) { return acc; });
        break;

    case 2:
        detail::global_pooling(
            X, Y, T{},
            [](auto acc, auto x) { return acc + x*x; },
            std::plus<T>(),
            [](auto acc, auto  ) { return std::sqrt(acc); });
        break;

    case 3:
        detail::global_pooling(
            X, Y, T{},
            [](auto acc, auto x) { return acc + std::abs(x*x*x); },
            std::plus<T>(),
            [](auto acc, auto  ) { return std::cbrt(acc); });
        break;

    default:
        detail::global_pooling(
            X, Y, T{},
            [p](auto acc, auto x) { return acc + std::pow(std::abs(x), p); },
            std::plus<T>(),
            [p](auto acc, auto  ) { return std::pow(acc, T{1}/p); });
        break;
    }
}

template <typename T>
void global_lppool(const DevTensor<T>& input, DevTensor<T>& output, int p) {
    auto h = input.extent(2), w = input.extent(3);
    auto filter = FilterShape2D(input.shape(), h, w).strides(h, w);
    lppool(input, output, filter, p);
}

template <typename T>
void softmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    assert(Y.shape() == X.shape());

    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    const auto m = X.shape().partial_size(0, axis);
    const auto n = X.size() / m;

    size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto px = x_buffer + b*n;
            auto py = y_buffer + b*n;

            T amax = px[0];
            for (size_t i = 1; i < n; i++) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < n; i++) {
                py[i] = std::exp(px[i] - amax);
                asum += py[i];
            }
            for (size_t i = 0; i < n; i++) {
                py[i] /= asum;
            }
        }
    });
}

template <typename T>
void softmax(const DevTensor<T>& X, DevTensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    assert(Y.shape() == X.shape() );
    gpgpu::dnn::softmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_tensor<TensorT> softmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        softmax(Y, Y, axis);
        return Y;
    } else {
        auto Y = tensor_type<TensorT>(X.shape());
        softmax(X, Y, axis);
        return Y;
    }
}

template <typename T>
void logsoftmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    assert(Y.shape() == X.shape());

    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("logsoftmax: invalid axis");

    const auto m = X.shape().partial_size(0, axis);
    const auto n = X.size() / m;

    size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto px = x_buffer + b*n;
            auto py = y_buffer + b*n;

            T amax = px[0];
            for (size_t i = 1; i < n; i++) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < n; i++)
                asum += std::exp(px[i] - amax);
            asum = std::log(asum);

            for (size_t i = 0; i < n; i++) {
                py[i] = px[i] - amax - asum;
            }
        }
    });
}

template <typename T>
void logsoftmax(const DevTensor<T>& X, DevTensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("logsoftmax: invalid axis");

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    assert(Y.shape() == X.shape());
    gpgpu::dnn::logsoftmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_tensor<TensorT> logsoftmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        logsoftmax(Y, Y, axis);
        return Y;
    } else {
        auto Y = tensor_type<TensorT>(X.shape());
        logsoftmax(X, Y, axis);
        return Y;
    }
}

template <typename T>
void hardmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    assert(Y.shape() == X.shape());

    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("hardmax: invalid axis");

    const auto m = X.shape().partial_size(0, axis);
    const auto n = X.size() / m;

    size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto px = x_buffer + b*n;
            auto py = y_buffer + b*n;

            T amax = px[0];
            int imax = 0;
            for (size_t i = 0; i < n; i++) {
                if (px[i] > amax) {
                    amax = px[i];
                    imax = i;
                }
                py[i] = 0;
            }
            py[imax] = 1;
        }
    });
}

template <typename T>
void hardmax(const DevTensor<T>& X, DevTensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    assert(Y.shape() == X.shape());
    gpgpu::dnn::hardmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_tensor<TensorT> hardmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        hardmax(Y, Y, axis);
        return Y;
    } else {
        auto Y = tensor_type<TensorT>(X.shape());
        hardmax(X, Y, axis);
        return Y;
    }
}

namespace detail {
inline int norm_axis(const char* name, int axis, size_t rank) {
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error(cxx::string_concat(name, ": invalid axis"));
    return axis;
}

inline bool check_reduced_shape(int axis, bool keepdims,
    const Shape& x_shape, const Shape& y_shape)
{
    auto x_dims = x_shape.extents();
    auto y_dims = y_shape.extents();

    if (keepdims) {
        if (x_dims.size() != y_dims.size())
            return false;
        if (y_dims[axis] != 1)
            return false;
        y_dims[axis] = x_dims[axis];
    } else {
        if (x_dims.size() != y_dims.size() + 1)
            return false;
        y_dims.insert(y_dims.begin()+axis, x_dims[axis]);
    }
    return x_dims == y_dims;
}

template <typename T, typename Compare>
void arg_reduce(
    const char* name, const Tensor<T>& X, Tensor<int>& Y,
    int axis, bool keepdims, Compare compare)
{
    axis = norm_axis(name, axis, X.rank());
    if (!check_reduced_shape(axis, keepdims, X.shape(), Y.shape()))
        throw shape_error(cxx::string_concat(name, ": incompatible output shape"));

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());

    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 16, 0, n, 64), [=](auto r) {
        for (int i = r.rows().begin(); i < r.rows().end(); i++) {
            for (int j = r.cols().begin(); j < r.cols().end(); j++) {
                auto px = x_buffer + i * k * n + j;
                T acc = *px;
                int index = 0;
                for (int ik = 1; ik < k; ik++) {
                    if (compare(px[ik*n], acc)) {
                        acc = px[ik*n];
                        index = ik;
                    }
                }
                y_buffer[i * n + j] = index;
            }
        }
    });
}
} // namespace detail

template <typename T>
void argmax(const Tensor<T>& X, Tensor<int>& Y, int axis, bool keepdims = true) {
    detail::arg_reduce("argmax", X, Y, axis, keepdims, std::greater<>());
}

template <typename T>
void argmin(const Tensor<T>& X, Tensor<int>& Y, int axis,bool keepdims = true) {
    detail:: arg_reduce("argmin", X, Y, axis, keepdims, std::less<>());
}

template <typename T>
void argmax(const DevTensor<T>& X, DevTensor<int>& Y, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmax", axis, X.rank());
    if (!detail::check_reduced_shape(axis, keepdims, X.shape(), Y.shape()))
        throw shape_error("argmax: incompatible output shape");

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());
    gpgpu::dnn::argmax(m, k, n, X.data(), Y.data());
}

template <typename T>
void argmin(const DevTensor<T>& X, DevTensor<int>& Y, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmin", axis, X.rank());
    if (!detail::check_reduced_shape(axis, keepdims, X.shape(), Y.shape()))
        throw shape_error("argmin: incompatible output shape");

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());
    gpgpu::dnn::argmin(m, k, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int>>
argmax(const TensorT& X, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmax", axis, X.rank());

    auto dims = X.shape().extents();
    if (keepdims) {
        dims[axis] = 1;
    } else {
        dims.erase(dims.begin() + axis);
    }

    auto Y = tensor_type<TensorT, int>(Shape(dims));
    argmax(X, Y, axis, keepdims);
    return Y;
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_type<TensorT, int>>
argmin(const TensorT& X, int axis, bool keepdims = true) {
    axis = detail::norm_axis("argmin", axis, X.rank());

    auto dims = X.shape().extents();
    if (keepdims) {
        dims[axis] = 1;
    } else {
        dims.erase(dims.begin() + axis);
    }

    auto Y = tensor_type<TensorT, int>(Shape(dims));
    argmin(X, Y, axis, keepdims);
    return Y;
}

}} // namespace dlf::dnn
