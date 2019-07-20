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
                    int kernel_index = kernel_h*kernel_w - kw_id - kernel_w*kh_id - 1;
                    int output_index = ((c_id*kernel_h*kernel_w + kernel_index)*output_h + h_id)*output_w + w_id;
                    y_buffer[output_index] = val;
                };
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
                        m, n, k, T{1},
                        w_buffer, W.stride(0),
                        work.data(), work.stride(0), T{0},
                        y_buffer, Y.stride(1));

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
                       filter.height(), filter.width(), filter.output_h(), filter.output_w(),
                       filter.num_kernels(), filter.group(), filter.kernel_h(), filter.kernel_w(),
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
    if (p == 2) {
        detail::pooling(X.data(), Y.data(), filter, false,
                        T{},
                        [](auto acc, auto x) { return acc + x*x; },
                        [](auto acc, auto) { return std::sqrt(acc); });
    } else {
        detail::pooling(X.data(), Y.data(), filter, false,
                        T{},
                        [p](auto acc, auto x) { return acc + std::pow(std::abs(x), p); },
                        [p](auto acc, auto) { return std::pow(acc, T{1}/p); });
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
                       p,
                       X.data(), Y.data());
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
    // FIXME
    auto h = input.extent(2);
    auto w = input.extent(3);
    FilterShape2D filter(input.shape(), h, w);
    filter.strides(h, w);
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
    // FIXME
    auto h = input.extent(2);
    auto w = input.extent(3);
    FilterShape2D filter(input.shape(), h, w);
    filter.strides(h, w);
    avgpool(input, output, filter, false);
}

template <typename T>
void global_lppool(const Tensor<T>& X, Tensor<T>& Y, const int p) {
    if (p == 2) {
        detail::global_pooling(
            X, Y, T{},
            [](auto acc, auto x) { return acc + x*x; },
            std::plus<T>(),
            [](auto acc, auto) { return std::sqrt(acc); });
    } else {
        detail::global_pooling(
            X, Y, T{},
            [p](auto acc, auto x) { return acc + std::pow(std::abs(x), p); },
            std::plus<T>(),
            [p](auto acc, auto) { return std::pow(acc, T{1}/p); });
    }
}

template <typename T>
void global_lppool(const DevTensor<T>& input, DevTensor<T>& output, int p) {
    // FIXME
    auto h = input.extent(2);
    auto w = input.extent(3);
    FilterShape2D filter(input.shape(), h, w);
    filter.strides(h, w);
    lppool(input, output, filter, p);
}

template <typename T>
void softmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    auto dims = X.shape().extents();
    auto M = std::accumulate(dims.begin(), dims.begin()+axis, size_t(1), std::multiplies<>());
    auto N = X.size() / M;

    assert(Y.shape() == X.shape() || Y.shape() == Shape({M, N}));

    size_t grainsize = std::max(size_t(1), GRAINSIZE / N);
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, M, grainsize), [=](const auto& r) {
        for (int b = r.begin(); b < r.end(); b++) {
            auto px = x_buffer + b*N;
            auto py = y_buffer + b*N;

            T amax = px[0];
            for (size_t i = 1; i < N; i++) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < N; i++) {
                py[i] = std::exp(px[i] - amax);
                asum += py[i];
            }
            for (size_t i = 0; i < N; i++) {
                py[i] /= asum;
            }
        }
    });
}

template <typename T>
void softmax(const DevTensor<T>& X, DevTensor<T>& Y, int axis = 1) {
    // FIXME
    auto temp = X.read();
    softmax(temp, temp, axis);
    Y.write(temp);
}

template <typename TensorT>
enable_if_tensor<TensorT> softmax(TensorT&& X, int axis = 1, bool keepdims = true) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");

    tensor_type<TensorT> Y;
    if (std::is_rvalue_reference<decltype(X)>::value) {
        Y = std::move(X);
        softmax(Y, Y, axis);
    } else {
        Y = tensor_type<TensorT>(X.shape());
        softmax(X, Y, axis);
    }

    if (!keepdims) {
        auto dims = Y.shape().extents();
        int m = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
        int n = Y.size() / m;
        Y.reshape({m, n});
    }

    return Y;
}

}} // namespace dlf::dnn
