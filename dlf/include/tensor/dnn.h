#pragma once

namespace dlf { namespace dnn {

//---------------------------------------------------------------------------
// Filter shape used by convolution and pooling

class Filter2D {
    size_t m_batches, m_channels, m_height, m_width;
    size_t m_num_kernels, m_kernel_h, m_kernel_w, m_group;
    size_t m_pad_top, m_pad_left, m_pad_bottom, m_pad_right;
    size_t m_stride_h, m_stride_w;
    size_t m_dilation_h, m_dilation_w;

public:
    Filter2D(const Shape& input_shape, const Shape& kernel_shape, size_t group = 1);
    Filter2D(const Shape& input_shape, size_t kernel_h, size_t kernel_w);

    Filter2D& pads(size_t top, size_t left, size_t bottom, size_t right) noexcept {
        m_pad_top = top;
        m_pad_left = left;
        m_pad_bottom = bottom;
        m_pad_right = right;
        return *this;
    }

    Filter2D& pads(size_t h, size_t w) noexcept {
        m_pad_top = m_pad_bottom = h;
        m_pad_left = m_pad_right = w;
        return *this;
    }

    template <typename I>
    Filter2D& pads(const std::vector<I>& pads) noexcept {
        static_assert(std::is_convertible<I, size_t>::value, "");
        assert(pads.size() == 4);
        m_pad_top = pads[0];
        m_pad_left = pads[1];
        m_pad_bottom = pads[2];
        m_pad_right = pads[3];
        return *this;
    }

    Filter2D& auto_pad(const std::string& mode);

    Filter2D& strides(size_t h, size_t w) noexcept {
        m_stride_h = h;
        m_stride_w = w;
        return *this;
    }

    template <typename I>
    Filter2D& strides(const std::vector<I>& strides) noexcept {
        static_assert(std::is_convertible<I, size_t>::value, "");
        assert(strides.size() == 2);
        m_stride_h = strides[0];
        m_stride_w = strides[1];
        return *this;
    }

    Filter2D& dilations(size_t h, size_t w) noexcept {
        m_dilation_h = h;
        m_dilation_w = w;
        return *this;
    }

    template <typename I>
    Filter2D& dilations(const std::vector<I>& dilations) noexcept {
        static_assert(std::is_convertible<I, size_t>::value, "");
        assert(dilations.size() == 2);
        m_dilation_h = dilations[0];
        m_dilation_w = dilations[1];
        return *this;
    }

    size_t batches()     const noexcept { return m_batches; }
    size_t channels()    const noexcept { return m_channels; }
    size_t height()      const noexcept { return m_height; }
    size_t width()       const noexcept { return m_width; }
    size_t num_kernels() const noexcept { return m_num_kernels; }
    size_t kernel_h()    const noexcept { return m_kernel_h; }
    size_t kernel_w()    const noexcept { return m_kernel_w; }
    size_t group()       const noexcept { return m_group; }
    size_t pad_top()     const noexcept { return m_pad_top; }
    size_t pad_left()    const noexcept { return m_pad_left; }
    size_t pad_bottom()  const noexcept { return m_pad_bottom; }
    size_t pad_right()   const noexcept { return m_pad_right; }
    size_t pad_h()       const noexcept { return m_pad_top; }
    size_t pad_w()       const noexcept { return m_pad_left; }
    size_t stride_h()    const noexcept { return m_stride_h; }
    size_t stride_w()    const noexcept { return m_stride_w; }
    size_t dilation_h()  const noexcept { return m_dilation_h; }
    size_t dilation_w()  const noexcept { return m_dilation_w; }

    size_t output_h() const noexcept {
        auto size_h = height() + pad_top() + pad_bottom();
        auto padding_h = dilation_h() * (kernel_h() - 1) + 1;
        return (size_h >= padding_h) ? (size_h - padding_h) / stride_h() + 1 : 1;
    }

    size_t output_w() const noexcept {
        auto size_w = width() + pad_left() + pad_right();
        auto padding_w = dilation_w() * (kernel_w() - 1) + 1;
        return (size_w >= padding_w) ? (size_w - padding_w) / stride_w() + 1 : 1;
    }

    Shape input_shape() const noexcept {
        return Shape(batches(), channels(), height(), width());
    }

    Shape kernel_shape() const noexcept {
        return Shape(num_kernels(), channels()/group(), kernel_h(), kernel_w());
    }

    Shape output_shape() const noexcept {
        return Shape(batches(), num_kernels(), output_h(), output_w());
    }
};

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
void im2col(const T* x_buffer, T* y_buffer, const Filter2D& filter) {
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
void conv2d(const Tensor<T>& X, const Tensor<T>& W, Tensor<T>& Y, const Filter2D& filter,
            Tensor<T>* work = nullptr)
{
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    assert(Y.shape() == filter.output_shape());

    const auto group = filter.group();
    const auto m = filter.num_kernels() / group;
    const auto k = filter.channels() * filter.kernel_h() * filter.kernel_w() / group;
    const auto n = filter.output_h() * filter.output_w();

    Tensor<T> temp_buffer;
    if (work == nullptr) {
        temp_buffer = Tensor<T>({k, n});
        work = &temp_buffer;
    } else {
        work->reshape(k, n);
    }

    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    for (size_t b = 0; b < filter.batches(); b++) {
        auto w_buffer = W.data();
        for (size_t c = 0; c < group; c++) {
            im2col(x_buffer, work->data(), filter);

            cblas::gemm(cblas::Layout::RowMajor,
                        cblas::Transpose::NoTrans, cblas::Transpose::NoTrans,
                        m, n, k,
                        T{1}, w_buffer, W.stride(0),
                        work->data(), work->stride(0),
                        T{0}, y_buffer, Y.stride(1));

            x_buffer += X.stride(0) / group;
            y_buffer += Y.stride(0) / group;
            w_buffer += W.size() / group;
        }
    }
}

template <typename T>
size_t conv2dWorkspaceSize(const Tensor<T>&, const Tensor<T>&, const Filter2D& filter) {
    const auto group = filter.group();
    const auto k = filter.channels() * filter.kernel_h() * filter.kernel_w() / group;
    const auto n = filter.output_h() * filter.output_w();
    return k * n;
}

template <typename T>
void conv2d(const DevTensor<T>& X, const DevTensor<T>& W, DevTensor<T>& Y,
            const Filter2D& filter, DevTensor<T>* work = nullptr)
{
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
                       X.data(), W.data(), Y.data(),
                       work == nullptr ? nullptr : &work->data());
}

template <typename T>
size_t conv2dWorkspaceSize(const DevTensor<T>&, const DevTensor<T>&, const Filter2D& filter) {
    return gpgpu::dnn::conv2dWorkspaceSize<T>(
        filter.batches(), filter.channels(),
        filter.height(), filter.width(),
        filter.output_h(), filter.output_w(),
        filter.num_kernels(), filter.group(),
        filter.kernel_h(), filter.kernel_w(),
        filter.pad_top(), filter.pad_left(),
        filter.pad_bottom(), filter.pad_right(),
        filter.stride_h(), filter.stride_w(),
        filter.dilation_h(), filter.dilation_w());
}

namespace detail {
template <typename T, typename Accum, typename Reduce>
void pooling(const T* input, T* output,
             const Filter2D& filter,
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
void maxpool(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, false,
                    std::numeric_limits<T>::lowest(),
                    [](auto acc, auto x) { return std::max(acc, x); },
                    [](auto acc, auto)   { return acc; });
}

template <typename T>
void maxpool(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter) {
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
void avgpool(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    assert(Y.shape() == filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, count_include_pad,
                    T{}, std::plus<T>(), std::divides<>());
}

template <typename T>
void avgpool(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter, bool count_include_pad) {
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
void lppool(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter, const int p) {
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
void lppool(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter, int p) {
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
    auto filter = Filter2D(input.shape(), h, w).strides(h, w);
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
    auto filter = Filter2D(input.shape(), h, w).strides(h, w);
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
    auto filter = Filter2D(input.shape(), h, w).strides(h, w);
    lppool(input, output, filter, p);
}

template <typename T>
void softmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("softmax: invalid axis");
    Y.resize(X.shape());

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
    Y.resize(X.shape());

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    gpgpu::dnn::softmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> softmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        softmax(Y, Y, axis);
        return Y;
    } else {
        tensor_type<TensorT> Y{};
        softmax(X, Y, axis);
        return Y;
    }
}

template <typename T>
void logsoftmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("logsoftmax: invalid axis");
    Y.resize(X.shape());

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
    Y.resize(X.shape());

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    gpgpu::dnn::logsoftmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> logsoftmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        logsoftmax(Y, Y, axis);
        return Y;
    } else {
        tensor_type<TensorT> Y{};
        logsoftmax(X, Y, axis);
        return Y;
    }
}

template <typename T>
void hardmax(const Tensor<T>& X, Tensor<T>& Y, int axis = 1) {
    auto rank = X.rank();
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank)
        throw shape_error("hardmax: invalid axis");
    Y.resize(X.shape());

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
    Y.resize(X.shape());

    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    gpgpu::dnn::hardmax(m, n, X.data(), Y.data());
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT> hardmax(TensorT&& X, int axis = 1) {
    if (std::is_rvalue_reference<decltype(X)>::value) {
        auto Y = std::move(X);
        hardmax(Y, Y, axis);
        return Y;
    } else {
        tensor_type<TensorT> Y{};
        hardmax(X, Y, axis);
        return Y;
    }
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
space_to_depth(const TensorT& X, TensorT& Y, int blocksize) {
    if (blocksize <= 0)
        throw shape_error("space_to_depth: blocksize has incorrect value");
    if (X.rank() != 4)
        throw shape_error("space_to_depth: input tensor must be 4-dimensional");

    int n = X.extent(0), c = X.extent(1), h = X.extent(2), w = X.extent(3);
    if (h % blocksize != 0 || w % blocksize != 0)
        throw shape_error("space_to_depth: blocksize has incorrect value");

    auto x_shape = X.shape();
    x_shape = x_shape.reshape(n, c, h/blocksize, blocksize, w/blocksize, blocksize);
    x_shape = x_shape.transpose(0, 3, 5, 1, 2, 4);

    Y.resize(n, c*blocksize*blocksize, h/blocksize, w/blocksize);
    Y.reshape(n, blocksize, blocksize, c, h/blocksize, w/blocksize);
    reorder(X, x_shape, Y);
    Y.reshape(n, c*blocksize*blocksize, h/blocksize, w/blocksize);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
space_to_depth(const TensorT& X, int blocksize) {
    TensorT Y;
    space_to_depth(X, Y, blocksize);
    return Y;
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT, void>
depth_to_space(const TensorT& X, TensorT& Y, int blocksize, std::string mode = "DCR") {
    if (blocksize <= 0)
        throw shape_error("depth_to_space: blocksize has incorrect value");
    if (X.rank() != 4)
        throw shape_error("depth_to_space: input tensor must be 4-dimensional");
    if (mode != "DCR" && mode != "CRD")
        throw shape_error("depth_to_space: mode has incorrect value");

    int n = X.extent(0), c = X.extent(1), h = X.extent(2), w = X.extent(3);
    if (c % (blocksize*blocksize) != 0)
        throw shape_error("depth_to_space: blocksize has incorrect value");

    auto x_shape = X.shape();
    if (mode == "DCR") {
        x_shape = x_shape.reshape(n, blocksize, blocksize, c/(blocksize*blocksize), h, w);
        x_shape = x_shape.transpose(0, 3, 4, 1, 5, 2);
    } else {
        x_shape = x_shape.reshape(n, c/(blocksize*blocksize), blocksize, blocksize, h, w);
        x_shape = x_shape.transpose(0, 1, 4, 2, 5, 3);
    }

    Y.resize(n, c/(blocksize*blocksize), h*blocksize, w*blocksize);
    Y.reshape(n, c/(blocksize*blocksize), h, blocksize, w, blocksize);
    reorder(X, x_shape, Y);
    Y.reshape(n, c/(blocksize*blocksize), h*blocksize, w*blocksize);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
depth_to_space(const TensorT& X, int blocksize, std::string mode = "DCR") {
    TensorT Y;
    depth_to_space(X, Y, blocksize, mode);
    return Y;
}

}} // namespace dlf::dnn
