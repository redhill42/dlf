#pragma once

namespace dlf { namespace dnn {

//=--------------------------------------------------------------------------
// Filter descriptor used by convolution and pooling
//=--------------------------------------------------------------------------

class Filter2D {
    size_t m_batches, m_channels, m_height, m_width;
    size_t m_num_kernels, m_kernel_h, m_kernel_w, m_group;
    size_t m_pad_top, m_pad_left, m_pad_bottom, m_pad_right;
    size_t m_stride_h, m_stride_w;
    size_t m_dilation_h, m_dilation_w;

public:
    Filter2D(const Shape& input_shape, const Shape& kernel_shape, size_t group = 1);
    Filter2D(const Shape& input_shape, size_t kernel_h, size_t kernel_w);

    void set_shape(const Shape& input_shape, const Shape& kernel_shape, size_t group) noexcept;
    void set_shape(const Shape& input_shape) noexcept;

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

//=--------------------------------------------------------------------------
// Normalization
//=--------------------------------------------------------------------------

template <typename T>
void batch_norm(const Tensor<T>& X, Tensor<T>& Y,
                const Tensor<T>& scale, const Tensor<T>& bias,
                const Tensor<T>& mean, const Tensor<T>& var,
                const T epsilon = T(1e-5))
{
    assert(scale.is_vector() && scale.extent(0) == X.extent(1));
    assert(bias.is_vector()  && bias.extent(0)  == X.extent(1));
    assert(mean.is_vector()  && mean.extent(0)  == X.extent(1));
    assert(var.is_vector()   && var.extent(0)   == X.extent(1));

    Y.resize(X.shape());
    map([=](auto x, auto& y, auto s, auto b, auto m, auto v) {
        y = s * (x - m) / std::sqrt(v + epsilon) + b;
    })(X, Y, unsqueeze_right(scale, X.rank() - 1),
             unsqueeze_right(bias,  X.rank() - 1),
             unsqueeze_right(mean,  X.rank() - 1),
             unsqueeze_right(var,   X.rank() - 1));
}

template <typename T>
void batch_norm(const DevTensor<T>& X, DevTensor<T>& Y,
                const DevTensor<T>& scale, const DevTensor<T>& bias,
                const DevTensor<T>& mean, const DevTensor<T>& var,
                const T epsilon = T(1e-5))
{
    assert(scale.is_vector() && scale.extent(0) == X.extent(1));
    assert(bias.is_vector()  && bias.extent(0)  == X.extent(1));
    assert(mean.is_vector()  && mean.extent(0)  == X.extent(1));
    assert(var.is_vector()   && var.extent(0)   == X.extent(1));

    Y.resize(X.shape());
    gpgpu::dnn::batch_norm(X.shape().extents(), X.data(), Y.data(),
                           scale.data(), bias.data(), mean.data(),
                           var.data(), epsilon);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
batch_norm(const TensorT& X,
           const TensorT& scale, const TensorT& bias,
           const TensorT& mean, const TensorT& var,
           const tensor_value_type<TensorT> epsilon = 1e-5)
{
    TensorT Y{};
    batch_norm(X, Y, scale, bias, mean, var, epsilon);
    return Y;
}

template <typename T>
void lrn(const Tensor<T>& X, Tensor<T>& Y, const int n,
         const T alpha = 0.00001, const T beta = 0.75, const T bias = 1.0)
{
    assert(n > 0);
    Y.resize(X.shape());

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
    assert(nsize > 0);
    Y.resize(X.shape());
    gpgpu::dnn::lrn(X.shape().extents(), X.data(), Y.data(), nsize, alpha, beta, bias);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
lrn(const TensorT& X, const int nsize,
    const tensor_value_type<TensorT> alpha = 0.00001,
    const tensor_value_type<TensorT> beta = 0.75,
    const tensor_value_type<TensorT> bias = 1.0)
{
    TensorT Y{};
    lrn(X, Y, nsize, alpha, beta, bias);
    return Y;
}

//=--------------------------------------------------------------------------
// Convolution
//=--------------------------------------------------------------------------

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
void conv2d(const Tensor<T>& X, const Tensor<T>& W, Tensor<T>& Y, const Filter2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    Y.resize(filter.output_shape());

    const auto group = filter.group();
    const auto m = filter.num_kernels() / group;
    const auto k = filter.channels() * filter.kernel_h() * filter.kernel_w() / group;
    const auto n = filter.output_h() * filter.output_w();

    Tensor<T> work = Tensor<T>({k, n});
    auto x_buffer = X.data();
    auto y_buffer = Y.data();

    for (size_t b = 0; b < filter.batches(); b++) {
        auto w_buffer = W.data();
        for (size_t c = 0; c < group; c++) {
            im2col(x_buffer, work.data(), filter);

            detail::gemm(
                cblas::Transpose::NoTrans,
                cblas::Transpose::NoTrans,
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
void conv2d(const DevTensor<T>& X, const DevTensor<T>& W, DevTensor<T>& Y, const Filter2D& filter) {
    assert(X.shape() == filter.input_shape());
    assert(W.shape() == filter.kernel_shape());
    Y.resize(filter.output_shape());
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

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
conv2d(const TensorT& X, const TensorT& W, const Filter2D& filter) {
    TensorT Y{};
    conv2d(X, W, Y, filter);
    return Y;
}

//=--------------------------------------------------------------------------
// Pooling
//=--------------------------------------------------------------------------

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
    auto M = X.extent(0) * X.extent(1);
    auto N = X.size() / M;

    auto output_shape = X.shape().extents();
    std::fill(output_shape.begin()+2, output_shape.end(), 1);
    Y.resize(Shape{output_shape});

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
void max_pooling(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter) {
    assert(X.shape() == filter.input_shape());
    Y.resize(filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, false,
                    std::numeric_limits<T>::lowest(),
                    [](auto acc, auto x) { return std::max(acc, x); },
                    [](auto acc, auto)   { return acc; });
}

template <typename T>
void max_pooling(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter) {
    assert(X.shape() == filter.input_shape());
    Y.resize(filter.output_shape());
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

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
max_pooling(const TensorT& X, const Filter2D& filter) {
    tensor_type<TensorT> Y{};
    max_pooling(X, Y, filter);
    return Y;
}

template <typename T>
void average_pooling(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    Y.resize(filter.output_shape());
    detail::pooling(X.data(), Y.data(), filter, count_include_pad,
                    T{}, std::plus<T>(), std::divides<>());
}

template <typename T>
void average_pooling(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter, bool count_include_pad) {
    assert(X.shape() == filter.input_shape());
    Y.resize(filter.output_shape());
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

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
inline average_pooling(const TensorT& X, const Filter2D& filter, bool count_include_pad) {
    tensor_type<TensorT> Y{};
    average_pooling(X, Y, filter, count_include_pad);
    return Y;
}

template <typename T>
void lp_pooling(const Tensor<T>& X, Tensor<T>& Y, const Filter2D& filter, const int p) {
    assert(X.shape() == filter.input_shape());
    assert(p > 0);

    Y.resize(filter.output_shape());

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
void lp_pooling(const DevTensor<T>& X, DevTensor<T>& Y, const Filter2D& filter, int p) {
    assert(X.shape() == filter.input_shape());
    Y.resize(filter.output_shape());
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

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
lp_pooling(const TensorT& X, const Filter2D& filter, int p) {
    tensor_type<TensorT> Y{};
    lppool(X, Y, filter, p);
    return Y;
}

template <typename T>
void global_max_pooling(const Tensor<T>& X, Tensor<T>& Y) {
    detail::global_pooling(
        X, Y, std::numeric_limits<T>::lowest(),
        [](auto acc, auto x){ return std::max(acc, x); },
        [](auto x, auto y)  { return std::max(x, y); },
        [](auto acc, auto)  { return acc; });
}

template <typename T>
void global_max_pooling(const DevTensor<T>& input, DevTensor<T>& output) {
    auto h = input.extent(2), w = input.extent(3);
    auto filter = Filter2D(input.shape(), h, w).strides(h, w);
    max_pooling(input, output, filter);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
global_max_pooling(const TensorT& X) {
    TensorT Y{};
    global_max_pooling(X, Y);
    return Y;
}

template <typename T>
void global_average_pooling(const Tensor<T>& X, Tensor<T>& Y) {
    detail::global_pooling(
        X, Y, T{},
        std::plus<T>(),
        std::plus<T>(),
        [](auto acc, auto n){ return acc / n; });
}

template <typename T>
void global_average_pooling(const DevTensor<T>& X, DevTensor<T>& Y) {
    auto h = X.extent(2), w = X.extent(3);
    auto filter = Filter2D(X.shape(), h, w).strides(h, w);
    average_pooling(X, Y, filter, false);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
global_average_pooling(const TensorT& X) {
    TensorT Y{};
    global_average_pooling(X, Y);
    return Y;
}

template <typename T>
void global_lp_pooling(const Tensor<T>& X, Tensor<T>& Y, const int p) {
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
void global_lp_pooling(const DevTensor<T>& input, DevTensor<T>& output, int p) {
    auto h = input.extent(2), w = input.extent(3);
    auto filter = Filter2D(input.shape(), h, w).strides(h, w);
    lp_pooling(input, output, filter, p);
}

template <typename TensorT>
enable_if_non_view_tensor<TensorT>
global_lp_pooling(const TensorT& X, int p) {
    TensorT Y{};
    global_lp_pooling(X, Y, p);
    return Y;
}

//=--------------------------------------------------------------------------
// Normalize
//=--------------------------------------------------------------------------

namespace detail {
template <typename T>
void softmax(const size_t m, const size_t n, Tensor<T>& X) {
    const size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto buffer = X.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](auto r) {
        for (int b = r.begin(); b < r.end(); ++b) {
            auto px = buffer + b*n;

            T amax = px[0];
            for (size_t i = 1; i < n; ++i) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < n; ++i) {
                px[i] = std::exp(px[i] - amax);
                asum += px[i];
            }
            for (size_t i = 0; i < n; ++i) {
                px[i] /= asum;
            }
        }
    });
}

template <typename T>
inline void softmax(const size_t m, const size_t n, DevTensor<T>& X) {
    gpgpu::dnn::softmax(m, n, X.data(), 0);
}
} // namespace detail

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
softmax(const TensorT& X, TensorR&& Y, int axis = 1) {
    dlf::detail::norm_axis(X.rank(), axis);
    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    reorder(X, Y);
    detail::softmax(m, n, Y);
}

template <typename TensorT>
enable_if_tensor<TensorT>
softmax(const TensorT& X, int axis = 1) {
    tensor_type<TensorT> Y{};
    softmax(X, Y, axis);
    return Y;
}

template <typename TensorT>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    !is_tensor_view<TensorT>::value &&
    !std::is_lvalue_reference<TensorT>::value,
    tensor_type<TensorT>>
softmax(TensorT&& X, int axis = 1) {
    softmax(X, X, axis);
    return std::move(X);
}

namespace detail {
template <typename T>
void logsoftmax(const size_t m, const size_t n, Tensor<T>& X) {
    const size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto buffer = X.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](auto r) {
        for (int b = r.begin(); b < r.end(); ++b) {
            auto px = buffer + b*n;

            T amax = px[0];
            for (size_t i = 1; i < n; ++i) {
                amax = std::max(amax, px[i]);
            }

            T asum = 0;
            for (size_t i = 0; i < n; ++i) {
                asum += std::exp(px[i] - amax);
            }
            asum = std::log(asum);

            for (size_t i = 0; i < n; ++i) {
                px[i] = px[i] - amax - asum;
            }
        }
    });
}

template <typename T>
inline void logsoftmax(const size_t m, const size_t n, DevTensor<T>& X) {
    gpgpu::dnn::logsoftmax(m, n, X.data(), 0);
}
} // namespace detail

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
logsoftmax(const TensorT& X, TensorR&& Y, int axis = 1) {
    dlf::detail::norm_axis(X.rank(), axis);
    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    reorder(X, Y);
    detail::logsoftmax(m, n, Y);
}

template <typename TensorT>
enable_if_tensor<TensorT>
logsoftmax(const TensorT& X, int axis = 1) {
    tensor_type<TensorT> Y{};
    logsoftmax(X, Y, axis);
    return Y;
}

template <typename TensorT>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    !is_tensor_view<TensorT>::value &&
    !std::is_lvalue_reference<TensorT>::value,
    tensor_type<TensorT>>
logsoftmax(TensorT&& X, int axis = 1) {
    logsoftmax(X, X, axis);
    return std::move(X);
}

namespace detail {
template <typename T>
void hardmax(const size_t m, const size_t n, Tensor<T>& X) {
    const size_t grainsize = std::max(size_t(1), GRAINSIZE / n);
    auto buffer = X.data();

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](auto r) {
        for (int b = r.begin(); b < r.end(); ++b) {
            auto px = buffer + b*n;

            T amax = px[0];
            int imax = 0;
            for (size_t i = 0; i < n; ++i) {
                if (px[i] > amax) {
                    amax = px[i];
                    imax = i;
                }
                px[i] = 0;
            }
            px[imax] = 1;
        }
    });
}

template <typename T>
inline void hardmax(const size_t m, const size_t n, DevTensor<T>& X) {
    gpgpu::dnn::hardmax(m, n, X.data(), 0);
}
} // namespace detail

template <typename TensorT, typename TensorR>
std::enable_if_t<
    is_exactly_same_tensor<TensorT, TensorR>::value &&
    !std::is_const<std::remove_reference_t<TensorR>>::value>
hardmax(const TensorT& X, TensorR&& Y, int axis = 1) {
    dlf::detail::norm_axis(X.rank(), axis);
    auto m = X.shape().partial_size(0, axis);
    auto n = X.size() / m;
    reorder(X, Y);
    detail::hardmax(m, n, Y);
}

template <typename TensorT>
enable_if_tensor<TensorT>
hardmax(const TensorT& X, int axis = 1) {
    tensor_type<TensorT> Y{};
    hardmax(X, Y, axis);
    return Y;
}

template <typename TensorT>
std::enable_if_t<
    is_tensor<TensorT>::value &&
    !is_tensor_view<TensorT>::value &&
    !std::is_lvalue_reference<TensorT>::value,
    tensor_type<TensorT>>
hardmax(TensorT&& X, int axis = 1) {
    hardmax(X, X, axis);
    return std::move(X);
}

/**
 * Rearranges blocks of spatial data into depth. More specifically, this op
 * outputs a copy of the input tensor where values from the height and width
 * dimensions are moved to the depth dimension.
 *
 * @param X Input tensor of [N,C,H,W], where N is the batch axis, C is the
 *        channel or depth, H is the height and W is the width.
 * @param Y Output tensor of [N, C*blocksize*blocksize, H/blocksize,
 *        W/blocksize].
 * @param blocksize Blocks of [blocksize,blocksize] are moved.
 */
template <typename TensorT>
enable_if_tensor<TensorT, void>
space_to_depth(TensorT&& X, tensor_type<TensorT>& Y, int blocksize) {
    if (blocksize <= 0)
        throw shape_error("space_to_depth: blocksize has incorrect value");
    if (X.rank() != 4)
        throw shape_error("space_to_depth: input tensor must be 4-dimensional");

    int n = X.extent(0), c = X.extent(1), h = X.extent(2), w = X.extent(3);
    if (h % blocksize != 0 || w % blocksize != 0)
        throw shape_error("space_to_depth: blocksize has incorrect value");

    Y.resize(n, c*blocksize*blocksize, h/blocksize, w/blocksize);
    Y.reshape(n, blocksize, blocksize, c, h/blocksize, w/blocksize);

    auto x_view = reshape(std::forward<TensorT>(X),
        {n, c, h/blocksize, blocksize, w/blocksize, blocksize});
    reorder(x_view.transpose(0, 3, 5, 1, 2, 4), Y);
    Y.reshape(n, c*blocksize*blocksize, h/blocksize, w/blocksize);
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline space_to_depth(TensorT&& X, int blocksize) {
    tensor_type<TensorT> Y{};
    space_to_depth(std::forward<TensorT>(X), Y, blocksize);
    return Y;
}

/**
 * Rearranges (permutes) data from depth into blocks of spatial data. This is
 * reverse transformation of space_to_depth. More specifically, this op outputs
 * a copy of the input tensor where values from the depth dimension are moved
 * in spatial blocks to the height and width dimensions. By default, mode = DCR.
 * In the DCR mode, elements along the depth dimension from the input tensor are
 * rearranged in the following order: depth, column, and then row. The output Y
 * is computed from the input X as below:
 *
 *   n, c, h, w = X.shape
 *   tmp = reshape(X, (n, blocksize, blocksize, c/blocksize^2, h, w))
 *   tmp = transpose(tmp, (0, 3, 4, 1, 5, 2))
 *   Y = reshape(tmp, (b, c/blocksize^2, h*blocksize, w*blocksize))
 *
 * In the CRD mode, elements along the depth dimension from the input tensor are
 * rearranged in the following order: column, row, and the depth. The output Y
 * is computed from the input X as below:
 *
 *   n, c, h, w = X.shape
 *   tmp = reshape(X, (n, c/blocksize^2, blocksize, blocksize, h, w))
 *   tmp = transpose(tmp, (0, 1, 4, 2, 5, 3))
 *   Y = reshape(tmp, (n, c/blocksize^2, h*blocksize, w*blocksize))
 *
 * @param X Input tensor of [N,C,H,W], where N is the batch axis, C is the channel
 *        or depth, H is the height and W is the width.
 * @param Y Output tensor of [N, C/(blocksize*blocksize), H*blocksize,
 *        W*blocksize].
 * @param blocksize Blocks of [blocksize,blocksize] are moved.
 * @param mode DCR (default) for depth-column-row order re-arrangement. Use CRD
 *        for column-row-depth order.
 */
template <typename TensorT>
enable_if_tensor<TensorT, void>
depth_to_space(TensorT&& X, tensor_type<TensorT>& Y, int blocksize, std::string mode = "DCR") {
    if (blocksize <= 0)
        throw shape_error("depth_to_space: blocksize has incorrect value");
    if (X.rank() != 4)
        throw shape_error("depth_to_space: input tensor must be 4-dimensional");
    if (mode != "DCR" && mode != "CRD")
        throw shape_error("depth_to_space: mode has incorrect value");

    int n = X.extent(0), c = X.extent(1), h = X.extent(2), w = X.extent(3);
    if (c % (blocksize*blocksize) != 0)
        throw shape_error("depth_to_space: blocksize has incorrect value");

    Y.resize(n, c/(blocksize*blocksize), h*blocksize, w*blocksize);
    Y.reshape(n, c/(blocksize*blocksize), h, blocksize, w, blocksize);

    if (mode == "DCR") {
        auto x_view = reshape(std::forward<TensorT>(X),
            {n, blocksize, blocksize, c/(blocksize*blocksize), h, w});
        reorder(x_view.transpose(0, 3, 4, 1, 5, 2), Y);
        Y.reshape(n, c/(blocksize*blocksize), h*blocksize, w*blocksize);
    } else {
        auto x_view = reshape(std::forward<TensorT>(X),
            {n, c/(blocksize*blocksize), blocksize, blocksize, h, w});
        reorder(x_view.transpose(0, 1, 4, 2, 5, 3), Y);
        Y.reshape(n, c/(blocksize*blocksize), h*blocksize, w*blocksize);
    }
}

template <typename TensorT>
enable_if_tensor<TensorT>
inline depth_to_space(TensorT&& X, int blocksize, std::string mode = "DCR") {
    tensor_type<TensorT> Y{};
    depth_to_space(std::forward<TensorT>(X), Y, blocksize, mode);
    return Y;
}

namespace detail {
template <typename T>
struct BoundingBox {
    T start_y, start_x, end_y, end_x;

    static BoundingBox corner(T y1, T x1, T y2, T x2) {
        if (y1 > y2)
            std::swap(y1, y2);
        if (x1 > x2)
            std::swap(x1, x2);
        return BoundingBox{y1, x1, y2, x2};
    }

    static inline BoundingBox corner(const T* data) {
        return corner(data[0], data[1], data[2], data[3]);
    }

    static inline BoundingBox center(T x_center, T y_center, T width, T height) {
        return corner(y_center - height/2, x_center - width/2,
                      y_center + height/2, x_center + width/2);
    }

    static inline BoundingBox center(const T* data) {
        return center(data[0], data[1], data[2], data[3]);
    }

    T area() const noexcept {
        return (end_x - start_x + 1) * (end_y - start_y + 1);
    }

    T iou(const BoundingBox& other) const noexcept {
        auto x1 = std::max(start_x, other.start_x);
        auto x2 = std::min(end_x,   other.end_x);
        auto y1 = std::max(start_y, other.start_y);
        auto y2 = std::min(end_y,   other.end_y);

        auto overlap = std::max(T(0), x2 - x1 + 1) * std::max(T(0), y2 - y1 + 1);
        return overlap == 0 ? 0 : overlap / (area() + other.area() - overlap);
    }
};

template <typename T>
BoundingBox<T> get_box(const T* boxes, int32_t index, bool center_point_box) {
    return center_point_box ? BoundingBox<T>::center(boxes + index*4)
                            : BoundingBox<T>::corner(boxes + index*4);
}

template <typename T>
void nms(int32_t batch, int32_t klass, int32_t spatial_dim,
         const T* boxes, const T* scores, std::vector<int32_t>& indices,
         bool center_point_box, int32_t max_output_boxes,
         T iou_threshold, T score_threshold)
{
    // Sort by confidence score of bounding boxes
    std::vector<int32_t> order(spatial_dim);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [scores](auto x, auto y) {
        return scores[x] <= scores[y];
    });

    // Suppress bounding boxes with score threshold
    if (score_threshold > 0) {
        order.erase(std::remove_if(
            order.begin(), order.end(), [scores, score_threshold](auto i) {
                return scores[i] < score_threshold;
            }), order.end());
    }

    // Iterate bounding boxes
    int32_t num_boxes = 0;
    while (!order.empty() && num_boxes < max_output_boxes) {
        // The index of largest confidence score
        auto index = order.back(); order.pop_back();
        auto pivot = get_box(boxes, index, center_point_box);

        // Pick the bounding box with the largest confidence score
        indices.push_back(batch);
        indices.push_back(klass);
        indices.push_back(index);
        ++num_boxes;

        // Suppress bounding boxes with iou threshold
        order.erase(std::remove_if(
            order.begin(), order.end(), [=, &pivot](auto i) {
                return pivot.iou(get_box(boxes, i, center_point_box)) > iou_threshold;
            }), order.end());
    }
}
} // namespace detail

/**
 * Filter out boxes that have high intersection-over-union (IOU) overlap with
 * previously selected boxes. Bounding boxes with score less than score_threshold
 * are removed. Bounding box format is indicated by attribute center_point_box.
 * Note that this algorithm is agnostic to where the origin is in the coordinate
 * system and more generally is invariant to orthogonal transformations and
 * translations of the coordinate system; thus translating or reflections of the
 * coordinate system result in the same boxes being selected by algorithm. The
 * selected_indices output is a set of integers indexing into the input collection
 * of bounding boxes representing the selected boxes. The bounding box coordinates
 * corresponding to the selected indices can then be obtained using the gather or
 * gather_nd operation.
 *
 * @param boxes An input tensor with shape [num_batches, spatial_dimension, 4].
 *        The single box data format is indicated by center_point box.
 * @param scores An input tensor with shape [num_batches, num_classes, spatial_dimension].
 * @param selected_indices Selected indices from the boxes tensor.
 *        [num_selected_indices, 3], the select selected index format is
 *        [batch_index, class_index, box_index].
 * @param center_point_box Indicate the format of the box data. The default is false.
 *        False - the box data is supplied as [y1, x1, y2, x2] where (y1,x1) and
 *        (y2,x2) are the coordinates of any diagonal pair of box corners and the
 *        coordinates can be provided as normalized (i.e., lying in the interval
 *        [0, 1]) or absolute. Mostly used for TF models.
 *        True - the box data is supplied as [x_center, y_center, width, height].
 *        Mostly used for Pytorch models.
 * @param max_output_boxes_per_class Integer representing the maximum number of
 *        boxes to be selected per batch per class.
 * @param iou_threshold Float representing the threshold for deciding whether
 *        boxes overlap too much with respect to IOU.
 * @param score_threshold Float representing the threshold for deciding when to
 *        remove boxes based on score.
 */
template <typename T>
void nms(const Tensor<T>& boxes, const Tensor<T>& scores,
         Tensor<int32_t>& selected_indices,
         bool center_point_box = false,
         int32_t max_output_boxes_per_class = 0,
         T iou_threshold = 0, T score_threshold = 0)
{
    assert(boxes.rank() == 3);
    assert(boxes.extent(2) == 4);
    assert(scores.rank() == 3);
    assert(scores.extent(0) == boxes.extent(0));
    assert(scores.extent(2) == boxes.extent(1));

    auto num_batches = static_cast<int32_t>(boxes.extent(0));
    auto spatial_dim = static_cast<int32_t>(boxes.extent(1));
    auto num_classes = static_cast<int32_t>(scores.extent(1));
    std::vector<int32_t> indices;

    for (int32_t batch = 0; batch < num_batches; ++batch) {
        for (int32_t klass = 0; klass < num_classes; ++klass) {
            auto p_boxes  = boxes.data() + batch*spatial_dim*4;
            auto p_scores = scores.data() + (batch*num_classes + klass)*spatial_dim;
            detail::nms(batch, klass, spatial_dim,
                        p_boxes, p_scores, indices,
                        center_point_box, max_output_boxes_per_class,
                        iou_threshold, score_threshold);
        }
    }

    auto num_selected = indices.size() / 3;
    selected_indices.resize(num_selected, 3);
    std::copy(indices.begin(), indices.end(), selected_indices.begin());
}

template <typename T>
void nms(const DevTensor<T>& boxes, const DevTensor<T>& scores,
         DevTensor<int32_t>& selected_indices,
         bool center_point_box = false,
         int32_t max_output_boxes_per_class = 0,
         T iou_threshold = 0, T score_threshold = 0)
{
    // FIXME
    Tensor<int32_t> indices;
    nms(boxes.read(), scores.read(), indices,
        center_point_box, max_output_boxes_per_class,
        iou_threshold, score_threshold);

    selected_indices.resize(indices.shape());
    selected_indices.write(indices);
}

}} // namespace dlf::dnn
