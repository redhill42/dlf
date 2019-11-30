#pragma once

namespace detail {

//=--------------------------------------------------------------------------
// Normalization
//=--------------------------------------------------------------------------

template <typename T>
void batch_norm(const Tensor<T>& X, Tensor<T>& Y,
                const Tensor<T>& scale, const Tensor<T>& bias,
                const Tensor<T>& mean, const Tensor<T>& var,
                const T epsilon = T(1e-5))
{
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
    gpgpu::dnn::batch_norm(X.shape().extents(), X.data(), Y.data(),
                           scale.data(), bias.data(), mean.data(),
                           var.data(), epsilon);
}

template <typename T>
void lrn(const Tensor<T>& X, Tensor<T>& Y, const int n,
         const T alpha = 0.00001, const T beta = 0.75, const T bias = 1.0)
{
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
    gpgpu::dnn::lrn(X.shape().extents(), X.data(), Y.data(), nsize, alpha, beta, bias);
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

            dlf::detail::gemm(
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

//=--------------------------------------------------------------------------
// Pooling
//=--------------------------------------------------------------------------

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

//=--------------------------------------------------------------------------
// Normalize
//=--------------------------------------------------------------------------

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

//=--------------------------------------------------------------------------
// Non Maximum Suppression
//=--------------------------------------------------------------------------

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
