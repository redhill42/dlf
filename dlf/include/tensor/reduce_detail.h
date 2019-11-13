#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Tensor reduction implementation
//==-------------------------------------------------------------------------

template <typename R, typename IteratorX, typename IteratorY,
          typename Map, typename Reduce, typename Final>
void reduce(const int m, const int n,
            IteratorX x_begin, IteratorY y_begin,
            const R& identity, Map f, Reduce g, Final h)
{
    if (m*n < GRAINSIZE) {
        auto px = x_begin;
        auto py = y_begin;
        for (int i = 0; i < m; ++i, ++py) {
            auto acc = identity;
            for (int j = 0; j < n; ++j, ++px)
                acc = g(acc, f(*px));
            *py = h(acc, n);
        }
    } else {
        auto grainsize = std::max(1, GRAINSIZE/n);
        tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [=](auto r) {
            auto py = y_begin + r.begin();
            for (int i = r.begin(); i < r.end(); ++i, ++py) {
                *py = h(tbb::parallel_reduce(
                    tbb::blocked_range<int>(0, n, GRAINSIZE),
                    identity,
                    [=](auto c, auto acc) {
                        auto px = x_begin + (i*n + c.begin());
                        for (int j = c.size(); j > 0; --j, ++px)
                            acc = g(acc, f(*px));
                        return acc;
                    },
                    g), n);
            }
        });
    }
}

template <typename R, typename IteratorX, typename Map, typename Reduce, typename Final>
inline void reduce(const int m, const int n,
                   IteratorX x_begin, const Shape& y_shape, R* y_data,
                   const R& identity, Map f, Reduce g, Final h)
{
    if (y_shape.is_contiguous()) {
        reduce(m, n, x_begin, y_data + y_shape.offset(), identity, f, g, h);
    } else {
        auto y_begin = shaped_iterator<R>(y_shape, y_data, 0);
        reduce(m, n, x_begin, y_begin, identity, f, g, h);
    }
}

template <typename T, typename R, typename Map, typename Reduce, typename Final>
inline void reduce(const int m, const int n,
                   const Shape& x_shape, const T* x_data,
                   const Shape& y_shape, R* y_data,
                   const char*, const R& identity,
                   Map f, Reduce g, Final h)
{
    if (x_shape.is_contiguous()) {
        reduce(m, n, x_data + x_shape.offset(), y_shape, y_data, identity, f, g, h);
    } else {
        auto x_begin = const_shaped_iterator<T>(x_shape, x_data, 0);
        reduce(m, n, x_begin, y_shape, y_data, identity, f, g, h);
    }
}

template <typename T, typename Map, typename Reduce, typename Final>
inline void reduce(const int m, const int n,
                   const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
                   const Shape& y_shape, gpgpu::Buffer<T>& y_data,
                   const char* name, const T&, Map, Reduce, Final)
{
    gpgpu::dnn::reduce(name, m, n,
                       x_shape.extents(), x_shape.strides(),
                       x_data, x_shape.offset(),
                       y_shape.extents(), y_shape.strides(),
                       y_data, y_shape.offset());
}

template <typename TensorT, typename TensorR>
Shape prepare_reduce(const TensorT& X, TensorR& Y, std::vector<int>&& axes, bool keepdims, int* m, int* n) {
    auto rank = X.rank();
    detail::norm_axes(rank, axes, true);

    std::vector<size_t> output_dims;
    std::vector<size_t> transpose_perm;
    *m = *n = 1;

    for (int i = 0; i < rank; i++) {
        // axes empty means reduce all dim
        if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
            output_dims.push_back(X.extent(i));
            transpose_perm.push_back(i);
        } else if (keepdims) {
            output_dims.push_back(1);
        }
    }
    for (int i = 0; i < rank; i++) {
        if (axes.empty() || std::find(axes.begin(), axes.end(), i) != axes.end()) {
            transpose_perm.push_back(i);
            *n *= X.extent(i);
        } else {
            *m *= X.extent(i);
        }
    }

    Y.resize(Shape(output_dims));
    return X.shape().transpose(transpose_perm);
}

template <typename TensorT, typename TensorR, typename Map, typename Reduce, typename Final>
void reduce(const TensorT& X, TensorR& Y, std::vector<int>&& axes, bool keepdims,
            const char* name, const tensor_value_type<TensorR>& identity,
            Map f, Reduce g, Final h)
{
    int m, n;
    auto x_shape = prepare_reduce(X, Y, std::move(axes), keepdims, &m, &n);
    reduce(m, n, x_shape, X.data(), Y.shape(), Y.data(), name, identity, f, g, h);
}

//==-------------------------------------------------------------------------
// Arg reduction operations
//==-------------------------------------------------------------------------

template <typename T, typename Compare>
void arg_reduce(const Shape& x_shape, const T* x_data,
                const Shape& y_shape, int* y_data,
                const char*, Compare compare)
{
    auto k = x_shape.extent(-1);
    auto n = x_shape.size() / k;
    auto strideK = x_shape.stride(-1);

    tbb::parallel_for(tbb::blocked_range<int>(0, n, std::max(size_t(1), GRAINSIZE/k)), [=](auto r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = x_data + x_shape.linear_offset(i*k);
            auto py = y_data + y_shape.linear_offset(i);
            T acc = *px;
            int index = 0;
            for (int ik = 1; ik < k; ++ik) {
                px += strideK;
                if (compare(*px, acc)) {
                    acc = *px;
                    index = ik;
                }
            }
            *py = index;
        }
    });
}

template <typename T, typename Compare>
void arg_reduce(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
                const Shape& y_shape, gpgpu::Buffer<int>& y_data,
                const char* name, Compare)
{
    auto k = x_shape.extent(-1);
    auto n = x_shape.size() / k;
    gpgpu::dnn::arg_reduce(
        name, n, k,
        x_shape.extents(), x_shape.strides(),
        x_data, x_shape.offset(),
        y_shape.extents(), y_shape.strides(),
        y_data, y_shape.offset());
}

template <typename TensorT, typename TensorR, typename Compare>
void arg_reduce(const TensorT& X, TensorR& Y, int axis, bool keepdims,
                const char* name, Compare compare)
{
    norm_axis(X.rank(), axis);

    auto y_dims = X.shape().extents();
    if (keepdims) {
        y_dims[axis] = 1;
    } else {
        y_dims.erase(y_dims.begin() + axis);
    }

    auto x_view = moveaxis(X, axis, -1);
    Y.resize(Shape{y_dims});
    arg_reduce(x_view.shape(), x_view.data(), Y.shape(), Y.data(), name, compare);
}

//==-------------------------------------------------------------------------
// Scan
//==-------------------------------------------------------------------------

template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T& id, Op op,
          const Shape& x_shape, const T* x_data,
          const Shape& y_shape, T* y_data)
{
    const auto grainsize = std::max(1, GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](const auto& r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            tbb::parallel_scan(tbb::blocked_range<int>(0, n, GRAINSIZE),
                id,
                [&](const auto& c, auto acc, const bool is_final_scan) {
                    auto px = x_data + x_shape.linear_offset(i*n + c.begin());
                    auto py = y_data + y_shape.linear_offset(i*n + c.begin());
                    auto ix = x_shape.stride(-1);
                    auto iy = y_shape.stride(-1);

                    auto j = static_cast<int>(c.size());
                    if (is_final_scan && exclusive) {
                        for (; j > 0; --j, px += ix, py += iy) {
                            *py = acc;
                            acc = op(acc, *px);
                        }
                    } else if (is_final_scan && !exclusive) {
                        for (; j > 0; --j, px += ix, py += iy) {
                            acc = op(acc, *px);
                            *py = acc;
                        }
                    } else {
                        for (; j > 0; --j, px += ix) {
                            acc = op(acc, *px);
                        }
                    }
                    return acc;
                },
                op);
        }
    });
}

template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T&, Op,
          const Shape& x_shape, const gpgpu::Buffer<T>& x_buffer,
          const Shape& y_shape, gpgpu::Buffer<T>& y_buffer)
{
    gpgpu::dnn::scan(Op::cumulative, m, n, exclusive, x_shape.extents(),
                     x_buffer, x_shape.offset(), x_shape.strides(),
                     y_buffer, y_shape.offset(), y_shape.strides());
}

}} // namespace dlf::detail
