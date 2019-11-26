#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Tensor reduction implementation
//==-------------------------------------------------------------------------

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

template <typename R, typename IteratorX, typename IteratorY,
          typename Map, typename Reduce, typename Final>
void do_reduce(const int m, const int n,
               IteratorX x_begin, IteratorY y_begin,
               const R& identity, Map f, Reduce g, Final h)
{
    if (m*n < GRAINSIZE) {
        for (int i = 0; i < m; ++i, ++y_begin) {
            auto x_end = x_begin + n;
            *y_begin = h(cxx::transform_reduce(x_begin, x_end, identity, g, f), n);
            x_begin = x_end;
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
                        return cxx::transform_reduce(px, px + c.size(), acc, g, f);
                    },
                    g), n);
            }
        });
    }
}

template <typename TensorT, typename TensorR, typename Map, typename Reduce, typename Final>
void reduce(const TensorT& X, TensorR& Y, std::vector<int>&& axes, bool keepdims,
            const tensor_value_type<TensorR>& identity, Map f, Reduce g, Final h)
{
    int m, n;
    auto x_shape = prepare_reduce(X, Y, std::move(axes), keepdims, &m, &n);
    unravel([=](auto x_begin, auto y_begin) {
        do_reduce(m, n, x_begin, y_begin, identity, f, g, h);
    }, X.view(x_shape), Y);
}

template <typename Reducer, typename TensorT, typename TensorR>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
inline reduce(const TensorT& X, TensorR& Y, std::vector<int>&& axes, bool keepdims) {
    reduce(X, Y, std::move(axes), keepdims,
           Reducer::identity(),
           typename Reducer::Map(),
           typename Reducer::Reduce(),
           typename Reducer::Final());
}

template <typename Reducer, typename TensorT, typename TensorR>
std::enable_if_t<is_gpu_tensor<TensorT>::value>
inline reduce(const TensorT& X, TensorR& Y, std::vector<int>&& axes, bool keepdims) {
    int m, n;
    auto x_shape = prepare_reduce(X, Y, std::move(axes), keepdims, &m, &n);
    gpgpu::dnn::reduce(xfn::reduction_kernel_name(Reducer()), m, n,
                       x_shape.extents(), x_shape.strides(),
                       X.data(), x_shape.offset(),
                       Y.shape().extents(), Y.shape().strides(),
                       Y.data(), Y.shape().offset());
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
T serial_scan(int n, int i, bool is_exclusive, bool is_final_scan, T acc, Op op,
              const Shape& x_shape, const T* x_data,
              const Shape& y_shape,       T* y_data)
{
    auto px = x_data + x_shape.linear_offset(i);
    auto py = y_data + y_shape.linear_offset(i);
    auto ix = x_shape.stride(-1);
    auto iy = y_shape.stride(-1);

    if (is_final_scan) {
        if (is_exclusive) {
            for (; n > 0; --n, px += ix, py += iy) {
                *py = acc;
                acc = op(std::move(acc), *px);
            }
        } else {
            for (; n > 0; --n, px += ix, py += iy) {
                acc = op(std::move(acc), *px);
                *py = acc;
            }
        }
    } else {
        for (; n > 0; --n, px += ix) {
            acc = op(std::move(acc), *px);
        }
    }
    return acc;
}

template <typename T, typename Op>
void parallel_scan(int n, int i, int j, bool exclusive, const T& id, Op op,
                   const Shape& x_shape, const T* x_data,
                   const Shape& y_shape,       T* y_data)
{
    if (n < GRAINSIZE) {
        for (; i < j; ++i) {
            serial_scan(n, i*n, exclusive, true, id, op,
                        x_shape, x_data, y_shape, y_data);
        }
    } else {
        for (; i < j; ++i) {
            tbb::parallel_scan(tbb::blocked_range<int>(0, n, GRAINSIZE),
                id,
                [=, &x_shape, &y_shape](const auto& r, auto acc, const bool is_final_scan) {
                    return serial_scan(
                        r.size(), i*n + r.begin(), exclusive, is_final_scan,
                        std::move(acc), op, x_shape, x_data, y_shape, y_data);
                },
                op);
        }
    }
}

template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T& id, Op op,
          const Shape& x_shape, const T* x_data,
          const Shape& y_shape,       T* y_data)
{
    if (m*n < GRAINSIZE) {
        for (int i = 0; i < m; ++i) {
            serial_scan(n, i*n, exclusive, true, id, op,
                        x_shape, x_data, y_shape, y_data);
        }
    } else {
        const auto grainsize = std::max(1, GRAINSIZE/n);
        tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](const auto& r) {
            parallel_scan(n, r.begin(), r.end(), exclusive, id, op,
                          x_shape, x_data, y_shape, y_data);
        });
    }
}

template <typename T, typename Op>
void scan(int m, int n, bool exclusive, const T&, Op op,
          const Shape& x_shape, const gpgpu::Buffer<T>& x_buffer,
          const Shape& y_shape, gpgpu::Buffer<T>& y_buffer)
{
    gpgpu::dnn::scan(xfn::scan_kernel_name(op),
                     m, n, exclusive, x_shape.extents(),
                     x_buffer, x_shape.offset(), x_shape.strides(),
                     y_buffer, y_shape.offset(), y_shape.strides());
}

}} // namespace dlf::detail
