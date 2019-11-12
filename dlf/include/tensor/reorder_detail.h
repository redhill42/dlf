#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Utilities
//==-------------------------------------------------------------------------

template <typename T>
void copy(int n, const T* px, const int x_inc, T* py, const int y_inc) {
    if (x_inc == 1 && y_inc == 1) {
        std::copy(px, px + n, py);
    } else {
        for (; n > 0; --n, px += x_inc, py += y_inc)
            *py = *px;
    }
}

template <typename T>
void move(int n, T* px, const int x_inc, T* py, const int y_inc) {
    if (x_inc == 1 && y_inc == 1) {
        std::move(px, px + n, py);
    } else {
        for (; n > 0; --n, px += x_inc, py += y_inc)
            *py = std::move(*px);
    }
}

template <typename T, typename Compare>
int lower_bound(int n, const T* data, const int stride, const T& value, Compare comp) {
    int i = 0;
    while (n != 0) {
        int m = n / 2;
        if (comp(data[(m + i) * stride], value)) {
            i += m + 1;
            n -= m + 1;
        } else {
            n = m;
        }
    }
    return i;
}

template <typename T, typename Compare>
int upper_bound(int n, const T* data, const int stride, const T& value, Compare comp) {
    int i = 0;
    while (n != 0) {
        int m = n / 2;
        if (comp(value, data[(m + i) * stride]))
            n = m;
        else {
            i += m + 1;
            n -= m + 1;
        }
    }
    return i;
}

//==-------------------------------------------------------------------------
// Tensor reorder operations
//==-------------------------------------------------------------------------

#if HAS_MKL
template <typename Src, typename Dst>
std::enable_if_t<
    is_cpu_tensor<Src>::value && is_cpu_tensor<Dst>::value,
    bool>
inline reorder_transpose(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    if (src_shape.rank() != 2)
        return false;
    return cblas::omatcopy2(
        'R', 'T', src_shape.extent(1), src_shape.extent(0),
        tensor_value_type<Src>{1},
        src.data() + src_shape.offset(), src_shape.stride(1), src_shape.stride(0),
        dst.data() + dst_shape.offset(), dst_shape.stride(0), dst_shape.stride(1)) ;
}
#endif

template <typename Src, typename Dst>
std::enable_if_t<is_cpu_tensor<Src>::value && is_cpu_tensor<Dst>::value>
reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);
    using T = tensor_value_type<Src>;
    using U = tensor_value_type<Dst>;

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

#if HAS_MKL
    if (reorder_transpose(src, src_shape, dst, dst_shape))
        return;
#endif

    if (dst_shape.is_contiguous()) {
        if (src.original_shape().size() == 1) {
            std::fill(dst.data() + dst_shape.offset(),
                      dst.data() + dst_shape.offset() + dst_shape.size(),
                      *src.data());
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src.data() + src_shape.offset(),
                      src.data() + src_shape.offset() + src_shape.size(),
                      dst.data() + dst_shape.offset());
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src.data(), 0),
                  const_shaped_iterator<T>(src_shape, src.data(), src_shape.size()),
                  dst.data() + dst_shape.offset());
    } else {
        if (src.original_shape().size() == 1) {
            std::fill(shaped_iterator<U>(dst_shape, dst.data(), 0),
                      shaped_iterator<U>(dst_shape, dst.data(), dst_shape.size()),
                      *src.data());
            return;
        }

        if (src_shape.is_contiguous()) {
            par::copy(src.data() + src_shape.offset(),
                      src.data() + src_shape.offset() + src_shape.size(),
                      shaped_iterator<U>(dst_shape, dst.data(), 0));
            return;
        }

        par::copy(const_shaped_iterator<T>(src_shape, src.data(), 0),
                  const_shaped_iterator<T>(src_shape, src.data(), src_shape.size()),
                  shaped_iterator<U>(dst_shape, dst.data(), 0));
    }
}

template <typename Src, typename Dst>
std::enable_if_t<is_gpu_tensor<Src>::value && is_gpu_tensor<Dst>::value, bool>
reorder_transpose(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    if (src_shape.rank() != 2)
        return false;
    if (src_shape.stride(0) != 1 || static_cast<int>(src_shape.stride(1)) < src_shape.extent(0))
        return false;
    if (dst_shape.stride(1) != 1 || static_cast<int>(dst_shape.stride(0)) < dst_shape.extent(1))
        return false;
    gpgpu::blas::omatcopy(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::Trans,
                          src_shape.extent(1),
                          src_shape.extent(0),
                          tensor_value_type<Src>{1},
                          src.data(), src_shape.offset(), src_shape.stride(1),
                          dst.data(), dst_shape.offset(), dst_shape.stride(0));
    return true;
}

template <typename Src, typename Dst>
std::enable_if_t<is_gpu_tensor<Src>::value && is_gpu_tensor<Dst>::value>
reorder(const Src& src, const Shape& src_shape, Dst& dst, const Shape& dst_shape) {
    assert(src_shape == dst_shape);

    if (src_shape.is_contiguous() && dst_shape.is_contiguous() &&
        src.data() == dst.data() && src_shape.offset() == dst_shape.offset())
        return;

    if (reorder_transpose(src, src_shape, dst, dst_shape))
        return;

    if (src.original_shape().is_tail(src_shape) && dst_shape.is_contiguous()) {
        gpgpu::dnn::copy(src.original_shape().size(), src.data(), src_shape.offset(),
                         dst_shape.size(), dst.data(), dst_shape.offset());
    } else {
        gpgpu::dnn::copy(src_shape.size(), src_shape.extents(),
                         src.data(), src_shape.offset(), src_shape.strides(),
                         dst.data(), dst_shape.offset(), dst_shape.strides());
    }
}

template <typename TensorT>
enable_if_tensor<TensorT, tensor_view_type<TensorT>>
reshape(const TensorT& X, Shape&& new_shape) {
    if (X.shape().is_contiguous()) {
        return X.view(std::move(new_shape));
    } else {
        tensor_type<TensorT> Y{};
        reorder(X, Y);
        Y.reshape(std::move(new_shape));
        return Y.view();
    }
}

//==-------------------------------------------------------------------------
// Reverse
//==-------------------------------------------------------------------------

template <typename T>
void parallel_reverse(const size_t n, T* first, T* last, const int stride) {
    constexpr size_t REVERSE_CUT_OFF = 1000;
    if (n <= REVERSE_CUT_OFF) {
        last -= stride;
        for (size_t i = 0; i < n; ++i, first += stride, last -= stride)
            std::swap(*first, *last);
    } else {
        auto mid = n / 2;
        auto offset = mid * stride;
        tbb::parallel_invoke(
            [&]{ parallel_reverse(  mid, first,        last,        stride); },
            [&]{ parallel_reverse(n-mid, first+offset, last-offset, stride); });
    }
}

template <typename T>
void reverse(const Shape& x_shape, T* x_data) {
    auto n = x_shape.extent(-1);
    auto m = x_shape.size() / n;
    auto grainsize = std::max(size_t(1), GRAINSIZE/n);

    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        for (int batch = r.begin(); batch < r.end(); ++batch) {
            auto px = x_data + x_shape.linear_offset(batch * n);
            auto stride = static_cast<int>(x_shape.stride(-1));
            parallel_reverse(n/2, px, px + n*stride, stride);
        }
    });
}

template <typename T>
void reverse(const Shape& x_shape, gpgpu::Buffer<T>& x_data) {
    auto n = x_shape.extent(-1);
    auto m = x_shape.size() / n;
    gpgpu::dnn::reverse(m, n, x_shape.extents(), x_shape.strides(), x_data, x_shape.offset());
}

//==-------------------------------------------------------------------------
// Gather and scatter
//==-------------------------------------------------------------------------

inline int normalize_index(int index, const int max_item) {
    if (index < 0)
        index += max_item;
    return cxx::clamp(index, 0, max_item-1);
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
gather(const TensorX& X, TensorY& Y, const TensorI& indices,
       int m, int n, int chunk, int max_item)
{
    tbb::parallel_for(tbb::blocked_range2d<int>(0, m, 32, 0, n, 32), [&](auto r) {
        auto px = X.begin() + r.rows().begin() * chunk * max_item;
        for (int i = r.rows().begin(); i < r.rows().end(); ++i, px += chunk*max_item) {
            auto pi = indices.begin() + r.cols().begin();
            auto py = Y.begin() + (i*n + r.cols().begin()) * chunk;
            for (int j = r.cols().size(); j > 0; --j, ++pi, py += chunk) {
                auto id = normalize_index(*pi, max_item);
                std::copy(px + id*chunk, px + (id+1)*chunk, py);
            }
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
gather(const TensorX& X, TensorY& Y, const TensorI& indices,
       int m, int n, int chunk, int max_item)
{
    gpgpu::dnn::gather(
        m, n, chunk, max_item,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value, void>
gather_elements(const TensorX& X, TensorY& Y, const TensorI& indices, int axis) {
    tbb::parallel_for(tbb::blocked_range<int>(0, indices.size(), GRAINSIZE), [&](auto r) {
        const auto i_stride1 = indices.shape().partial_size(axis+1, indices.rank());
        const auto i_stride2 = i_stride1 * indices.extent(axis);
        const auto x_stride1 = X.shape().partial_size(axis+1, X.rank());
        const auto x_stride2 = x_stride1 * X.extent(axis);

        auto px = X.data();
        auto pi = indices.begin() + r.begin();
        auto py = Y.begin() + r.begin();

        const bool x_contiguous = X.shape().is_contiguous();
        const auto x_offset = X.shape().offset();
        const auto max_item = static_cast<int>(X.extent(axis));

        for (int id = r.begin(); id < r.end(); ++id, ++pi, ++py) {
            auto tmp = normalize_index(*pi, max_item);
            auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
            *py = px[x_contiguous ? x_id + x_offset : X.shape().linear_offset(x_id)];
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value, void>
gather_elements(const TensorX& X, TensorY& Y, const TensorI& indices, int axis) {
    gpgpu::dnn::gather_elements(
        Y.size(), axis,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
scatter_elements(TensorX& X, const TensorI& indices, const TensorY& updates, int axis) {
    tbb::parallel_for(tbb::blocked_range<int>(0, updates.size(), GRAINSIZE), [&](auto r) {
        const auto i_stride1 = indices.shape().partial_size(axis+1, indices.rank());
        const auto i_stride2 = i_stride1 * indices.extent(axis);
        const auto x_stride1 = X.shape().partial_size(axis+1, X.rank());
        const auto x_stride2 = x_stride1 * X.extent(axis);

        auto px = X.data();
        auto pi = indices.begin() + r.begin();
        auto pu = updates.begin() + r.begin();

        const bool x_contiguous = X.shape().is_contiguous();
        const auto x_offset = X.shape().offset();
        const auto max_item = static_cast<int>(X.extent(axis));

        for (int id = r.begin(); id < r.end(); ++id, ++pu, ++pi) {
            auto tmp = normalize_index(*pi, max_item);
            auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
            px[x_contiguous ? x_id + x_offset : X.shape().linear_offset(x_id)] = *pu;
        }
    });
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
scatter_elements(TensorX& X, const TensorI& indices, const TensorY& updates, int axis) {
    gpgpu::dnn::scatter_elements(
        indices.size(), axis,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        updates.shape().extents(), updates.shape().strides(),
        updates.data(), updates.shape().offset());
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
gather_nd(const TensorX& X, TensorY& Y, const TensorI& indices,
          const int n, const int k, const int chunk)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 64), [&](auto r) {
        auto px = X.begin();
        auto py = Y.begin() + r.begin()*chunk;
        auto pi = indices.begin() + r.begin()*k;
        auto dims = X.shape().extents();

        for (int i = r.begin(); i < r.end(); ++i) {
            // compute slice offset
            int offset = 0, dim = 1;
            for (int j = 0; j < k; ++j, ++pi) {
                offset = offset*dim + normalize_index(*pi, dims[j]);
                dim = dims[j];
            }
            offset *= chunk;

            // copy slice
            std::copy(px+offset, px+offset+chunk, py);
            py += chunk;
        }
    });
}

template <typename TensorX, typename TensorY, typename TensorI>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
gather_nd(const TensorX& X, TensorY& Y, const TensorI& indices,
          const int n, const int k, const int chunk)
{
    gpgpu::dnn::gather_nd(
        n, k, chunk,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        Y.shape().extents(), Y.shape().strides(),
        Y.data(), Y.shape().offset());
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value>
scatter_nd(TensorX& X, const TensorI& indices, const TensorY& updates,
           const int n, const int k, const int chunk)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1), [&](auto r) {
        auto px = X.begin();
        auto py = updates.begin() + r.begin()*chunk;
        auto pi = indices.begin() + r.begin()*k;
        auto dims = X.shape().extents();

        for (int i = r.begin(); i < r.end(); ++i) {
            // compute slice offset
            int offset = 0, dim = 1;
            for (int j = 0; j < k; ++j, ++pi) {
                offset = offset*dim + detail::normalize_index(*pi, dims[j]);
                dim = dims[j];
            }
            offset *= chunk;

            // copy slice
            std::copy(py, py+chunk, px+offset);
            py += chunk;
        }
    });
}

template <typename TensorX, typename TensorI, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value>
scatter_nd(TensorX& X, const TensorI& indices, const TensorY& updates,
           const int n, const int k, const int chunk)
{
    gpgpu::dnn::scatter_nd(
        n, k, chunk,
        X.shape().extents(), X.shape().strides(),
        X.data(), X.shape().offset(),
        indices.shape().extents(), indices.shape().strides(),
        indices.data(), indices.shape().offset(),
        updates.shape().extents(), updates.shape().strides(),
        updates.data(), updates.shape().offset());
}

//==-------------------------------------------------------------------------
// Merge
//==-------------------------------------------------------------------------

constexpr int MERGE_CUT_OFF = 1000;

template <typename T, typename Compare>
void serial_merge(int x_len, const T* x_data, const int x_inc,
                  int y_len, const T* y_data, const int y_inc,
                  T* z_data, const int z_inc, Compare comp)
{
    while (x_len > 0 && y_len > 0) {
        if (comp(*y_data, *x_data)) {
            *z_data = *y_data;
            y_data += y_inc;
            --y_len;
        } else {
            *z_data = *x_data;
            x_data += x_inc;
            --x_len;
        }
        z_data += z_inc;
    }
    if (x_len > 0)
        detail::copy(x_len, x_data, x_inc, z_data, z_inc);
    if (y_len > 0)
        detail::copy(y_len, y_data, y_inc, z_data, z_inc);
}

template <typename T, typename Compare>
void parallel_merge(int x_len, const T* x_data, const int x_inc,
                    int y_len, const T* y_data, const int y_inc,
                    T* z_data, const int z_inc, Compare comp)
{
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_merge(x_len, x_data, x_inc, y_len, y_data, y_inc, z_data, z_inc, comp);
    } else {
        int xm, ym;
        if (x_len < y_len) {
            ym = y_len / 2;
            xm = detail::upper_bound(x_len, x_data, x_inc, y_data[ym * y_inc], comp);
        } else {
            xm = x_len / 2;
            ym = detail::lower_bound(y_len, y_data, y_inc, x_data[xm * x_inc], comp);
        }

        int zm = xm + ym;
        tbb::parallel_invoke(
            [=] {
                parallel_merge(xm, x_data, x_inc, ym, y_data, y_inc, z_data, z_inc, comp);
            },
            [=] {
                parallel_merge(x_len - xm, x_data + xm*x_inc, x_inc,
                               y_len - ym, y_data + ym*y_inc, y_inc,
                               z_data + zm*z_inc, z_inc, comp);
            });
    }
}

template <typename T, typename Compare>
void merge(const Shape& x_shape, const T* x_data,
           const Shape& y_shape, const T* y_data,
           const Shape& z_shape,       T* z_data,
           Compare comp)
{
    assert(x_shape.extent(-1) + y_shape.extent(-1) == z_shape.extent(-1));
    const auto n = z_shape.extent(-1);
    const auto batch_size = z_shape.size() / n;
    const auto grainsize = std::max(size_t(1), GRAINSIZE / n);

    tbb::parallel_for(tbb::blocked_range<size_t>(0, batch_size, grainsize), [&](auto r) {
        auto x_len = x_shape.extent(-1);
        auto y_len = y_shape.extent(-1);
        auto x_inc = x_shape.stride(-1);
        auto y_inc = y_shape.stride(-1);
        auto z_inc = z_shape.stride(-1);

        for (auto batch = r.begin(); batch < r.end(); ++batch) {
            auto px = x_data + x_shape.linear_offset(batch * x_len);
            auto py = y_data + y_shape.linear_offset(batch * y_len);
            auto pz = z_data + z_shape.linear_offset(batch * (x_len + y_len));
            parallel_merge(x_len, px, x_inc, y_len, py, y_inc, pz, z_inc, comp);
        }
    });
}

template <typename T, typename Compare>
void merge(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
           const Shape& y_shape, const gpgpu::Buffer<T>& y_data,
           const Shape& z_shape, gpgpu::Buffer<T>& z_data,
           Compare)
{
    const std::string comp = Compare::name;
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::merge(dir,
        x_shape.extents(), x_shape.strides(), x_data, x_shape.offset(),
        y_shape.extents(), y_shape.strides(), y_data, y_shape.offset(),
        z_shape.extents(), z_shape.strides(), z_data, z_shape.offset());
}

//==-------------------------------------------------------------------------
// Sort
//==-------------------------------------------------------------------------

constexpr int SORT_CUT_OFF = 500;

template <typename T, typename Compare>
void serial_sort(const int n,
                 const T* x_data, const int x_inc,
                       T* y_data, const int y_inc,
                 Compare comp)
{
    if (x_data != y_data)
        detail::copy(n, x_data, x_inc, y_data, y_inc);
    if (y_inc == 1)
        std::sort(y_data, y_data + n, comp);
    else
        std::sort(strided_iterator<T>(y_data, y_inc, 0),
                  strided_iterator<T>(y_data, y_inc, n),
                  comp);
}

template <typename T, typename Compare>
void serial_merge(int x_len, T* x_data, const int x_inc,
                  int y_len, T* y_data, const int y_inc,
                             T* z_data, const int z_inc,
                  Compare comp)
{
    while (x_len > 0 && y_len > 0) {
        if (comp(*y_data, *x_data)) {
            *z_data = std::move(*y_data);
            y_data += y_inc;
            --y_len;
        } else {
            *z_data = std::move(*x_data);
            x_data += x_inc;
            --x_len;
        }
        z_data += z_inc;
    }
    if (x_len > 0)
        detail::move(x_len, x_data, x_inc, z_data, z_inc);
    if (y_len > 0)
        detail::move(y_len, y_data, y_inc, z_data, z_inc);
}

template <typename T, typename Compare>
void parallel_merge(int x_len, T* x_data, const int x_inc,
                    int y_len, T* y_data, const int y_inc,
                               T* z_data, const int z_inc,
                    Compare comp)
{
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_merge(x_len, x_data, x_inc, y_len, y_data, y_inc, z_data, z_inc, comp);
    } else {
        int xm, ym;
        if (x_len < y_len) {
            ym = y_len / 2;
            xm = upper_bound(x_len, x_data, x_inc, y_data[ym*y_inc], comp);
        } else {
            xm = x_len / 2;
            ym = lower_bound(y_len, y_data, y_inc, x_data[xm*x_inc], comp);
        }

        int zm = xm + ym;
        tbb::parallel_invoke(
            [=]{ parallel_merge(xm, x_data, x_inc, ym, y_data, y_inc, z_data, z_inc, comp); },
            [=]{ parallel_merge(x_len - xm, x_data + xm*x_inc, x_inc,
                                y_len - ym, y_data + ym*y_inc, y_inc,
                                            z_data + zm*z_inc, z_inc,
                                comp); });
    }
}

template <typename T, typename Compare>
void parallel_merge_sort_aux(const int n,
                             const T* x_data, const int x_inc,
                                   T* y_data, const int y_inc,
                                   T* t_data, const int t_inc,
                             Compare comp)
{
    if (n <= SORT_CUT_OFF) {
        serial_sort(n, x_data, x_inc, y_data, y_inc, comp);
    } else {
        auto mid = n / 2;
        tbb::parallel_invoke(
            [=] { parallel_merge_sort_aux(mid, x_data, x_inc,
                                          t_data, t_inc,
                                          y_data, y_inc, comp);
            },
            [=] { parallel_merge_sort_aux(n - mid,
                                          x_data + mid*x_inc, x_inc,
                                          t_data + mid*t_inc, t_inc,
                                          y_data + mid*y_inc, y_inc, comp);
            });
        parallel_merge(mid, t_data, t_inc,
                       n-mid, t_data + mid*t_inc, t_inc,
                       y_data, y_inc, comp);
    }
}

template <typename T, typename Compare>
void parallel_merge_sort(const int n,
                         const T* x_data, const int x_inc,
                               T* y_data, const int y_inc,
                         Compare comp)
{
    if (n < SORT_CUT_OFF) {
        serial_sort(n, x_data, x_inc, y_data, y_inc, comp);
    } else {
        std::vector<T> aux(n);
        parallel_merge_sort_aux(n, x_data, x_inc, y_data, y_inc, aux.data(), 1, comp);
    }
}

template <typename T, typename Compare>
void sort(const Shape& x_shape, const T* x_data,
          const Shape& y_shape,       T* y_data,
          Compare comp)
{
    const auto n = x_shape.extent(-1);
    const auto m = x_shape.size() / n;

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        const auto x_inc = static_cast<int>(x_shape.stride(-1));
        const auto y_inc = static_cast<int>(y_shape.stride(-1));
        for (int batch = r.begin(); batch < r.end(); ++batch) {
            auto px = x_data + x_shape.linear_offset(batch*n);
            auto py = y_data + y_shape.linear_offset(batch*n);
            parallel_merge_sort(n, px, x_inc, py, y_inc, comp);
        }
    });
}

template <typename T, typename Compare>
void sort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
          const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
          Compare)
{
    std::string_view comp = Compare::name;
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::sort(dir, x_shape.extents(),
                     x_data, x_shape.offset(), x_shape.strides(),
                     y_data, y_shape.offset(), y_shape.strides());
}

//==-------------------------------------------------------------------------
// Argsort
//==-------------------------------------------------------------------------

template <typename T, typename I, typename Compare>
void argsort(const Shape& x_shape, const T* x_data,
             const Shape& i_shape,       I* i_data,
             Compare comp)
{
    const auto n = x_shape.extent(-1);
    const auto m = x_shape.size() / n;

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        const auto x_inc = static_cast<int>(x_shape.stride(-1));
        const auto i_inc = static_cast<int>(i_shape.stride(-1));

        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = x_data + x_shape.linear_offset(i*n);
            auto pi = i_data + i_shape.linear_offset(i*n);

            if (i_inc == 1) {
                std::iota(pi, pi+n, 0);
                tbb::parallel_sort(pi, pi+n, [=](const auto a, const auto b) {
                    return comp(px[a*x_inc], px[b*x_inc]);
                });
            } else {
                auto begin = strided_iterator<I>(pi, i_inc, 0);
                auto end   = strided_iterator<I>(pi, i_inc, n);
                std::iota(begin, end, 0);
                tbb::parallel_sort(begin, end, [=](const auto a, const auto b) {
                    return comp(px[a*x_inc], px[b*x_inc]);
                });
            }
        }
    });
}

template <typename T, typename I, typename Compare>
void argsort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
             const Shape& i_shape,       gpgpu::Buffer<I>& i_data,
             Compare)
{
    const std::string_view comp = Compare::name;
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::argsort(dir, x_shape.extents(),
                        x_data, x_shape.offset(), x_shape.strides(),
                        i_data, i_shape.offset(), i_shape.strides());
}

/*-------------------------------------------------------------------------*/

constexpr int ARGSORT_CUT_OFF = 100;

template <typename K, typename V>
void insert(int loc, int n, K key, V val, K* k_data, int k_inc, V* v_data, int v_inc) {
    if (k_inc == 1) {
        std::move_backward(k_data + loc, k_data + n, k_data + n + 1);
        k_data[loc] = std::move(key);
    } else {
        for (int i = n; i > loc; --i)
            k_data[i * k_inc] = std::move(k_data[(i - 1) * k_inc]);
        k_data[loc * k_inc] = std::move(key);
    }

    if (v_inc == 1) {
        std::move_backward(v_data + loc, v_data + n, v_data + n + 1);
        v_data[loc] = std::move(val);
    } else {
        for (int i = n; i > loc; --i)
            v_data[i * v_inc] = std::move(v_data[(i - 1) * v_inc]);
        v_data[loc * v_inc] = std::move(val);
    }
}

template <typename K, typename V, typename Generator, typename Compare>
void serial_sort( /* insertion sort */
    const int n,
    const K* x_data, const int x_inc,
          K* y_data, const int y_inc,
          V* v_data, const int v_inc,
    const int index, Generator gen, Compare comp)
{
    auto px = x_data;
    auto py = y_data;
    auto pv = v_data;

    for (int i = 0; i < n; ++i, px += x_inc, py += y_inc, pv += v_inc) {
        auto loc = upper_bound(i, y_data, y_inc, *px, comp);
        auto val = gen(index + i);

        if (x_data == y_data) {
            if (loc != i)
                insert(loc, i, std::move(*py), std::move(val), y_data, y_inc, v_data, v_inc);
            else
                *pv = std::move(val);
        } else {
            if (loc != i)
                insert(loc, i, *px, std::move(val), y_data, y_inc, v_data, v_inc);
            else {
                *py = *px;
                *pv = std::move(val);
            }
        }
    }
}

template <typename K, typename V, typename Compare>
void serial_merge(int x_len, K* x_key, const int x_key_inc,
                             V* x_val, const int x_val_inc,
                  int y_len, K* y_key, const int y_key_inc,
                             V* y_val, const int y_val_inc,
                             K* z_key, const int z_key_inc,
                             V* z_val, const int z_val_inc,
                  Compare comp)
{
    while (x_len > 0 && y_len > 0) {
        if (comp(*y_key, *x_key)) {
            *z_key = std::move(*y_key);
            *z_val = std::move(*y_val);
            y_key += y_key_inc;
            y_val += y_val_inc;
            --y_len;
        } else {
            *z_key = std::move(*x_key);
            *z_val = std::move(*x_val);
            x_key += x_key_inc;
            x_val += x_val_inc;
            --x_len;
        }
        z_key += z_key_inc;
        z_val += z_val_inc;
    }
    if (x_len > 0) {
        detail::move(x_len, x_key, x_key_inc, z_key, z_key_inc);
        detail::move(x_len, x_val, x_val_inc, z_val, z_val_inc);
    }
    if (y_len > 0) {
        detail::move(y_len, y_key, y_key_inc, z_key, z_key_inc);
        detail::move(y_len, y_val, y_val_inc, z_val, z_val_inc);
    }
}

template <typename K, typename V, typename Compare>
void parallel_merge(int x_len, K* x_key, const int x_key_inc,
                               V* x_val, const int x_val_inc,
                    int y_len, K* y_key, const int y_key_inc,
                               V* y_val, const int y_val_inc,
                               K* z_key, const int z_key_inc,
                               V* z_val, const int z_val_inc,
                    Compare comp)
{
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_merge(x_len, x_key, x_key_inc, x_val, x_val_inc,
                     y_len, y_key, y_key_inc, y_val, y_val_inc,
                            z_key, z_key_inc, z_val, z_val_inc,
                     comp);
    } else {
        int xm, ym;
        if (x_len < y_len) {
            ym = y_len / 2;
            xm = upper_bound(x_len, x_key, x_key_inc, y_key[ym*y_key_inc], comp);
        } else {
            xm = x_len / 2;
            ym = lower_bound(y_len, y_key, y_key_inc, x_key[xm*x_key_inc], comp);
        }

        int zm = xm + ym;
        tbb::parallel_invoke(
            [=] {
                parallel_merge(xm, x_key, x_key_inc, x_val, x_val_inc,
                               ym, y_key, y_key_inc, y_val, y_val_inc,
                                   z_key, z_key_inc, z_val, z_val_inc,
                               comp);
            },
            [=] {
                parallel_merge(x_len - xm, x_key + xm*x_key_inc, x_key_inc,
                                           x_val + xm*x_val_inc, x_val_inc,
                               y_len - ym, y_key + ym*y_key_inc, y_key_inc,
                                           y_val + ym*y_val_inc, y_val_inc,
                                           z_key + zm*z_key_inc, z_key_inc,
                                           z_val + zm*z_val_inc, z_val_inc,
                               comp);
            });
    }
}

template <typename K, typename V, typename Generator, typename Compare>
void parallel_merge_sort_aux(const int n,
                             const K* x_key, const int x_key_inc,
                                   K* y_key, const int y_key_inc,
                                   V* y_val, const int y_val_inc,
                                   K* t_key, const int t_key_inc,
                                   V* t_val, const int t_val_inc,
                             const int index, Generator gen, Compare comp)
{
    if (n <= ARGSORT_CUT_OFF) {
        serial_sort(n, x_key, x_key_inc, y_key, y_key_inc, y_val, y_val_inc, index, gen, comp);
    } else {
        auto mid = n / 2;

        tbb::parallel_invoke(
            [=] {
                parallel_merge_sort_aux(
                    mid,
                    x_key, x_key_inc,
                    t_key, t_key_inc,
                    t_val, t_val_inc,
                    y_key, y_key_inc,
                    y_val, y_val_inc,
                    index, gen, comp);
            },
            [=] {
                parallel_merge_sort_aux(
                    n-mid,
                    x_key + mid*x_key_inc, x_key_inc,
                    t_key + mid*t_key_inc, t_key_inc,
                    t_val + mid*t_val_inc, t_val_inc,
                    y_key + mid*y_key_inc, y_key_inc,
                    y_val + mid*y_val_inc, y_val_inc,
                    index + mid, gen, comp);
            });

        parallel_merge(  mid, t_key,                 t_key_inc,
                              t_val,                 t_val_inc,
                       n-mid, t_key + mid*t_key_inc, t_key_inc,
                              t_val + mid*t_val_inc, t_val_inc,
                       y_key, y_key_inc, y_val, y_val_inc, comp);
    }
}

template <typename T, typename I, typename Compare>
void parallel_merge_argsort(const int n,
                            const T* x_data, const int x_inc,
                                  T* y_data, const int y_inc,
                                  I* i_data, const int i_inc,
                            Compare comp)
{
    if (n <= ARGSORT_CUT_OFF) {
        serial_sort(n, x_data, x_inc, y_data, y_inc, i_data, i_inc,
                    0, xfn::identity<I>(), comp);
    } else {
        std::vector<T> aux_data(n);
        std::vector<I> aux_idx(n);
        parallel_merge_sort_aux(
            n, x_data, x_inc, y_data, y_inc, i_data, i_inc,
            aux_data.data(), 1, aux_idx.data(), 1,
            0, xfn::identity<I>(), comp);
    }
}

template <typename T, typename I, typename Compare>
void argsort(const Shape& x_shape, const T* x_data,
             const Shape& y_shape,       T* y_data,
             const Shape& i_shape,       I* i_data,
             Compare comp)
{
    const auto n = x_shape.extent(-1);
    const auto m = x_shape.size() / n;

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        const auto x_inc = static_cast<int>(x_shape.stride(-1));
        const auto y_inc = static_cast<int>(y_shape.stride(-1));
        const auto i_inc = static_cast<int>(i_shape.stride(-1));

        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = x_data + x_shape.linear_offset(i*n);
            auto py = y_data + y_shape.linear_offset(i*n);
            auto pi = i_data + i_shape.linear_offset(i*n);
            parallel_merge_argsort(n, px, x_inc, py, y_inc, pi, i_inc, comp);
        }
    });
}

template <typename T, typename I, typename Compare>
void argsort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
             const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
             const Shape& i_shape,       gpgpu::Buffer<I>& i_data,
             Compare)
{
    const std::string_view comp = Compare::name;
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::argsort(dir, x_shape.extents(),
                        x_data, x_shape.offset(), x_shape.strides(),
                        y_data, y_shape.offset(), y_shape.strides(),
                        i_data, i_shape.offset(), i_shape.strides());
}

//==-------------------------------------------------------------------------
// Top-K
//==-------------------------------------------------------------------------

template <typename K, typename V, typename Compare>
void percolate_down(std::pair<K,V>* heap, size_t k, size_t i, Compare comp) {
    const auto left  = i * 2;
    const auto right = i * 2 + 1;

    auto j = i;
    if (left <= k && comp(heap[j].first, heap[left].first))
        j = left;
    if (right <= k && comp(heap[j].first, heap[right].first))
        j = right;
    if (j != i) {
        std::swap(heap[i], heap[j]);
        percolate_down(heap, k, j, comp);
    }
}

template <typename K, typename V, typename Compare>
std::vector<std::pair<K,V>>
heap_top_k(const size_t n, const size_t k,
           const K* data, const size_t stride,
           V index, Compare comp)
{
    std::vector<std::pair<K,V>> heap;

    // build heap from first k elements
    heap.reserve(k);
    for (size_t i = 0; i < k; ++i, data += stride, ++index)
        heap.emplace_back(*data, index);
    for (size_t i = k/2; i > 0; --i)
        percolate_down(heap.data()-1, k, i, comp);

    // replace minimum element with greater value
    for (size_t i = k; i < n; ++i, data += stride, ++index) {
        if (comp(*data, heap.front().first)) {
            heap.front().first  = *data;
            heap.front().second = index;
            percolate_down(heap.data()-1, k, 1, comp);
        }
    }

    return heap;
}

template <typename T, typename I, typename Compare>
void top_k(const Shape& x_shape, const T* x_data,
           const Shape& y_shape,       T* y_data,
           const Shape& i_shape,       I* i_data,
           bool sorted, Compare comp)
{
    const auto n = x_shape.extent(-1);
    const auto k = y_shape.extent(-1);
    const auto m = x_shape.size() / n;
    assert(k <= n);

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        for (int batch = r.begin(); batch < r.end(); ++batch) {
            auto px = x_data + x_shape.linear_offset(batch*n);
            auto py = y_data + y_shape.linear_offset(batch*k);
            auto pi = i_data + i_shape.linear_offset(batch*k);

            const auto x_inc = static_cast<int>(x_shape.stride(-1));
            const auto y_inc = static_cast<int>(y_shape.stride(-1));
            const auto i_inc = static_cast<int>(i_shape.stride(-1));

            auto heap = heap_top_k(n, k, px, x_inc, 0, comp);
            if (sorted) {
                std::sort(heap.begin(), heap.end(), [comp](const auto& x, const auto& y) {
                    return comp(x.first, y.first);
                });
            }

            auto it = heap.begin();
            for (int i = 0; i < k; ++i, ++it, py += y_inc, pi += i_inc) {
                *py = it->first;
                *pi = it->second;
            }
        }
    });
}

template <typename T, typename I>
void top_k(const Shape& x_shape, const T* x_data,
           const Shape& y_shape,       T* y_data,
           const Shape& i_shape,       I* i_data,
           bool largest, bool sorted)
{
    if (largest)
        top_k(x_shape, x_data, y_shape, y_data, i_shape, i_data, sorted, std::greater<T>());
    else
        top_k(x_shape, x_data, y_shape, y_data, i_shape, i_data, sorted, std::less<T>());
}

template <typename T, typename I>
void top_k(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
           const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
           const Shape& i_shape,       gpgpu::Buffer<I>& i_data,
           bool largest, bool /*sorted*/)
{
    gpgpu::dnn::top_k(
        y_shape.extent(-1), largest, x_shape.extents(), y_shape.extents(),
        x_data, x_shape.offset(), x_shape.strides(),
        y_data, y_shape.offset(), y_shape.strides(),
        i_data, i_shape.offset(), i_shape.strides());
}

}} // namespace dlf::detail
