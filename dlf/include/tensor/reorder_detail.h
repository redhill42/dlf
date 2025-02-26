#pragma once

namespace dlf { namespace detail {

//==-------------------------------------------------------------------------
// Tensor reorder operations
//==-------------------------------------------------------------------------

template <typename TensorX, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value && is_cpu_tensor<TensorY>::value, bool>
inline reorder_transpose(const TensorX& X, TensorY& Y) {
    if (X.rank() != 2)
        return false;
    if (X.stride(0) != 1 || static_cast<int>(X.stride(1)) < X.extent(0))
        return false;
    if (Y.stride(1) != 1 || static_cast<int>(Y.stride(0)) < Y.extent(1))
        return false;
    cblas::omatcopy(cblas::Transpose::Trans, X.extent(1), X.extent(0),
                    X.data() + X.shape().offset(), X.stride(1),
                    Y.data() + Y.shape().offset(), Y.stride(0));
    return true;
}

template <typename TensorX, typename TensorY>
std::enable_if_t<is_cpu_tensor<TensorX>::value && is_cpu_tensor<TensorY>::value>
reorder_impl(const TensorX& X, TensorY&& Y) {
    assert(X.shape() == Y.shape());

    if (X.shape().is_contiguous() && Y.shape().is_contiguous() &&
        X.data() == Y.data() && X.shape().offset() == Y.shape().offset())
        return;

    if (reorder_transpose(X, Y))
        return;

    map(xfn::transfer<>(), Y, X);
}

template <typename TensorX, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value && is_gpu_tensor<TensorY>::value, bool>
reorder_transpose(const TensorX& X, TensorY& Y) {
    if (X.rank() != 2)
        return false;
    if (X.stride(0) != 1 || static_cast<int>(X.stride(1)) < X.extent(0))
        return false;
    if (Y.stride(1) != 1 || static_cast<int>(Y.stride(0)) < Y.extent(1))
        return false;
    gpgpu::blas::omatcopy(gpgpu::blas::Layout::RowMajor,
                          gpgpu::blas::Transpose::Trans,
                          X.extent(1), X.extent(0),
                          xfn::one<tensor_value_type<TensorX>>(),
                          X.data(), X.shape().offset(), X.stride(1),
                          Y.data(), Y.shape().offset(), Y.stride(0));
    return true;
}

template <typename TensorX, typename TensorY>
std::enable_if_t<is_gpu_tensor<TensorX>::value && is_gpu_tensor<TensorY>::value>
reorder_impl(const TensorX& X, TensorY&& Y) {
    assert(X.shape() == Y.shape());

    if (X.shape().is_contiguous() && Y.shape().is_contiguous() &&
        X.data() == Y.data() && X.shape().offset() == Y.shape().offset())
        return;

    if (reorder_transpose(X, Y))
        return;

    if (X.original_shape().is_tail(X.shape()) && Y.shape().is_contiguous()) {
        gpgpu::dnn::copy(X.original_shape().size(), X.data(), X.shape().offset(),
                         Y.size(), Y.data(), Y.shape().offset());
    } else {
        gpgpu::dnn::copy(X.size(), X.shape().extents(),
                         X.data(), X.shape().offset(), X.shape().strides(),
                         Y.data(), Y.shape().offset(), Y.shape().strides());
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
void serial_reverse(T* first, T* last, T* result, const int stride) {
    if (stride == 1) {
        for (; first != last; ++result) {
            std::swap(*--last, *result);
        }
    } else {
        for (; first != last; result += stride) {
            last -= stride;
            std::swap(*last, *result);
        }
    }
}

template <typename T>
void parallel_reverse(const size_t n, T* first, T* last, const int stride) {
    auto m = n / 2;
    if (m < GRAINSIZE) {
        serial_reverse(first, first + m*stride, last - m*stride, stride);
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, m, GRAINSIZE), [=](const auto& r) {
            serial_reverse(first + r.begin() * stride, first + r.end() * stride,
                           last - r.end() * stride, stride);
        });
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
            parallel_reverse(n, px, px + n*stride, stride);
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
    const auto i_stride1    = indices.shape().partial_size(axis+1, indices.rank());
    const auto i_stride2    = i_stride1 * indices.extent(axis);
    const auto x_stride1    = X.shape().partial_size(axis+1, X.rank());
    const auto x_stride2    = x_stride1 * X.extent(axis);

    const auto x_data       = X.data();
    const auto x_shape      = X.shape();
    const auto x_contiguous = x_shape.is_contiguous();
    const auto x_offset     = x_shape.offset();
    const auto max_item     = X.extent(axis);

    map([=](auto& y, auto i, auto id) {
        auto tmp = normalize_index(i, static_cast<int>(max_item));
        auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        y = x_data[x_contiguous ? x_id + x_offset : x_shape.linear_offset(x_id)];
    }, Y, indices, map_id());
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
    const auto i_stride1    = indices.shape().partial_size(axis+1, indices.rank());
    const auto i_stride2    = i_stride1 * indices.extent(axis);
    const auto x_stride1    = X.shape().partial_size(axis+1, X.rank());
    const auto x_stride2    = x_stride1 * X.extent(axis);

          auto x_data       = X.data();
    const auto x_shape      = X.shape();
    const auto x_contiguous = x_shape.is_contiguous();
    const auto x_offset     = x_shape.offset();
    const auto max_item     = X.extent(axis);

    map([=](auto i, auto y, auto id) {
        auto tmp = normalize_index(i, static_cast<int>(max_item));
        auto x_id = (id % i_stride1) + (tmp * x_stride1) + (id / i_stride2 * x_stride2);
        x_data[x_contiguous ? x_id + x_offset : x_shape.linear_offset(x_id)] = y;
    }, indices, updates, map_id());
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
                dim = dims[j+1];
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
                dim = dims[j+1];
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
void serial_merge(size_t x_len, const T* x_data, const int x_inc,
                  size_t y_len, const T* y_data, const int y_inc,
                                      T* z_data, const int z_inc,
                  Compare comp)
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
        cxx::copy(x_len, x_data, x_inc, z_data, z_inc);
    if (y_len > 0)
        cxx::copy(y_len, y_data, y_inc, z_data, z_inc);
}

template <typename T, typename Compare>
void serial_merge(size_t x_len, T* x_data, const int x_inc,
                  size_t y_len, T* y_data, const int y_inc,
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
        cxx::move(x_len, x_data, x_inc, z_data, z_inc);
    if (y_len > 0)
        cxx::move(y_len, y_data, y_inc, z_data, z_inc);
}

template <typename T, typename Compare>
struct merge_task : tbb::task {
    using R = std::remove_const_t<T>;

    size_t x_len; T* x_data; const int x_inc;
    size_t y_len; T* y_data; const int y_inc;
                  R* z_data; const int z_inc;
    Compare comp;

    merge_task(size_t x_len, T* x_data, const int x_inc,
               size_t y_len, T* y_data, const int y_inc,
                             R* z_data, const int z_inc,
               Compare comp)
        : x_len(x_len), x_data(x_data), x_inc(x_inc),
          y_len(y_len), y_data(y_data), y_inc(y_inc),
                        z_data(z_data), z_inc(z_inc),
          comp(comp) {}

    task* execute() override;

    static void run(size_t x_len, T* x_data, const int x_inc,
                    size_t y_len, T* y_data, const int y_inc,
                                  R* z_data, const int z_inc,
                    Compare comp)
    {
        spawn_root_and_wait(*new(allocate_root()) merge_task(
            x_len, x_data, x_inc, y_len, y_data, y_inc, z_data, z_inc, comp));
    }
};

template <typename T, typename Compare>
tbb::task* merge_task<T, Compare>::execute() {
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_merge(x_len, x_data, x_inc, y_len, y_data, y_inc, z_data, z_inc, comp);
        return nullptr;
    } else {
        size_t xm, ym;
        if (x_len < y_len) {
            ym = y_len / 2;
            xm = cxx::upper_bound(x_len, x_data, x_inc, y_data[ym*y_inc], comp);
        } else {
            xm = x_len / 2;
            ym = cxx::lower_bound(y_len, y_data, y_inc, x_data[xm*x_inc], comp);
        }

        auto right = new(allocate_additional_child_of(*parent()))
            merge_task(x_len - xm, x_data + xm*x_inc, x_inc,
                       y_len - ym, y_data + ym*y_inc, y_inc,
                       z_data + (xm + ym)*z_inc, z_inc, comp);
        spawn(*right);
        recycle_as_continuation();
        x_len = xm;
        y_len = ym;
        return this;
    }
}

template <typename T, typename Compare>
void parallel_merge(size_t x_len, T* x_data, const int x_inc,
                    size_t y_len, T* y_data, const int y_inc,
                    std::remove_const_t<T>* z_data, const int z_inc,
                    Compare comp)
{
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_merge(x_len, x_data, x_inc, y_len, y_data, y_inc, z_data, z_inc, comp);
    } else {
        merge_task<T, Compare>::run(x_len, x_data, x_inc,
                                    y_len, y_data, y_inc,
                                    z_data, z_inc, comp);
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
           Compare comp)
{
    const cxx::string_view name = xfn::function_kernel_name(comp);
    const int dir = name != "less" && name != "less_equal";
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
        cxx::copy(n, x_data, x_inc, y_data, y_inc);
    if (y_inc == 1)
        std::sort(y_data, y_data + n, comp);
    else
        std::sort(strided_iterator<T>(y_data, y_inc, 0),
                  strided_iterator<T>(y_data, y_inc, n),
                  comp);
}

template <typename T, typename Compare>
struct sort_task : public tbb::task {
    int n;
    const T* x_data; int x_inc;
          T* y_data; int y_inc;
          T* t_data; int t_inc;
    Compare comp;

    sort_task(const int n,
              const T* x_data, const int x_inc,
                    T* y_data, const int y_inc,
                    T* t_data, const int t_inc,
              Compare comp)
        : n(n), x_data(x_data), x_inc(x_inc),
                y_data(y_data), y_inc(y_inc),
                t_data(t_data), t_inc(t_inc),
          comp(comp) {}

    task* execute() override;

    static void run(const int n,
                    const T* x_data, const int x_inc,
                          T* y_data, const int y_inc,
                    Compare comp)
    {
        std::vector<T> aux(n);
        spawn_root_and_wait(*new(allocate_root()) sort_task(
            n, x_data, x_inc, y_data, y_inc, aux.data(), 1, comp));
    }
};

template <typename T, typename Compare>
tbb::task* sort_task<T, Compare>::execute() {
    if (n <= SORT_CUT_OFF) {
        serial_sort(n, x_data, x_inc, y_data, y_inc, comp);
        return nullptr;
    } else {
        auto m = n / 2;
        auto c = new(allocate_continuation()) merge_task<T, Compare>(
            m, t_data, t_inc, n-m, t_data + m*t_inc, t_inc, y_data, y_inc, comp);
        c->set_ref_count(2);

        spawn(*new(c->allocate_child()) sort_task(
            n-m, x_data + m*x_inc, x_inc,
                 t_data + m*t_inc, t_inc,
                 y_data + m*y_inc, y_inc, comp));

        n = m;
        std::swap(y_data, t_data);
        std::swap(y_inc,  t_inc);
        recycle_as_child_of(*c);
        return this;
    }
}

template <typename T, typename Compare>
void parallel_sort(const int n,
                   const T* x_data, const int x_inc,
                         T* y_data, const int y_inc,
                   Compare comp)
{
    if (n < SORT_CUT_OFF) {
        serial_sort(n, x_data, x_inc, y_data, y_inc, comp);
    } else {
        sort_task<T, Compare>::run(n, x_data, x_inc, y_data, y_inc, comp);
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
            parallel_sort(n, px, x_inc, py, y_inc, comp);
        }
    });
}

template <typename T, typename Compare>
void sort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
          const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
          Compare comp)
{
    const cxx::string_view name = xfn::function_kernel_name(comp);
    const int dir = name != "less" && name != "less_equal";
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
             Compare comp)
{
    const cxx::string_view name = xfn::function_kernel_name(comp);
    const int dir = name != "less" && name != "less_equal";
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
void serial_argsort( /* insertion sort */
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
        auto loc = cxx::upper_bound(i, y_data, y_inc, *px, comp);
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
void serial_arg_merge(
    size_t x_len, K* x_key, const int x_key_inc,
                  V* x_val, const int x_val_inc,
    size_t y_len, K* y_key, const int y_key_inc,
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
        cxx::move(x_len, x_key, x_key_inc, z_key, z_key_inc);
        cxx::move(x_len, x_val, x_val_inc, z_val, z_val_inc);
    }
    if (y_len > 0) {
        cxx::move(y_len, y_key, y_key_inc, z_key, z_key_inc);
        cxx::move(y_len, y_val, y_val_inc, z_val, z_val_inc);
    }
}

template <typename K, typename V, typename Compare>
struct arg_merge_task : tbb::task {
    size_t x_len; K* x_key; const int x_key_inc;
                  V* x_val; const int x_val_inc;
    size_t y_len; K* y_key; const int y_key_inc;
                  V* y_val; const int y_val_inc;
                  K* z_key; const int z_key_inc;
                  V* z_val; const int z_val_inc;
    Compare comp;

    arg_merge_task(size_t x_len, K* x_key, const int x_key_inc,
                                 V* x_val, const int x_val_inc,
                   size_t y_len, K* y_key, const int y_key_inc,
                                 V* y_val, const int y_val_inc,
                                 K* z_key, const int z_key_inc,
                                 V* z_val, const int z_val_inc,
                   Compare comp)
        : x_len(x_len), x_key(x_key), x_key_inc(x_key_inc),
                        x_val(x_val), x_val_inc(x_val_inc),
          y_len(y_len), y_key(y_key), y_key_inc(y_key_inc),
                        y_val(y_val), y_val_inc(y_val_inc),
                        z_key(z_key), z_key_inc(z_key_inc),
                        z_val(z_val), z_val_inc(z_val_inc),
          comp(comp) {}

    task* execute() override;
};

template <typename K, typename V, typename Compare>
tbb::task* arg_merge_task<K, V, Compare>::execute() {
    if (x_len + y_len <= MERGE_CUT_OFF) {
        serial_arg_merge(x_len, x_key, x_key_inc, x_val, x_val_inc,
                         y_len, y_key, y_key_inc, y_val, y_val_inc,
                                z_key, z_key_inc, z_val, z_val_inc,
                         comp);
        return nullptr;
    } else {
        size_t xm, ym;
        if (x_len < y_len) {
            ym = y_len / 2;
            xm = cxx::upper_bound(x_len, x_key, x_key_inc, y_key[ym*y_key_inc], comp);
        } else {
            xm = x_len / 2;
            ym = cxx::upper_bound(y_len, y_key, y_key_inc, x_key[xm*x_key_inc], comp);
        }

        auto zm = xm + ym;
        auto right = new(allocate_additional_child_of(*parent()))
            arg_merge_task(x_len - xm, x_key + xm*x_key_inc, x_key_inc,
                                       x_val + xm*x_val_inc, x_val_inc,
                           y_len - ym, y_key + ym*y_key_inc, y_key_inc,
                                       y_val + ym*y_val_inc, y_val_inc,
                                       z_key + zm*z_key_inc, z_key_inc,
                                       z_val + zm*z_val_inc, z_val_inc,
                           comp);
        spawn(*right);
        recycle_as_continuation();
        x_len = xm;
        y_len = ym;
        return this;
    }
}

template <typename K, typename V, typename Generator, typename Compare>
struct argsort_task : public tbb::task {
    int n;
    const K* x_key; int x_key_inc;
          K* y_key; int y_key_inc;
          V* y_val; int y_val_inc;
          K* t_key; int t_key_inc;
          V* t_val; int t_val_inc;
    const int index; Generator gen; Compare comp;

    argsort_task(const int n,
                 const K* x_key, const int x_key_inc,
                       K* y_key, const int y_key_inc,
                       V* y_val, const int y_val_inc,
                       K* t_key, const int t_key_inc,
                       V* t_val, const int t_val_inc,
                 const int index, Generator gen, Compare comp)
        : n(n), x_key(x_key), x_key_inc(x_key_inc),
                y_key(y_key), y_key_inc(y_key_inc),
                y_val(y_val), y_val_inc(y_val_inc),
                t_key(t_key), t_key_inc(t_key_inc),
                t_val(t_val), t_val_inc(t_val_inc),
          index(index), gen(gen), comp(comp) {}

    task* execute() override;

    static void run(const int n,
                    const K* x_key, const int x_key_inc,
                          K* y_key, const int y_key_inc,
                          V* y_val, const int y_val_inc,
                    Generator gen, Compare comp)
    {
        std::vector<K> aux_key(n);
        std::vector<V> aux_val(n);
        spawn_root_and_wait(*new(allocate_root()) argsort_task(
            n, x_key, x_key_inc, y_key, y_key_inc, y_val, y_val_inc,
            aux_key.data(), 1, aux_val.data(), 1, 0, gen, comp));
    }
};

template <typename K, typename V, typename Generator, typename Compare>
tbb::task* argsort_task<K, V, Generator, Compare>::execute() {
    if (n <= ARGSORT_CUT_OFF) {
        serial_argsort(n, x_key, x_key_inc, y_key, y_key_inc, y_val, y_val_inc,
                       index, gen, comp);
        return nullptr;
    } else {
        auto m = n / 2;
        auto c = new(allocate_continuation()) arg_merge_task<K, V, Compare>(
            m,   t_key, t_key_inc, t_val, t_val_inc,
            n-m, t_key + m*t_key_inc, t_key_inc,
                 t_val + m*t_val_inc, t_val_inc,
            y_key, y_key_inc, y_val, y_val_inc, comp);
        c->set_ref_count(2);

        spawn(*new(c->allocate_child()) argsort_task(
            n-m, x_key + m*x_key_inc, x_key_inc,
                 t_key + m*t_key_inc, t_key_inc,
                 t_val + m*t_val_inc, t_val_inc,
                 y_key + m*y_key_inc, y_key_inc,
                 y_val + m*y_val_inc, y_val_inc,
                 index + m, gen, comp));

        n = m;
        std::swap(y_key, t_key);
        std::swap(y_key_inc, t_key_inc);
        std::swap(y_val, t_val);
        std::swap(y_val_inc, t_val_inc);
        recycle_as_child_of(*c);
        return this;
    }
}

template <typename T, typename I, typename Compare>
void parallel_argsort(const int n,
                      const T* x_data, const int x_inc,
                            T* y_data, const int y_inc,
                            I* i_data, const int i_inc,
                      Compare comp)
{
    using Generator = xfn::identity<I>;

    if (n <= ARGSORT_CUT_OFF) {
        serial_argsort(n, x_data, x_inc, y_data, y_inc, i_data, i_inc,
                       0, Generator(), comp);
    } else {
        argsort_task<T, I, Generator, Compare>::run(
            n, x_data, x_inc, y_data, y_inc, i_data, i_inc, Generator(), comp);
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
            parallel_argsort(n, px, x_inc, py, y_inc, pi, i_inc, comp);
        }
    });
}

template <typename T, typename I, typename Compare>
void argsort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
             const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
             const Shape& i_shape,       gpgpu::Buffer<I>& i_data,
             Compare comp)
{
    const cxx::string_view name = xfn::function_kernel_name(comp);
    const int dir = name != "less" && name != "less_equal";
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
