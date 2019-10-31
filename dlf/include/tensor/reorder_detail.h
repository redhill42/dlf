#pragma once

namespace dlf { namespace detail {

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
// Merge and sort
//==-------------------------------------------------------------------------

// Reference:
// https://library.technion.ac.il/projects/ele/2011/Merge_path.pdf

template <typename T, typename Compare>
void select(const T* X, int Lx, int dx, const T* Y, int Ly, int dy,
            int k, int& ix, int& iy, Compare comp)
{
    int ix_start = std::min(k, Lx);
    int ix_end   = std::max(0, k - Ly);
    int iy_start = std::max(0, k - Lx);
    int start = 0, end = ix_start - ix_end;

    // Binary search diagonal intersection
    do {
        int m = (start + end) / 2;
        ix = ix_start - m;
        iy = iy_start + m;
        if (iy != Ly && (ix == Lx || comp(Y[iy * dy], X[ix * dx]))) {
            start = m + 1;
        } else {
            end = m - 1;
        }
    } while (start <= end);

    // Adjust boundary
    if (ix > 0 && iy < Ly && (ix == Lx || comp(Y[iy * dy], X[ix * dx])))
        ix--, iy++;
    if (iy > 0 && ix < Lx && comp(X[ix * dx], Y[(iy - 1) * dy]))
        ix++, iy--;
}

template <typename T, typename Compare>
void merge(const Shape& x_shape, const T* x_data,
           const Shape& y_shape, const T* y_data,
           const Shape& z_shape, T* z_data,
           Compare comp)
{
    const auto Lx = static_cast<int>(x_shape.extent(-1));
    const auto Ly = static_cast<int>(y_shape.extent(-1));
    const auto Lz = static_cast<int>(z_shape.extent(-1));
    const auto dx = static_cast<int>(x_shape.stride(-1));
    const auto dy = static_cast<int>(y_shape.stride(-1));
    const auto dz = static_cast<int>(z_shape.stride(-1));
    const auto batch_count = z_shape.size() / Lz;

    const auto grainsize = std::max(1, GRAINSIZE / Lz);
    tbb::parallel_for(tbb::blocked_range2d<int>(0, batch_count, grainsize, 0, Lz, GRAINSIZE), [&](auto r) {
        for (int batch = r.rows().begin(); batch < r.rows().end(); ++batch) {
            auto k = r.cols().begin();

            auto px = x_data + x_shape.linear_offset(batch * Lx);
            auto py = y_data + y_shape.linear_offset(batch * Ly);
            auto pz = z_data + z_shape.linear_offset(batch * Lz + k);
            auto px_end = px + Lx * dx;
            auto py_end = py + Ly * dy;

            int ix, iy;
            select(px, Lx, dx, py, Ly, dy, k, ix, iy, comp);
            assert(ix >= 0 && ix <= Lx);
            assert(iy >= 0 && iy <= Ly);
            assert(!(ix == Lx && iy == Ly));

            px += ix * dx, py += iy * dy;
            for (int cnt = r.cols().size(); cnt > 0; --cnt, pz += dz) {
                if (py == py_end) {
                    if (dx == 1 && dz == 1) {
                        std::copy(px, px_end, pz);
                    } else {
                        for (; cnt > 0; --cnt, px += dx, pz += dz) {
                            *pz = *px;
                        }
                    }
                    break;
                } else if (px == px_end) {
                    if (dy == 1 && dz == 1) {
                        std::copy(py, py_end, pz);
                    } else {
                        for (; cnt > 0; --cnt, py += dy, pz += dz) {
                            *pz = *py;
                        }
                    }
                    break;
                } else {
                    if (comp(*py, *px)) {
                        *pz = *py;
                        py += dy;
                    } else {
                        *pz = *px;
                        px += dx;
                    }
                }
            }
        }
    });
}

template <typename T, typename Compare>
void merge(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
           const Shape& y_shape, const gpgpu::Buffer<T>& y_data,
           const Shape& z_shape, gpgpu::Buffer<T>& z_data,
           Compare /*FIXME*/)
{
    gpgpu::dnn::merge(x_shape.extents(), x_shape.strides(),
                      x_data, x_shape.offset(),
                      y_shape.extents(), y_shape.strides(),
                      y_data, y_shape.offset(),
                      z_shape.extents(), z_shape.strides(),
                      z_data, z_shape.offset());
}

template <typename T, typename Compare>
void sort(const Shape& shape, T* data, Compare comp) {
    const auto n = shape.extent(-1);
    const auto m = shape.size() / n;
    const auto inc = static_cast<int>(shape.stride(-1));

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = data + shape.linear_offset(i*n);
            if (inc == 1)
                tbb::parallel_sort(px, px+n, comp);
            else
                tbb::parallel_sort(strided_iterator<T>(px, inc, 0),
                                   strided_iterator<T>(px, inc, n),
                                   comp);
        }
    });
}

template <typename T>
void sort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
          const Shape& y_shape,       gpgpu::Buffer<T>& y_data,
          const std::string& comp)
{
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::sort(dir, x_shape.extents(),
                     x_data, x_shape.offset(), x_shape.strides(),
                     y_data, y_shape.offset(), y_shape.strides());
}

template <typename T, typename R, typename Compare>
void argsort(const Shape& x_shape, const T* x_data,
             const Shape& y_shape,       R* y_data,
             Compare comp)
{
    const auto n = x_shape.extent(-1);
    const auto m = x_shape.size() / n;
    const auto x_inc = static_cast<int>(x_shape.stride(-1));
    const auto y_inc = static_cast<int>(y_shape.stride(-1));

    const auto grainsize = std::max(size_t(1), GRAINSIZE/n);
    tbb::parallel_for(tbb::blocked_range<int>(0, m, grainsize), [&](auto r) {
        for (int i = r.begin(); i < r.end(); ++i) {
            auto px = x_data + x_shape.linear_offset(i*n);
            auto py = y_data + y_shape.linear_offset(i*n);
            if (y_inc == 1) {
                std::iota(py, py+n, 0);
                tbb::parallel_sort(py, py+n, [=](const auto a, const auto b) {
                    return comp(px[a * x_inc], px[b * x_inc]);
                });
            } else {
                std::iota(strided_iterator<R>(py, y_inc, 0),
                          strided_iterator<R>(py, y_inc, n),
                          0);
                tbb::parallel_sort(strided_iterator<R>(py, y_inc, 0),
                                   strided_iterator<R>(py, y_inc, n),
                                   [=](const auto a, const auto b) {
                    return comp(px[a * x_inc], px[b * x_inc]);
                });
            }
        }
    });
}

template <typename T, typename R, typename Compare>
void argsort(const Shape& x_shape, const gpgpu::Buffer<T>& x_data,
             const Shape& y_shape,       gpgpu::Buffer<R>& y_data,
             Compare)
{
    const std::string comp = Compare::name;
    const int dir = comp != "less" && comp != "less_equal";
    gpgpu::dnn::argsort(dir, x_shape.extents(),
                        x_data, x_shape.offset(), x_shape.strides(),
                        y_data, y_shape.offset(), y_shape.strides());
}

}} // namespace dlf::detail
