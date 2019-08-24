#pragma once

namespace dlf {

template <typename T>
class strided_iterator {
    T* m_data;
    size_t m_stride;

public:
    using value_type = T;
    using difference_type = ptrdiff_t;
    using reference = value_type&;
    using pointer = value_type*;
    using iterator_category = std::random_access_iterator_tag;

    strided_iterator(T* data, size_t stride, difference_type start)
        : m_data(data + start*stride), m_stride(stride) {}

    strided_iterator& operator++() noexcept { m_data += m_stride; return *this; }
    strided_iterator& operator--() noexcept { m_data -= m_stride; return *this; }
    strided_iterator  operator++(int) noexcept { auto t = *this; ++(*this); return t; }
    strided_iterator  operator--(int) noexcept { auto t = *this; --(*this); return t; }
    strided_iterator& operator+=(difference_type n) noexcept { m_data += n*m_stride; return *this; }
    strided_iterator& operator-=(difference_type n) noexcept { m_data -= n*m_stride; return *this; }

    strided_iterator operator+(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, n); }
    strided_iterator operator-(difference_type n) const noexcept
        { return strided_iterator(m_data, m_stride, -n); }
    difference_type operator-(const strided_iterator& rhs) const noexcept
        { return (m_data - rhs.m_data) / m_stride; }

    reference operator*() const noexcept { return *m_data; }
    pointer operator->() const noexcept { return m_data; }
    reference operator[](int n) const noexcept { return m_data[n*m_stride]; }

    bool operator==(const strided_iterator& rhs) const noexcept
        { return m_data == rhs.m_data; }
    bool operator!=(const strided_iterator& rhs) const noexcept
        { return m_data != rhs.m_data; }
    bool operator<(const strided_iterator& rhs) const noexcept
        { return m_data < rhs.m_data; }
    bool operator<=(const strided_iterator& rhs) const noexcept
        { return m_data <= rhs.m_data; }
    bool operator>(const strided_iterator& rhs) const noexcept
        { return m_data > rhs.m_data; }
    bool operator>=(const strided_iterator& rhs) const noexcept
        { return m_data >= rhs.m_data; }
};

template <typename TensorT, typename Compare>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
sort(TensorT& X, int axis, Compare comp) {
    detail::norm_axis(X.rank(), axis);

    auto m = X.shape().partial_size(0, axis);
    auto k = X.extent(axis);
    auto n = X.shape().partial_size(axis+1, X.rank());
    auto strideK = X.stride(axis);

    if (strideK == 1) {
        tbb::parallel_for(tbb::blocked_range<int>(0, m*n, std::max(size_t(1), GRAINSIZE/k)), [&](auto r) {
            for (int id = r.begin(); id < r.end(); ++id) {
                auto px = X.data() + X.shape().linear_offset((id/n)*k*n + (id%n));
                std::sort(px, px+k, comp);
            }
        });
    } else {
        using T = tensor_value_type<TensorT>;
        tbb::parallel_for(tbb::blocked_range<int>(0, m*n, std::max(size_t(1), GRAINSIZE/k)), [&](auto r) {
            for (int id = r.begin(); id < r.end(); ++id) {
                auto px = X.data() + X.shape().linear_offset((id/n)*k*n + (id%n));
                std::sort(strided_iterator<T>(px, strideK, 0),
                          strided_iterator<T>(px, strideK, k),
                          comp);
            }
        });
    }
}

template <typename TensorT>
std::enable_if_t<is_cpu_tensor<TensorT>::value>
inline sort(TensorT& X, int axis = -1) {
    sort(X, axis, std::less<tensor_value_type<TensorT>>());
}

} // namespace dlf
