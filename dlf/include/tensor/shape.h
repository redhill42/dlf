#ifndef _TENSOR_SHAPE_HPP
#define _TENSOR_SHAPE_HPP

#include <vector>
#include <cassert>

namespace dlf {

/**
 * The Shape defines the dimensions of a Tensor.
 */
class Shape
{
    std::vector<size_t> m_dims;

public:
    Shape() = default;
    explicit Shape(std::vector<size_t> init) : m_dims(std::move(init)) {}
    Shape(std::initializer_list<size_t> init) : m_dims(init) {}

    /**
     * Return number of dimensions in this shape.
     */
    size_t rank() const noexcept {
        return m_dims.size();
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t extent(size_t dim) const noexcept {
        return m_dims[dim];
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t operator[](size_t dim) const noexcept {
        return m_dims[dim];
    }

    /**
     * Return pair of extents if this shape represents a matrix.
     */
    std::pair<size_t,size_t> extent() const noexcept {
        assert(rank() == 2);
        return std::pair(extent(0), extent(1));
    }

    /**
     * Change dimensions of this shape. The new shape must compatible to
     * this shape.
     */
    bool reshape(Shape newshape);

    /**
     * Compare two shapes for equality.
     */
    bool operator==(const Shape& other) const {
        return m_dims == other.m_dims;
    }

    /**
     * Compare two shapes for non-equality.
     */
    bool operator!=(const Shape& other) const {
        return m_dims != other.m_dims;
    }

    /**
     * Shrink one level of dimensions.
     */
    Shape shrink() const {
        return Shape(std::vector<size_t>(std::next(m_dims.begin()), m_dims.end()));
    }

    /**
     * Returns the data size defined by this shape.
     */
    size_t size() const noexcept;

    /**
     * Return the data offset for the given index.
     */
    size_t offset(std::initializer_list<size_t> index) const noexcept;

    /**
     * Returns the next index within this shape.
     *
     * @return true if next index is available
     */
    bool next(std::vector<size_t>& index) const noexcept;
};

} //namespace dlf

#endif //_SHAPE_HPP
