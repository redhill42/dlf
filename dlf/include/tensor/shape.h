#pragma once

#include <vector>
#include <cassert>
#include <cstdlib>

namespace dlf {

/**
 * The Shape defines the dimensions of a Tensor.
 */
class Shape
{
    struct dim_t {
        size_t extent;
        size_t stride;

        bool operator==(const dim_t& rhs) const noexcept {
            return extent == rhs.extent && stride == rhs.stride;
        }
    };

    std::vector<dim_t> m_dims;
    size_t m_size = 0;

    void init(const std::vector<size_t>& extents) noexcept;

public:
    Shape() = default;
    explicit Shape(const std::vector<size_t>& dims) { init(dims); };
    Shape(std::initializer_list<size_t> dims) { init(dims); }
    Shape(const Shape&) = default;
    Shape& operator=(const Shape&) = default;
    Shape(Shape&& rhs);
    Shape& operator=(Shape&& rhs);

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
        return m_dims[dim].extent;
    }

    /**
     * Returns the stride in a given dimension.
     */
    size_t stride(size_t dim) const noexcept {
        return m_dims[dim].stride;
    }

    /**
     * Returns the total number of elements defined by this shape.
     */
    size_t size() const noexcept {
        return m_size;
    }

    /**
     * Change dimensions of this shape. The new shape must compatible to
     * this shape.
     */
    bool reshape(std::vector<size_t> newshape) noexcept;

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
};

/**
 * Base class for shaped objects.
 */
class Shaped {
private:
    Shape m_shape;

public:
    Shaped() = default;
    explicit Shaped(const Shape& shape) : m_shape(shape) {}
    explicit Shaped(Shape&& shape) : m_shape(std::move(shape)) {}

    /**
     * Returns the shape of this shaped object.
     */
    const Shape& shape() const noexcept {
        return m_shape;
    }

    /**
     * Reshape the tensor without changing shaped data. The new shape
     * should be compatible with the original shape. At most one
     * dimension of the new shape can be -1. In this case, the
     * actual dimension value is inferred from the size of the tensor
     * and the remaining dimensions.
     *
     * @param newshape specifies the new shape.
     * @return true if shape changed, false if new shape is not
     * compatible with current shape.
     */
    bool reshape(const std::vector<size_t> newshape) noexcept {
        return m_shape.reshape(newshape);
    }

    /**
     * Return number of dimensions in this shape.
     */
    size_t rank() const noexcept {
        return m_shape.rank();
    }

    /**
     * Returns number of elements in a given dimension.
     */
    size_t extent(size_t dim) const noexcept {
        return m_shape.extent(dim);
    }

    /**
     * Returns the stride in a given dimension.
     */
    size_t stride(size_t dim) const noexcept {
        return m_shape.stride(dim);
    }

    /**
     * Returns the total size of this tensor.
     */
    size_t size() const noexcept {
        return m_shape.size();
    }

    /**
     * Return true if this is an empty tensor.
     */
    bool empty() const noexcept {
        return size() == 0;
    }

    /**
     * Returns true if this shaped object represent a 1-dimensional vector.
     */
    bool is_vector() const noexcept {
        return rank() == 1;
    }

    /**
     * Returns true if this shaped object represents a 2-dimensional matrix.
     */
    bool is_matrix() const noexcept {
        return rank() == 2;
    }

    /**
     * Returns true if this shaped object is a square matrix.
     */
    bool is_square() const noexcept {
        return is_matrix() && extent(0) == extent(1);
    }
};

} //namespace dlf
