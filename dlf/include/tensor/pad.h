#pragma once

namespace dlf {

enum class PadMode {
    Constant, Edge, Reflect, Symmetric, Wrap, Maximum, Minimum, Mean
};

namespace detail {
template <typename TensorT>
inline void prepend_constant(TensorT& X, const tensor_value_type<TensorT>& val, int pad_amt, int axis) {
    if (pad_amt > 0) {
        X.slice({0}, {pad_amt}, {axis}, {1}).fill(val);
    }
}

template <typename TensorT>
inline void append_constant(TensorT& X, const tensor_value_type<TensorT>& val, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        X.slice({-pad_amt}, {dim}, {axis}, {1}).fill(val);
    }
}

template <typename TensorT>
void prepend_edge(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({pad_amt}, {pad_amt+1}, {axis}, {1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice.broadcast_to(y_edge_slice.shape()), y_edge_slice);
    }
}

template <typename TensorT>
void append_edge(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-1}, {-pad_amt}, {axis}, {1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice.broadcast_to(y_edge_slice.shape()), y_edge_slice);
    }
}

template <typename TensorT>
void prepend_reflect(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({2*pad_amt}, {pad_amt}, {axis}, {-1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void append_reflect(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-2}, {-2*pad_amt-2}, {axis}, {-1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void prepend_symmetric(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto x_edge_slice = X.slice({2*pad_amt-1}, {pad_amt-1}, {axis}, {-1});
        auto y_edge_slice = X.slice({0}, {pad_amt}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void append_symmetric(TensorT& X, int pad_amt, int axis) {
    if (pad_amt > 0) {
        auto dim = static_cast<int>(X.extent(axis));
        auto x_edge_slice = X.slice({-pad_amt-1}, {-2*pad_amt-1}, {axis}, {-1});
        auto y_edge_slice = X.slice({-pad_amt}, {dim}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename TensorT>
void pad_wrap(TensorT& X, int pad_before, int pad_after, int axis) {
    auto dim = static_cast<int>(X.extent(axis));

    if (pad_before < 0 || pad_after < 0) // FIXME
        throw std::logic_error("pad: negative wrap padding is not supported");

    if (pad_before > 0) {
        auto x_edge_slice = X.slice({dim-(pad_before+pad_after)}, {dim-pad_after}, {axis}, {1});
        auto y_edge_slice = X.slice({0}, {pad_before}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
    if (pad_after > 0) {
        auto x_edge_slice = X.slice({pad_before}, {pad_before+pad_after}, {axis}, {1});
        auto y_edge_slice = X.slice({-pad_after}, {dim}, {axis}, {1});
        reorder(x_edge_slice, y_edge_slice);
    }
}

template <typename Reducer, typename TensorT>
void pad_reduce(TensorT& X, int pad_before, int pad_after, int axis) {
    if (pad_before <= 0 && pad_after <= 0)
        return;

    if (pad_before < 0)
        pad_before = 0;
    if (pad_after < 0)
        pad_after = 0;

    auto dim = static_cast<int>(X.extent(axis));
    auto x_reduce_slice = X.slice({pad_before}, {dim - pad_after}, {axis}, {1});
    auto y_reduce_slice = X.slice({0}, {1}, {axis}, {1});
    reduce<Reducer>(x_reduce_slice, y_reduce_slice, {axis}, true);

    if (pad_before > 0) {
        auto y_edge_slice = X.slice({0}, {pad_before}, {axis}, {1});
        reorder(y_reduce_slice.broadcast_to(y_edge_slice.shape()), y_edge_slice);
    }
    if (pad_after > 0) {
        auto y_edge_slice = X.slice({-pad_after}, {dim}, {axis}, {1});
        reorder(y_reduce_slice.broadcast_to(y_edge_slice.shape()), y_edge_slice);
    }
}
} // namespace detail

/**
 * Pad a tensor.
 *
 * @param X The input tensor.
 * @param Y The output tensor.
 * @param pads List of integers indicating the number of padding elements to
 *        add or remove (if negative) at the beginning and end of each axis.
 *        For 2D it is the number of pixels. `pads` rank should be double of
 *        the input's rank. `pads` format should be as follow [x1_begin,
 *        x2_begin,...,x1_end,x2_end,...], where xi_begin, the number of pixels
 *        added at the beginning of axis `i`, and xi_end, the number of pixels
 *        added at the end of axis `i`.
 * @param mode Pad mode. One of the following values:
 *        Constant:
 *            Pads with a constant value.
 *        Edge:
 *            Pads with the edge values.
 *        Reflect:
 *            Pads with the reflection of the vector mirrored on the first
 *            and last values of the vector along each axis.
 *        Symmetric:
 *            Pads with the reflection of the vector mirrored along the
 *            edge.
 *        Wrap:
 *            Pads with the wrap of the vector along the axis. The first values
 *            are used to pad the end and the end values are used to pad the
 *            beginning.
 *        Maximum:
 *            Pads with the maximum value of all or part of the vector along
 *            each axis.
 *        Minimum:
 *            Pads with the minimum value of all or part of the vector along
 *            each axis.
 *        Mean:
 *            Pads with the mean value of all or part of the vector along
 *            each axis.
 * @param val When mode is 'constant', the value to be filled.
 */
template <typename TensorT>
enable_if_tensor<TensorT, void>
pad(const TensorT& X, tensor_type<TensorT>& Y,
    const std::vector<int>& pads, const PadMode mode = PadMode::Constant,
    const tensor_value_type<TensorT>& val = tensor_value_type<TensorT>{})
{
    // Validate pads and calculate output shape
    std::vector<size_t> y_dims;
    auto rank = X.rank();
    if (pads.size() != rank*2)
        throw std::invalid_argument("pad: the 'pads' argument has incorrect rank");
    for (int i = 0; i < rank; i++) {
        auto old_dim = static_cast<int>(X.extent(i));
        auto new_dim = old_dim + pads[i] + pads[i + rank];
        if (new_dim <= 0 || (pads[i]<0 && -pads[i]>old_dim) || (pads[i+rank]<0 && -pads[i+rank]>old_dim))
            throw shape_error("pad: the 'pads' argument contains invalid value");
        y_dims.push_back(new_dim);
    }
    Y.resize(Shape(y_dims));

    // Copy core data
    std::vector<Range> x_slice, y_slice;
    for (int i = 0; i < rank; i++) {
        int x_start = 0, x_end = X.extent(i);
        int y_start = 0, y_end = Y.extent(i);
        if (pads[i] < 0) {
            x_start = -pads[i];
        } else {
            y_start = pads[i];
        }
        if (pads[i+rank] < 0) {
            x_end += pads[i+rank];
        } else {
            y_end -= pads[i+rank];
        }
        x_slice.push_back({x_start, x_end});
        y_slice.push_back({y_start, y_end});
    }
    reorder(X.slice(x_slice), Y.slice(y_slice));

    // Padding
    switch (mode) {
    case PadMode::Constant:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_constant(Y, val, pads[axis], axis);
            detail::append_constant(Y, val, pads[axis+rank], axis);
        }
        break;

    case PadMode::Edge:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_edge(Y, pads[axis], axis);
            detail::append_edge(Y, pads[axis+rank], axis);
        }
        break;

    case PadMode::Reflect:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_reflect(Y, pads[axis], axis);
            detail::append_reflect(Y, pads[axis+rank], axis);
        }
        break;

    case PadMode::Symmetric:
        for (int axis = 0; axis < rank; axis++) {
            detail::prepend_symmetric(Y, pads[axis], axis);
            detail::append_symmetric(Y, pads[axis+rank], axis);
        }
        break;

    case PadMode::Wrap:
        for (int axis = 0; axis < rank; axis++) {
            detail::pad_wrap(Y, pads[axis], pads[axis+rank], axis);
        }
        break;

    case PadMode::Maximum:
        for (int axis = 0; axis < rank; axis++) {
            detail::pad_reduce<xfn::reduce_max<tensor_value_type<TensorT>>>(
                Y, pads[axis], pads[axis+rank], axis);
        }
        break;

    case PadMode::Minimum:
        for (int axis = 0; axis < rank; axis++) {
            detail::pad_reduce<xfn::reduce_min<tensor_value_type<TensorT>>>(
                Y, pads[axis], pads[axis+rank], axis);
        }
        break;

    case PadMode::Mean:
        for (int axis = 0; axis < rank; axis++) {
            detail::pad_reduce<xfn::reduce_mean<tensor_value_type<TensorT>>>(
                Y, pads[axis], pads[axis+rank], axis);
        }
        break;

    default:
        throw std::logic_error("pad: unsupported mode");
    }
}

template <typename TensorT>
enable_if_tensor<TensorT>
pad(const TensorT& X, const std::vector<int>& pads,
    const PadMode mode = PadMode::Constant,
    const tensor_value_type<TensorT>& val = tensor_value_type<TensorT>{})
{
    tensor_type<TensorT> Y{};
    pad(X, Y, pads, mode, val);
    return Y;
}

} // namespace dlf
