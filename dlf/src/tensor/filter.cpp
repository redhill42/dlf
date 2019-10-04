#include "tensor.h"

namespace dlf { namespace dnn {

Filter2D::Filter2D(const Shape& input_shape, const Shape& kernel_shape, size_t group) {
    set_shape(input_shape, kernel_shape, group);

    m_pad_top = m_pad_left = m_pad_bottom = m_pad_right = 0;
    m_stride_h = m_stride_w = 1;
    m_dilation_h = m_dilation_w = 1;
}

Filter2D::Filter2D(const Shape& input_shape, size_t kernel_h, size_t kernel_w) {
    set_shape(input_shape);

    m_num_kernels = m_channels;
    m_kernel_h = kernel_h;
    m_kernel_w = kernel_w;
    m_group = 1;

    m_pad_top = m_pad_left = m_pad_bottom = m_pad_right = 0;
    m_stride_h = m_stride_w = 1;
    m_dilation_h = m_dilation_w = 1;
}

void Filter2D::set_shape(const Shape& input_shape, const Shape& kernel_shape, size_t group) noexcept {
    assert(input_shape.rank() == 4);
    assert(kernel_shape.rank() == 4);
    assert(input_shape.extent(1) == kernel_shape.extent(1)*group);
    assert(kernel_shape.extent(0) % group == 0);

    m_batches = input_shape.extent(0);
    m_channels = input_shape.extent(1);
    m_height = input_shape.extent(2);
    m_width = input_shape.extent(3);

    m_num_kernels = kernel_shape.extent(0);
    m_kernel_h = kernel_shape.extent(2);
    m_kernel_w = kernel_shape.extent(3);
    m_group = group;
}

void Filter2D::set_shape(const Shape& input_shape) noexcept {
    assert(input_shape.rank() == 4);
    m_batches = input_shape.extent(0);
    m_channels = input_shape.extent(1);
    m_height = input_shape.extent(2);
    m_width = input_shape.extent(3);
}

static void pad(const std::string& mode,
                int size, int kernel, int stride, int dilation,
                size_t* pad_begin, size_t* pad_end)
{
    assert(size > 0 && kernel > 0 && stride > 0 && dilation > 0);

    int kernel_size = (kernel - 1) * dilation + 1;
    int output_size = (size - 1) / stride + 1;
    int padding = (output_size - 1) * stride + kernel_size - size;
    if (padding < 0) padding = 0;
    int half_pad = padding >> 1;

    if (mode == "SAME_UPPER") {
        *pad_begin = half_pad;
        *pad_end = padding - half_pad;
    } else {
        *pad_begin = padding - half_pad;
        *pad_end = half_pad;
    }
}

Filter2D& Filter2D::auto_pad(const std::string& mode) {
    if (mode == "VALID") {
        m_pad_top = m_pad_left = m_pad_bottom = m_pad_right = 0;
        return *this;
    }
    if (mode == "SAME_UPPER" || mode == "SAME_LOWER") {
        pad(mode, height(), kernel_h(), stride_h(), dilation_h(), &m_pad_top, &m_pad_bottom);
        pad(mode, width(), kernel_w(), stride_w(), dilation_w(), &m_pad_left, &m_pad_right);
    }
    return *this;
}

}} // namespace dlf::dnn
