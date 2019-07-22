#pragma once

#include "gpgpu_cu.hpp"

#define checkCUDNN(expression)                                      \
  {                                                                 \
    cudnnStatus_t status = (expression);                            \
    if (status != CUDNN_STATUS_SUCCESS) {                           \
      throw APIError(status, "cudnn", cudnnGetErrorString(status)); \
    }                                                               \
  }

namespace gpgpu { namespace dnn {

inline cudnnHandle_t cudnn_handle(const Queue& queue) {
    auto& q = static_cast<const cu::cuQueue&>(queue.raw());
    return q.getCudnnHandle();
}

template <typename T>
constexpr cudnnDataType_t cudnnDataType = CUDNN_DATA_FLOAT;

template <> constexpr cudnnDataType_t cudnnDataType<half> = CUDNN_DATA_HALF;
template <> constexpr cudnnDataType_t cudnnDataType<float> = CUDNN_DATA_FLOAT;
template <> constexpr cudnnDataType_t cudnnDataType<double> = CUDNN_DATA_DOUBLE;
template <> constexpr cudnnDataType_t cudnnDataType<int32_t> = CUDNN_DATA_INT32;

template <typename T>
struct TensorDescriptor {
    cudnnTensorDescriptor_t desc;
    operator const cudnnTensorDescriptor_t() const { return desc; }
    operator cudnnTensorDescriptor_t() { return desc; }

    TensorDescriptor() {
        cudnnCreateTensorDescriptor(&desc);
    }

    explicit TensorDescriptor(size_t n, size_t c, size_t h, size_t w) {
        checkCUDNN(cudnnCreateTensorDescriptor(&desc));
        checkCUDNN(cudnnSetTensor4dDescriptor(
            desc, CUDNN_TENSOR_NCHW, cudnnDataType<T>, n, c, h, w));
    }

    explicit TensorDescriptor(const std::vector<size_t> dims) {
        int  rank = dims.size();
        int* extents = reinterpret_cast<int*>(alloca(rank * sizeof(int)));
        int* strides = reinterpret_cast<int*>(alloca(rank * sizeof(int)));

        for (int i = rank, size = 1; --i >= 0; ) {
            extents[i] = static_cast<int>(dims[i]);
            strides[i] = size;
            size *= dims[i];
        }

        checkCUDNN(cudnnCreateTensorDescriptor(&desc));
        checkCUDNN(cudnnSetTensorNdDescriptor(
            desc, cudnnDataType<T>, rank, extents, strides));
    }

    ~TensorDescriptor() {
        cudnnDestroyTensorDescriptor(desc);
    }
};

template <typename T>
struct FilterDescriptor {
    cudnnFilterDescriptor_t desc;
    operator const cudnnFilterDescriptor_t() const { return desc; }

    FilterDescriptor(size_t num_kernels, size_t channels, size_t kernel_h, size_t kernel_w) {
        checkCUDNN(cudnnCreateFilterDescriptor(&desc));
        checkCUDNN(cudnnSetFilter4dDescriptor(
            desc, cudnnDataType<T>, CUDNN_TENSOR_NCHW,
            num_kernels, channels, kernel_h, kernel_w));
    }

    ~FilterDescriptor() { cudnnDestroyFilterDescriptor(desc); }
};

template <typename T>
struct ConvolutionDescriptor {
    cudnnConvolutionDescriptor_t desc;
    operator const cudnnConvolutionDescriptor_t() { return desc; }

    ConvolutionDescriptor(size_t pad_h, size_t pad_w,
                          size_t stride_h, size_t stride_w,
                          size_t dilation_h, size_t dilation_w) {
        checkCUDNN(cudnnCreateConvolutionDescriptor(&desc));
        checkCUDNN(cudnnSetConvolution2dDescriptor(
            desc, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            CUDNN_CROSS_CORRELATION, cudnnDataType<T>));
    }

    ~ConvolutionDescriptor() { cudnnDestroyConvolutionDescriptor(desc); }
};

struct PoolingDescriptor {
    cudnnPoolingDescriptor_t desc;
    operator const cudnnPoolingDescriptor_t() { return desc; }

    PoolingDescriptor(cudnnPoolingMode_t mode, size_t kernel_h, size_t kernel_w,
                      size_t pad_h, size_t pad_w, size_t stride_h, size_t stride_w)
    {
        checkCUDNN(cudnnCreatePoolingDescriptor(&desc));
        checkCUDNN(cudnnSetPooling2dDescriptor(
            desc, mode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w));
    }

    ~PoolingDescriptor() { cudnnDestroyPoolingDescriptor(desc); }
};

struct LRNDescriptor {
    cudnnLRNDescriptor_t desc;
    operator const cudnnLRNDescriptor_t() { return desc; }

    LRNDescriptor(unsigned nsize, double alpha, double beta, double bias) {
        checkCUDNN(cudnnCreateLRNDescriptor(&desc));
        checkCUDNN(cudnnSetLRNDescriptor(desc, nsize, alpha, beta, bias));
    }

    ~LRNDescriptor() { cudnnDestroyLRNDescriptor(desc); }
};

}} // namespace gpgpu::dnn
