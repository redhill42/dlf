#include "xargsort.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xargsort<T>::Xargsort(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/local_merge.cl"
    #include "../../kernels/dnn/xsort.cl"
}) {}

#define WPT 4

template <typename T>
void Xargsort<T>::DoArgSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto y_buffer = context_.getTemporaryBuffer<T>(GetSize(dims));
    auto y_strides = MakeFlatShape(dims);
    DoArgSort(dir, dims, x_buffer, x_offset, x_strides,
                         y_buffer, y_buffer.offset(), y_strides,
                         i_buffer, i_offset, i_strides);
}

template <typename T>
void Xargsort<T>::DoArgSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    if (dims.back() <= 2*db_["WGS"]) {
        DoDirectArgSort(
            dir, dims, x_buffer, x_offset, x_strides,
                       y_buffer, y_offset, y_strides,
                       i_buffer, i_offset, i_strides);
    } else {
        DoIndirectArgSort(
            dir, dims, x_buffer, x_offset, x_strides,
                       y_buffer, y_offset, y_strides,
                       i_buffer, i_offset, i_strides);
    }
}

template <typename T>
void Xargsort<T>::DoDirectArgSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto n          = dims.back();
    auto batch_size = GetBatchSize(dims);
    auto local_size = NextPowerOfTwo(n) / 2;

    auto shape = PackShape(dims, x_strides, y_strides, i_strides, context_, queue_);
    auto kernel = program_.getKernel("DirectArgSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        i_buffer, static_cast<int>(i_offset), static_cast<int>(i_strides.back()));
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xargsort<T>::DoIndirectArgSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto n = dims.back();
    auto k = 2*db_["WGS"];

    // Create a contiguous shape
    std::vector<size_t> aux_strides(dims.size());
    size_t aux_size = 1;
    for (int i = dims.size()-1; i >= 0; --i) {
        aux_strides[i] = aux_size;
        aux_size *= dims[i];
    }

    // Allocate auxiliary ping-pong buffer
    auto aux = context_.getTemporaryBuffer<T>(aux_size);
    auto idx = context_.getTemporaryBuffer<int32_t>(aux_size);

    // Count number of steps in the mergesort, which will be used to determine
    // the start of ping-pong buffer
    int steps = 0;
    for (size_t t = k; t < n; t <<= 1)
        steps++;

    // Merge sorted blocks using auxiliary buffer
    if (steps & 1) {
        DoBlockArgSort(
            n, dir, dims,
            x_buffer, x_offset, x_strides,
            aux, aux.offset(), aux_strides,
            idx, idx.offset(), aux_strides);
        DoArgMerge(
            n, k, dir, dims,
            aux, aux.offset(), aux_strides,
            idx, idx.offset(), aux_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    } else {
        DoBlockArgSort(
            n, dir, dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
        DoArgMerge(
            n, k, dir, dims,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides,
            aux, aux.offset(), aux_strides,
            idx, idx.offset(), aux_strides);
    }
}

template <typename T>
void Xargsort<T>::DoBlockArgSort(
    const size_t n, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto shape = PackShape(dims, x_strides, y_strides, i_strides, context_, queue_);
    auto kernel = program_.getKernel("BlockArgSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        i_buffer, static_cast<int>(i_offset), static_cast<int>(i_strides.back()));

    auto n_ceiled = Ceil(n, 2*db_["WGS"]) / 2;
    auto global   = std::vector<size_t>{n_ceiled, GetBatchSize(dims)};
    auto local    = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xargsort<T>::DoArgMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
    Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
    Buffer<T>&        y_buffer, const size_t  y_offset, const std::vector<size_t>&  y_strides,
    Buffer<int32_t>& iy_buffer, const size_t iy_offset, const std::vector<size_t>& iy_strides)
{
    if (k >= n)
        return;
    if (k <= WPT*db_["WGS"])
        DoDirectArgMerge(
            n, k, dir, dims,
            x_buffer, x_offset, x_strides, ix_buffer, ix_offset, ix_strides,
            y_buffer, y_offset, y_strides, iy_buffer, iy_offset, iy_strides);
    else
        DoIndirectArgMerge(
            n, k, dir, dims,
            x_buffer, x_offset, x_strides, ix_buffer, ix_offset, ix_strides,
            y_buffer, y_offset, y_strides, iy_buffer, iy_offset, iy_strides);
    DoArgMerge(
            n, k*2, dir, dims,
            y_buffer, y_offset, y_strides, iy_buffer, iy_offset, iy_strides,
            x_buffer, x_offset, x_strides, ix_buffer, ix_offset, ix_strides);
}

template <typename T>
void Xargsort<T>::DoDirectArgMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
    Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
    Buffer<T>&        z_buffer, const size_t  z_offset, const std::vector<size_t>&  z_strides,
    Buffer<int32_t>& iz_buffer, const size_t iz_offset, const std::vector<size_t>& iz_strides)
{
    auto shape1 = PackShape(dims, x_strides, z_strides, context_, queue_);
    auto shape2 = PackShape(dims, ix_strides, iz_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectArgMerge");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape1, shape2,
         x_buffer, static_cast<int>( x_offset), static_cast<int>( x_strides.back()),
        ix_buffer, static_cast<int>(ix_offset), static_cast<int>(ix_strides.back()),
         z_buffer, static_cast<int>( z_offset), static_cast<int>( z_strides.back()),
        iz_buffer, static_cast<int>(iz_offset), static_cast<int>(iz_strides.back()));
    kernel.setLocalMemorySize((k+1) * 2*sizeof(T));

    auto local_size = k / WPT;
    auto blocks = CeilDiv(n, k*2);
    auto global = std::vector<size_t>{blocks * local_size * GetBatchSize(dims)};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xargsort<T>::DoIndirectArgMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    Buffer<T>&        x_buffer, const size_t  x_offset, const std::vector<size_t>&  x_strides,
    Buffer<int32_t>& ix_buffer, const size_t ix_offset, const std::vector<size_t>& ix_strides,
    Buffer<T>&        z_buffer, const size_t  z_offset, const std::vector<size_t>&  z_strides,
    Buffer<int32_t>& iz_buffer, const size_t iz_offset, const std::vector<size_t>& iz_strides)
{
    auto batch_size = GetBatchSize(dims);
    auto splits = CeilDiv(n, k*2);
    auto blocks = CeilDiv(k*2, db_["WGS"]*WPT);

    auto shape1 = PackShape(dims, x_strides, z_strides, context_, queue_);
    auto shape2 = PackShape(dims, ix_strides, iz_strides, context_, queue_);
    auto diag = context_.getTemporaryBuffer<int>(batch_size * splits * (2*(blocks + 1)));

    {
        auto kernel = program_.getKernel("MergePath");
        kernel.setArguments(
            static_cast<int>(n), static_cast<int>(k), dir,
            static_cast<int>(dims.size()), shape1,
            x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
            diag, static_cast<int>(diag.offset()));
        auto global = std::vector<size_t>{32 * splits * blocks, batch_size};
        auto local  = std::vector<size_t>{32, 1};
        RunKernel(kernel, queue_, device_, global, local, nullptr);
    }

    {
        auto kernel = program_.getKernel("IndirectArgMerge");
        kernel.setArguments(
            static_cast<int>(n), static_cast<int>(k), dir,
            static_cast<int>(dims.size()), shape1, shape2,
             x_buffer, static_cast<int>( x_offset), static_cast<int>( x_strides.back()),
            ix_buffer, static_cast<int>(ix_offset), static_cast<int>(ix_strides.back()),
             z_buffer, static_cast<int>( z_offset), static_cast<int>( z_strides.back()),
            iz_buffer, static_cast<int>(iz_offset), static_cast<int>(iz_strides.back()),
            diag, static_cast<int>(diag.offset()));
        auto global = std::vector<size_t>{db_["WGS"] * splits * blocks, batch_size};
        auto local  = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, nullptr);
    }
}

template class Xargsort<int16_t>;
template class Xargsort<int32_t>;
template class Xargsort<int64_t>;
template class Xargsort<half>;
template class Xargsort<float>;
template class Xargsort<double>;

}} // namespace gpgpu::dnn
