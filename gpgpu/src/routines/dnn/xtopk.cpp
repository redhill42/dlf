#include "xtopk.hpp"
#include "xargsort.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xtopk<T>::Xtopk(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/local_merge.cl"
    #include "../../kernels/dnn/xtopk.cl"
}) {}

#define WPT 4

static inline size_t remainder(size_t n, size_t m) {
    return ((n - 1) % m) + 1;
}

template <typename T>
void Xtopk<T>::DoTopK(
    const size_t limit, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto n = x_dims.back();
    assert(limit <= n);

    if (n <= 2*db_["WGS"] ) {
        DoDirectTopK(
            n, limit, dir, x_dims, y_dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    } else if (limit <= WPT*db_["WGS"]) {
        DoMergeTopK(
            n, limit, dir, x_dims, y_dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    } else {
        DoSortedTopK(
            n, limit, dir, x_dims, y_dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    }
}

template <typename T>
void Xtopk<T>::DoDirectTopK(
    const size_t n, const size_t limit, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
    auto y_shape = PackShape(y_dims, y_strides, i_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), dir,
        static_cast<int>(x_dims.size()), x_shape, y_shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        i_buffer, static_cast<int>(i_offset), static_cast<int>(i_strides.back()));

    auto batch_size = GetBatchSize(x_dims);
    auto local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xtopk<T>::DoMergeTopK(
    const size_t n, const size_t limit, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto batch_size = GetBatchSize(x_dims);
    auto block_size = 2 * db_["WGS"];
    auto blocks     = CeilDiv(n, block_size);

    auto aux = context_.getTemporaryBuffer<T>(batch_size * n * 2);
    auto idx = context_.getTemporaryBuffer<int32_t>(batch_size * n * 2);

    auto aux1_offset = aux.offset();
    auto aux2_offset = aux.offset() + batch_size * n;
    auto idx1_offset = idx.offset();
    auto idx2_offset = idx.offset() + batch_size * n;

    // Split input sequence into sorted subsequences
    DoBlockTopK(n, block_size, n, dir,
                x_dims, x_strides, x_buffer, x_offset,
                aux, aux1_offset, idx, idx1_offset);

    auto sequence_size = n;
    auto limit_ceiled  = std::max(block_size, NextPowerOfTwo(limit));

    while (blocks > 1) {
        auto next_blocks = CeilDiv(blocks, 2);
        auto next_block_size = std::min(block_size*2, limit_ceiled);
        auto next_sequence_size = (next_blocks - 1) * next_block_size;

        // handle boundary conditions
        if (block_size < limit_ceiled) {
            if (blocks & 1)
                next_sequence_size += remainder(sequence_size, block_size);
            else
                next_sequence_size += remainder(sequence_size, next_block_size);
        } else {
            if (blocks & 1)
                next_sequence_size += remainder(sequence_size, next_block_size);
            else
                next_sequence_size += next_block_size;
        }

        // Merge subsequences and select top-k
        DoMerge(sequence_size, next_sequence_size, next_block_size, dir,
                batch_size, block_size,
                aux, aux1_offset, idx, idx1_offset,
                aux, aux2_offset, idx, idx2_offset);

        blocks = next_blocks;
        block_size = next_block_size;
        sequence_size = next_sequence_size;
        std::swap(aux1_offset, aux2_offset);
        std::swap(idx1_offset, idx2_offset);
    }

    DoSelectTopK(sequence_size, limit, y_dims,
                 aux, aux1_offset, idx, idx1_offset,
                 y_buffer, y_offset, y_strides,
                 i_buffer, i_offset, i_strides);
}

template <typename T>
void Xtopk<T>::DoSortedTopK(
    const size_t n, const size_t limit, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    // Allocate auxiliary buffer
    auto batch_size = GetBatchSize(x_dims);
    auto aux = context_.getTemporaryBuffer<T>(batch_size * n);
    auto idx = context_.getTemporaryBuffer<int32_t>(batch_size * n);
    auto aux_strides = MakeFlatShape(x_dims);

    // Full sort on original sequence
    auto argsort = Xargsort<T>(queue_, nullptr);
    argsort.DoArgSort(dir, x_dims, x_buffer, x_offset, x_strides,
                      aux, aux.offset(), aux_strides,
                      idx, idx.offset(), aux_strides);

    // Select top-k elements from sorted sequence
    DoSelectTopK(n, limit, y_dims,
                 aux, aux.offset(), idx, idx.offset(),
                 y_buffer, y_offset, y_strides,
                 i_buffer, i_offset, i_strides);
}

template <typename T>
void Xtopk<T>::DoBlockTopK(
    const size_t n, const size_t limit, size_t y_len, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
          Buffer<T>& y_buffer, const size_t y_offset,
    Buffer<int32_t>& i_buffer, const size_t i_offset)
{
    auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
    auto kernel = program_.getKernel("BlockTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), static_cast<int>(y_len),
        dir, static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset),
        i_buffer, static_cast<int>(i_offset));

    auto blocks = CeilDiv(n, 2*db_["WGS"]);
    auto global = std::vector<size_t>{blocks * db_["WGS"], GetBatchSize(x_dims)};
    auto local  = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xtopk<T>::DoMerge(
    const size_t input_len,  const size_t output_len,
    const size_t limit, const int dir,
    const size_t batch_size, const size_t block_size,
    Buffer<T>&       x_buffer, const size_t x_offset,
    Buffer<int32_t>& i_buffer, const size_t i_offset,
    Buffer<T>&       y_buffer, const size_t y_offset,
    Buffer<int32_t>& j_buffer, const size_t j_offset)
{
    assert(limit <= WPT * db_["WGS"]);

    auto kernel = program_.getKernel("DoMerge");
    kernel.setArguments(
        static_cast<int>(input_len), static_cast<int>(output_len),
        static_cast<int>(limit), dir,
        x_buffer, static_cast<int>(x_offset),
        i_buffer, static_cast<int>(i_offset),
        y_buffer, static_cast<int>(y_offset),
        j_buffer, static_cast<int>(j_offset));
    kernel.setLocalMemorySize((block_size+1) * 2*(sizeof(T) + sizeof(int32_t)));

    auto local_size = block_size / WPT;
    auto blocks = CeilDiv(input_len, block_size*2);
    auto global = std::vector<size_t>{batch_size * blocks * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xtopk<T>::DoSelectTopK(
    const size_t n, const size_t limit, const std::vector<size_t>& y_dims,
    Buffer<T>&       x_buffer, const size_t x_offset,
    Buffer<int32_t>& i_buffer, const size_t i_offset,
    Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides)
{
    auto y_shape = PackShape(y_dims, y_strides, j_strides, context_, queue_);
    auto kernel = program_.getKernel("SelectTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit),
        static_cast<int>(y_dims.size()), y_shape,
        x_buffer, static_cast<int>(x_offset),
        i_buffer, static_cast<int>(i_offset),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        j_buffer, static_cast<int>(j_offset), static_cast<int>(j_strides.back()));

    auto global = std::vector<size_t>{db_["WGS"] * GetBatchSize(y_dims)};
    auto local  = std::vector<size_t>{db_["WGS"]};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtopk<int16_t>;
template class Xtopk<int32_t>;
template class Xtopk<int64_t>;
template class Xtopk<half>;
template class Xtopk<float>;
template class Xtopk<double>;

}} // namespace gpgpu::dnn
