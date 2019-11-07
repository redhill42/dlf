#include "xtopk.hpp"
#include "xargsort.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xtopk<T>::Xtopk(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xtopk.cl"
}) {}

template <typename T>
inline size_t Xtopk<T>::split(size_t n, size_t limit) {
    auto local_size = 2 * db_["WGS"];
    limit = std::min(limit, local_size);

    auto blocks = CeilDiv(n, local_size);
    auto block_size = blocks * limit;

    auto rem = n % local_size;
    if (rem < limit)
        block_size = block_size - limit + rem;
    if (block_size > n)
        block_size = n;
    return block_size;
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
    } else if (limit <= db_["WGS"]) {
        DoBlockCompactTopK(
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
void Xtopk<T>::DoBlockCompactTopK(
    const size_t n, const size_t limit, const int dir,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& y_dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto batch_size     = GetBatchSize(x_dims);
    auto local_size     = 2 * db_["WGS"];
    auto blocks         = CeilDiv(n, local_size);
    auto seq_size       = split(n, limit);
    auto subseq_size    = seq_size <= local_size ? 0 : split(seq_size, limit);

    auto aux = context_.getTemporaryBuffer<T>(batch_size * (seq_size + subseq_size));
    auto idx = context_.getTemporaryBuffer<int32_t>(batch_size * (seq_size + subseq_size));

    auto aux1_offset = aux.offset();
    auto idx1_offset = idx.offset();
    auto aux2_offset = aux1_offset + batch_size * seq_size;
    auto idx2_offset = idx1_offset + batch_size * seq_size;

    DoBlockTopK(n, limit, dir, blocks, seq_size, x_dims, x_strides,
                x_buffer, x_offset, aux, aux1_offset, idx, idx1_offset);

    while (seq_size > local_size) {
        auto kernel = program_.getKernel("BlockCompactTopK");
        kernel.setArguments(
            static_cast<int>(seq_size), static_cast<int>(limit),
            static_cast<int>(subseq_size), dir,
            aux, static_cast<int>(aux1_offset), idx, static_cast<int>(idx1_offset),
            aux, static_cast<int>(aux2_offset), idx, static_cast<int>(idx2_offset));

        auto global = std::vector<size_t>{blocks * db_["WGS"], batch_size};
        auto local  = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, nullptr);

        seq_size = subseq_size;
        subseq_size = split(seq_size, limit);
        std::swap(aux1_offset, aux2_offset);
        std::swap(idx1_offset, idx2_offset);
    }

    DoCompactTopK(seq_size, limit, dir, y_dims,
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
    // Calculate number of subsequences
    const auto batch_size = GetBatchSize(x_dims);
    const auto local_size = 2 * db_["WGS"];
    const auto blocks     = CeilDiv(n, local_size);
    const auto seq_size   = split(n, limit);

    // Allocate auxiliary buffer
    auto aux = context_.getTemporaryBuffer<T>(batch_size * seq_size);
    auto idx = context_.getTemporaryBuffer<int32_t>(batch_size * seq_size);

    // Create flat shape for auxiliary buffer
    auto aux_dims = x_dims;
    aux_dims.back() = seq_size;
    auto aux_strides = MakeFlatShape(aux_dims);

    if (seq_size < n) {
        // Split input sequence into subsequences, sort on subsequences and take
        // top-k elements in each subsequence
        DoBlockTopK(n, std::min(limit, local_size), dir, blocks, seq_size,
                    x_dims, x_strides, x_buffer, x_offset,
                    aux, aux.offset(), idx, idx.offset());

        // Full sort on compact sequence
        auto argsort = Xargsort<T>(queue_, nullptr);
        argsort.DoArgSort(dir, aux_dims,
                          aux, aux.offset(), aux_strides,
                          idx, idx.offset(), aux_strides,
                          aux, aux.offset(), aux_strides,
                          idx, idx.offset(), aux_strides);
    } else {
        // Full sort on original sequence
        auto argsort = Xargsort<T>(queue_, nullptr);
        argsort.DoArgSort(dir, x_dims,
                          x_buffer, x_offset, x_strides,
                          aux, aux.offset(), aux_strides,
                          idx, idx.offset(), aux_strides);
    }

    // Take top-k elements from sorted sequence
    DoSelectTopK(n, limit, y_dims,
                 aux, aux.offset(), idx, idx.offset(),
                 y_buffer, y_offset, y_strides,
                 i_buffer, i_offset, i_strides);
}

template <typename T>
void Xtopk<T>::DoBlockTopK(
    const size_t n, const size_t limit, const int dir,
    const size_t blocks, const size_t block_size,
    const std::vector<size_t>& x_dims, const std::vector<size_t>& x_strides,
    const Buffer<T>& x_buffer, const size_t x_offset,
          Buffer<T>& y_buffer, const size_t y_offset,
    Buffer<int32_t>& i_buffer, const size_t i_offset)
{
    auto x_shape = PackShape(x_dims, x_strides, context_, queue_);
    auto kernel = program_.getKernel("BlockTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), static_cast<int>(block_size),
        dir, static_cast<int>(x_dims.size()), x_shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset),
        i_buffer, static_cast<int>(i_offset));

    auto global = std::vector<size_t>{blocks * db_["WGS"], GetBatchSize(x_dims)};
    auto local  = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xtopk<T>::DoCompactTopK(
    const size_t n, const size_t limit, const int dir, const std::vector<size_t>& y_dims,
    Buffer<T>&       x_buffer, const size_t x_offset,
    Buffer<int32_t>& i_buffer, const size_t i_offset,
    Buffer<T>&       y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& j_buffer, const size_t j_offset, const std::vector<size_t>& j_strides)
{
    auto y_shape = PackShape(y_dims, y_strides, j_strides, context_, queue_);
    auto kernel = program_.getKernel("CompactTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), dir,
        static_cast<int>(y_dims.size()), y_shape,
        x_buffer, static_cast<int>(x_offset),
        i_buffer, static_cast<int>(i_offset),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        j_buffer, static_cast<int>(j_offset), static_cast<int>(j_strides.back()));

    auto local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));
    auto global = std::vector<size_t>{local_size * GetBatchSize(y_dims)};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
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
