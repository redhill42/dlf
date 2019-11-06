#include "xtopk.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xtopk<T>::Xtopk(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/xtopk.cl"
}) {}

template <typename T>
void Xtopk<T>::DoTopK(
    const size_t limit, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto n = dims.back();
    assert(limit <= n);

    if (n <= 2*db_["WGS"] ) {
        DoDirectTopK(
            n, limit, dir, dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    } else {
        DoIndirectTopK(
            n, limit, dir, dims,
            x_buffer, x_offset, x_strides,
            y_buffer, y_offset, y_strides,
            i_buffer, i_offset, i_strides);
    }
}

template <typename T>
void Xtopk<T>::DoDirectTopK(
    const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    auto shape = PackShape(dims, x_strides, y_strides, i_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()),
        i_buffer, static_cast<int>(i_offset), static_cast<int>(i_strides.back()));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xtopk<T>::DoIndirectTopK(
    const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides,
    Buffer<int32_t>& i_buffer, const size_t i_offset, const std::vector<size_t>& i_strides)
{
    // Calculate number of subsequences
    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = 2 * db_["WGS"];
    size_t blocks = CeilDiv(n, local_size);

    // Calculate subsequence length
    auto sublen = std::min(limit, local_size);
    auto block_size = blocks * sublen;
    auto rem = n % local_size;
    if (rem < sublen)
        block_size = block_size - sublen + rem;

    // Allocate auxiliary buffer
    auto aux = context_.getTemporaryBuffer<T>(batch_size * block_size);
    auto idx = context_.getTemporaryBuffer<int32_t>(batch_size * block_size);

    // Split input sequence into subsequences, sort on subsequences and take
    // top-k elements in each subsequence
    {
        auto shape = PackShape(dims, x_strides, context_, queue_);
        auto kernel = program_.getKernel("BlockTopK");
        kernel.setArguments(
            static_cast<int>(n), static_cast<int>(sublen),static_cast<int>(block_size),
            dir, static_cast<int>(dims.size()), shape,
            x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
            aux, static_cast<int>(aux.offset()),
            idx, static_cast<int>(idx.offset()));
        auto global = std::vector<size_t>{blocks * db_["WGS"], batch_size};
        auto local  = std::vector<size_t>{db_["WGS"], 1};
        RunKernel(kernel, queue_, device_, global, local, nullptr);
    }

    if (block_size <= local_size) {
        DoCompactTopK(block_size, limit, dir, dims,
                      aux, aux.offset(), idx, idx.offset(),
                      y_buffer, y_offset, y_strides,
                      i_buffer, i_offset, i_strides);
    } else {
        assert(false);
    }
}

template <typename T>
void Xtopk<T>::DoCompactTopK(
    const size_t n, const size_t limit, const int dir, const std::vector<size_t>& dims,
    Buffer<T>&        x_buffer, const size_t  x_offset,
    Buffer<int32_t>& ix_buffer, const size_t ix_offset,
    Buffer<T>&        y_buffer, const size_t  y_offset, const std::vector<size_t>&  y_strides,
    Buffer<int32_t>& iy_buffer, const size_t iy_offset, const std::vector<size_t>& iy_strides)
{
    auto shape = PackShape(dims, y_strides, iy_strides, context_, queue_);

    auto kernel = program_.getKernel("CompactTopK");
    kernel.setArguments(
        static_cast<int>(n), static_cast<int>(limit), dir,
        static_cast<int>(dims.size()), shape,
         x_buffer, static_cast<int>( x_offset),
        ix_buffer, static_cast<int>(ix_offset),
         y_buffer, static_cast<int>( y_offset), static_cast<int>( y_strides.back()),
        iy_buffer, static_cast<int>(iy_offset), static_cast<int>(iy_strides.back()));

    auto batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    auto local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * (sizeof(T) + sizeof(int32_t)));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template class Xtopk<int16_t>;
template class Xtopk<int32_t>;
template class Xtopk<int64_t>;
template class Xtopk<half>;
template class Xtopk<float>;
template class Xtopk<double>;

}} // namespace gpgpu::dnn
