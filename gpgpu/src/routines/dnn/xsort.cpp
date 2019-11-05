#include "xsort.hpp"
#include <cassert>

namespace gpgpu { namespace dnn {
using namespace gpgpu::blas;

template <typename T>
Xsort<T>::Xsort(const Queue& queue, Event* event, const std::string& name) :
    Routine(queue, event, name, {"Xaxpy"}, PrecisionValue<T>(), {}, {
    #include "../../kernels/dnn/local_merge.cl"
    #include "../../kernels/dnn/xsort.cl"
}) {}

#define WPT 4

template <typename T>
void Xsort<T>::DoSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    if (dims.back() <= 2*db_["WGS"]) {
        DoDirectSort(dir, dims,
                     x_buffer, x_offset, x_strides,
                     y_buffer, y_offset, y_strides);
    } else {
        DoIndirectSort(dir, dims,
                       x_buffer, x_offset, x_strides,
                       y_buffer, y_offset, y_strides);
    }
}

template <typename T>
void Xsort<T>::DoDirectSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto n = dims.back();
    auto shape = PackShape(dims, x_strides, y_strides, context_, queue_);

    auto kernel = program_.getKernel("DirectSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = NextPowerOfTwo(n) / 2;
    kernel.setLocalMemorySize(local_size*2 * sizeof(T));

    auto global = std::vector<size_t>{batch_size * local_size};
    auto local  = std::vector<size_t>{local_size};
    RunKernel(kernel, queue_, device_, global, local, event_);
}

template <typename T>
void Xsort<T>::DoIndirectSort(
    const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto n = dims.back();   // sequence length
    auto k = 2*db_["WGS"];  // subsequence length, double at each step

    if (n <= 2*WPT*db_["WGS"]) {
        // Direct merge without auxiliary buffer
        DoBlockSort(n, dir, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
        while (k < n) {
            DoDirectMerge(n, k, dir, dims, y_buffer, y_offset, y_strides, y_buffer, y_offset, y_strides);
            k *= 2;
        }
        return;
    }

    // Create a contiguous shape
    std::vector<size_t> aux_strides(dims.size());
    size_t aux_size = 1;
    for (int i = dims.size()-1; i >= 0; --i) {
        aux_strides[i] = aux_size;
        aux_size *= dims[i];
    }

    // Allocate auxiliary ping-pong buffer
    auto aux = context_.getTemporaryBuffer<T>(aux_size);

    // Count number of steps in the mergesort, which will be used to determine
    // the start of ping-pong buffer
    int steps = 0;
    for (size_t t = k; t < n; t <<= 1)
        steps++;

    // Merge sorted blocks using auxiliary buffer
    if (steps & 1) {
        DoBlockSort(n, dir, dims, x_buffer, x_offset, x_strides, aux, aux.offset(), aux_strides);
        DoMerge(n, k, dir, dims, aux, aux.offset(), aux_strides, y_buffer, y_offset, y_strides);
    } else {
        DoBlockSort(n, dir, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
        DoMerge(n, k, dir, dims, y_buffer, y_offset, y_strides, aux, aux.offset(), aux_strides);
    }
}

template <typename T>
void Xsort<T>::DoBlockSort(
    const size_t n, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    auto shape = PackShape(dims, x_strides, y_strides, context_, queue_);

    auto kernel = program_.getKernel("BlockSort");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        y_buffer, static_cast<int>(y_offset), static_cast<int>(y_strides.back()));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t local_size = db_["WGS"];
    auto n_ceiled = Ceil(n, local_size*2) / 2;

    auto global = std::vector<size_t>{n_ceiled, batch_size};
    auto local  = std::vector<size_t>{local_size, 1};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xsort<T>::DoMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
    Buffer<T>& y_buffer, const size_t y_offset, const std::vector<size_t>& y_strides)
{
    if (k >= n)
        return;
    if (k <= WPT*db_["WGS"])
        DoDirectMerge(n, k, dir, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
    else
        DoIndirectMerge(n, k, dir, dims, x_buffer, x_offset, x_strides, y_buffer, y_offset, y_strides);
    DoMerge(n, k*2, dir, dims, y_buffer, y_offset, y_strides, x_buffer, x_offset, x_strides);
}

template <typename T>
void Xsort<T>::DoDirectMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_strides)
{
    auto shape = PackShape(dims, x_strides, z_strides, context_, queue_);
    auto kernel = program_.getKernel("DirectMerge");
    kernel.setArguments(
        static_cast<int>(n), dir, static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        z_buffer, static_cast<int>(z_offset), static_cast<int>(z_strides.back()));
    kernel.setLocalMemorySize((k+1) * 2*sizeof(T));

    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    auto global = std::vector<size_t>{Ceil(n, k*2)/(WPT*2) * batch_size};
    auto local  = std::vector<size_t>{k/WPT};
    RunKernel(kernel, queue_, device_, global, local, nullptr);
}

template <typename T>
void Xsort<T>::DoIndirectMerge(
    const size_t n, const size_t k, const int dir, const std::vector<size_t>& dims,
    const Buffer<T>& x_buffer, const size_t x_offset, const std::vector<size_t>& x_strides,
          Buffer<T>& z_buffer, const size_t z_offset, const std::vector<size_t>& z_strides)
{
    size_t batch_size = std::accumulate(dims.begin(), dims.end()-1, 1, std::multiplies<>());
    size_t splits = CeilDiv(n, k*2);
    size_t blocks = CeilDiv(k*2, db_["WGS"]*WPT);

    auto diag = context_.getTemporaryBuffer<int>(batch_size * splits * (2*(blocks + 1)));
    auto shape = PackShape(dims, x_strides, z_strides, context_, queue_);

    auto kernel1 = program_.getKernel("MergePath");
    kernel1.setArguments(
        static_cast<int>(n), static_cast<int>(k), dir,
        static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        diag, static_cast<int>(diag.offset()));

    auto global1 = std::vector<size_t>{32 * splits * blocks, batch_size};
    auto local1  = std::vector<size_t>{32, 1};
    RunKernel(kernel1, queue_, device_, global1, local1, nullptr);

    auto kernel2 = program_.getKernel("IndirectMerge");
    kernel2.setArguments(
        static_cast<int>(n), static_cast<int>(k), dir,
        static_cast<int>(dims.size()), shape,
        x_buffer, static_cast<int>(x_offset), static_cast<int>(x_strides.back()),
        z_buffer, static_cast<int>(z_offset), static_cast<int>(z_strides.back()),
        diag, static_cast<int>(diag.offset()));

    auto global2 = std::vector<size_t>{db_["WGS"] * splits * blocks, batch_size};
    auto local2  = std::vector<size_t>{db_["WGS"], 1};
    RunKernel(kernel2, queue_, device_, global2, local2, nullptr);
}

template class Xsort<int16_t>;
template class Xsort<int32_t>;
template class Xsort<int64_t>;
template class Xsort<half>;
template class Xsort<float>;
template class Xsort<double>;

}} // namespace gpgpu::dnn
