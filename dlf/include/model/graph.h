#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>
#include <complex>
#include <cassert>

#if CPP_VER >= 17
#define HAS_VARIANT
#include <variant>
#endif

#include <gblas_half.h>

namespace dlf { namespace model {

// Graph represents one "function" of computation. It uses a simple ownership
// model where the graph owns all the nodes inside it. All references inside
// the graph are raw pointers. Destroying the Graph will invalidate any
// pointers to nodes in the graph.
class Graph;

// Node is the base class of the IR graph. It represents one computation
// and dependencies on a list of Values. The "prim-opts", so to speak.
class Node;

// A Value represents an input or output to node that is either a Tensor
// or an opaque Handle object, as determined by type().
class Value;

//==-------------------------------------------------------------------------
// Serialized tensor value
//==-------------------------------------------------------------------------

// Tensor data types, same value from ONNX
enum class DataType : int32_t {
    UNDEFINED = 0,
    FLOAT,
    UINT8,
    INT8,
    UINT16,
    INT16,
    INT32,
    INT64,
    STRING,
    BOOL,
    FLOAT16,        // IEEE754 half-precision floating-point format (16 bits wide).
    DOUBLE,
    UINT32,
    UINT64,
    COMPLEX64,      // complex with float32 real and imaginary components
    COMPLEX128,     // complex with float64 real and imaginary components
};

class Dimension {
    size_t m_value{};
    std::string m_symbol;

public:
    Dimension() = default;
    Dimension(size_t value) : m_value(value) {}
    Dimension(std::string symbol) : m_symbol(std::move(symbol)) {}

    bool has_value() const noexcept {
        return m_symbol.empty();
    }

    size_t value() const {
        if (!has_value())
            throw shape_error(cxx::string_concat(symbol(), ": missing dimension value"));
        return m_value;
    }

    void set_value(size_t value) noexcept {
        m_value = value;
        m_symbol.clear();
    }

    const std::string& symbol() const noexcept {
        return m_symbol;
    }

    operator size_t() const {
        return value();
    }

    Dimension& operator+=(size_t val) {
        set_value(value() + val);
        return *this;
    }

    Dimension& operator-=(size_t val) {
        set_value(value() - val);
        return *this;
    }

    friend bool operator==(const Dimension& x, const Dimension& y) {
        return x.m_value == y.m_value && x.m_symbol == y.m_symbol;
    }

    friend bool operator!=(const Dimension& x, const Dimension& y) {
        return !(x == y);
    }
};

class Dims {
    std::vector<Dimension> m_dims;

public:
    Dims() = default;
    explicit Dims(size_t n) : m_dims(n) {}

    Dims(const std::vector<size_t>& dims) {
        for (auto d : dims) {
            m_dims.push_back(d);
        }
    }

    Dims(std::initializer_list<size_t> il) {
        for (auto d : il) {
            m_dims.push_back(d);
        }
    }

    template <typename InputIterator>
    Dims(InputIterator first, InputIterator last)
        : m_dims(first, last) {}

    template <typename InputIterator>
    void assign(InputIterator first, InputIterator last) {
        m_dims.assign(first, last);
    }

    size_t rank() const noexcept {
        return m_dims.size();
    }

    Dimension& operator[](size_t i) noexcept {
        return m_dims[i];
    }

    const Dimension& operator[](size_t i) const noexcept {
        return m_dims[i];
    }

    size_t size() const {
        return std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<>());
    }

    size_t partial_size(size_t start, size_t end) const {
        return std::accumulate(m_dims.begin()+start, m_dims.begin()+end, 1, std::multiplies<>());
    }

    Shape shape() const {
        std::vector<size_t> extents;
        for (auto& d : m_dims)
            extents.push_back(d.value());
        return Shape(extents);
    }

    void append(Dimension dim) {
        m_dims.push_back(std::move(dim));
    }

    void insert(size_t i, Dimension dim) {
        m_dims.insert(m_dims.begin() + i, std::move(dim));
    }

    void erase(size_t i) {
        m_dims.erase(m_dims.begin() + i);
    }

    const Dimension& back() const {
        return m_dims.back();
    }

    Dimension& back() {
        return m_dims.back();
    }

    friend bool operator==(const Dims& x, const Dims& y) {
        return x.m_dims == y.m_dims;
    }

    friend bool operator!=(const Dims& x, const Dims& y) {
        return !(x == y);
    }

public:
    using value_type = typename std::vector<Dimension>::value_type;
    using reference = typename std::vector<Dimension>::reference;
    using const_reference = typename std::vector<Dimension>::const_reference;
    using pointer = typename std::vector<Dimension>::pointer;
    using const_pointer = typename std::vector<Dimension>::const_pointer;
    using size_type = typename std::vector<Dimension>::size_type;
    using difference_type = typename std::vector<Dimension>::difference_type;
    using iterator = typename std::vector<Dimension>::iterator;
    using const_iterator = typename std::vector<Dimension>::const_iterator;
    using reverse_iterator = typename std::vector<Dimension>::reverse_iterator;
    using const_reverse_iterator = typename std::vector<Dimension>::const_reverse_iterator;

    iterator begin() noexcept { return m_dims.begin(); }
    const_iterator begin() const noexcept { return m_dims.begin(); }
    iterator end() noexcept { return m_dims.end(); }
    const_iterator end() const noexcept { return m_dims.end(); }

    reverse_iterator rbegin() noexcept { return m_dims.rbegin(); }
    const_reverse_iterator rbegin() const noexcept { return m_dims.rbegin(); }
    reverse_iterator rend() noexcept { return m_dims.rend(); }
    const_reverse_iterator rend() const noexcept { return m_dims.rend(); }

    const_iterator cbegin() const noexcept { return begin(); }
    const_iterator cend() const noexcept { return end(); }
    const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    const_reverse_iterator crend() const noexcept { return rend(); }
};

template <typename T> constexpr DataType DataTypeTrait = DataType::UNDEFINED;
template <> constexpr DataType DataTypeTrait<float> = DataType::FLOAT;
template <> constexpr DataType DataTypeTrait<uint8_t> = DataType::UINT8;
template <> constexpr DataType DataTypeTrait<int8_t> = DataType::INT8;
template <> constexpr DataType DataTypeTrait<uint16_t> = DataType::UINT16;
template <> constexpr DataType DataTypeTrait<int16_t> = DataType::INT16;
template <> constexpr DataType DataTypeTrait<int32_t> = DataType::INT32;
template <> constexpr DataType DataTypeTrait<int64_t> = DataType::INT64;
template <> constexpr DataType DataTypeTrait<std::string> = DataType::STRING;
template <> constexpr DataType DataTypeTrait<bool> = DataType::BOOL;
template <> constexpr DataType DataTypeTrait<double> = DataType::DOUBLE;
template <> constexpr DataType DataTypeTrait<uint32_t> = DataType::UINT32;
template <> constexpr DataType DataTypeTrait<uint64_t> = DataType::UINT64;
template <> constexpr DataType DataTypeTrait<std::complex<float>> = DataType::COMPLEX64;
template <> constexpr DataType DataTypeTrait<std::complex<double>> = DataType::COMPLEX128;

namespace detail {
#ifdef HAS_VARIANT
struct TensorVariant {
    // Tensor content must be organized in row-major order.
    //
    // Depending on the data_type field, exactly one of the fields below  is
    // used to store the elements of the tensor.
    std::variant<
        // For float and complex64 values:
        // Complex64 tensors are encoded as a single array of floats,
        // with the real components appearing in odd numbered positions,
        // and corresponding imaginary component appearing in the
        // subsequent event numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        // is encoded as [1.0, 2.0, 3.0, 4.0]
        // When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
        std::vector<float>,

        // For double and complex128 values:
        // Complex128 tensors are encoded as a single array of doubles,
        // with the real components appearing in odd numbered positions,
        // and the corresponding imaginary component appearing in the
        // subsequent event numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
        // is encoded as [1.0, 2.0, 3.0, 4.0]
        // When this field is present, the data_type field MUST be DOUBLE or COMPLEX128.
        std::vector<double>,

        // For int32, uint8, int8, uint16, int16, bool, and float16 values:
        // float16 values must be bit-wise covered to an uint16_t prior
        // to writing to the buffer.
        // When this field is present, the data_type field must be
        // INT32, INT16, INT8, UINT16, UINT8, BOOL, or FLOAT16.
        std::vector<int32_t>,

        // For int64 values.
        // When this field is present, the data_type field MUST be INT64
        std::vector<int64_t>,

        // For uint64 and uint32 values.
        // When this field is present, the data_type field MUST be
        // UINT32 or UINT64.
        std::vector<uint64_t>,

        // For strings.
        // Each element of string_data is a UTF-8 encoded Unicode
        // string. No trailing null, no leading BOM.
        // When this field is present, the data_type field MUST be STRING
        std::vector<std::string>
    > var;

    void init(DataType type) {
        switch (type) {
        case DataType::FLOAT:
        case DataType::COMPLEX64:
            var.emplace<std::vector<float>>();
            break;

        case DataType::DOUBLE:
        case DataType::COMPLEX128:
            var.emplace<std::vector<double>>();
            break;

        case DataType::BOOL:
        case DataType::INT8:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::UINT8:
        case DataType::UINT16:
        case DataType::FLOAT16:
            var.emplace<std::vector<int32_t>>();
            break;

        case DataType::INT64:
            var.emplace<std::vector<int64_t>>();
            break;

        case DataType::UINT32:
        case DataType::UINT64:
            var.emplace<std::vector<uint64_t>>();
            break;

        case DataType::STRING:
            var.emplace<std::vector<std::string>>();
            break;

        default:
            throw std::bad_variant_access(); // FIXME
        }
    }

    void reset(DataType type) {
        init(type);
    }

    void clear() {
        std::visit([](auto& v){v.clear();}, var);
    }

    template <typename T>
    std::vector<T>& get() {
        return std::get<std::vector<T>>(var);
    }

    template <typename T>
    const std::vector<T>& get() const {
        return std::get<std::vector<T>>(var);
    }
};
#else
struct TensorVariant {
    std::vector<float> m_float_data;
    std::vector<double> m_double_data;
    std::vector<int32_t> m_int32_data;
    std::vector<int64_t> m_int64_data;
    std::vector<uint64_t> m_uint64_data;
    std::vector<std::string> m_string_data;

    void init(DataType) {}
    void reset(DataType) { clear(); }

    void clear() {
        m_float_data.clear();
        m_double_data.clear();
        m_int32_data.clear();
        m_int64_data.clear();
        m_uint64_data.clear();
        m_string_data.clear();
    }

    template <typename T>
    std::vector<T>& get();

    template <typename T>
    const std::vector<T>& get() const;
};

#define DEFINE_TENSOR_VARIANT(T, field) \
template <> inline std::vector<T>& TensorVariant::get<T>() { \
    return field; \
} \
template <> inline const std::vector<T>& TensorVariant::get<T>() const { \
    return field; \
}

DEFINE_TENSOR_VARIANT(float, m_float_data)
DEFINE_TENSOR_VARIANT(double, m_double_data)
DEFINE_TENSOR_VARIANT(int32_t, m_int32_data)
DEFINE_TENSOR_VARIANT(int64_t, m_int64_data)
DEFINE_TENSOR_VARIANT(uint64_t, m_uint64_data)
DEFINE_TENSOR_VARIANT(std::string, m_string_data)
#undef DEFINE_TENSOR_VARIANT
#endif
} // namespace detail

class TensorData final {
private:
    // The name of the tensor.
    std::string m_name;

    // The data type of the tensor.
    DataType m_type = DataType::UNDEFINED;

    // The shape of the tensor.
    Dims m_dims;

    detail::TensorVariant m_data;

    // Serializations can either use one of the fields above, or use this
    // raw bytes field. The only exception is the string case, where one is
    // required to store the content in the repeated bytes string_data field.
    //
    // When this raw_data field is used to store tensor value, elements MUST
    // be stored in as fixed-width, little-endian order.
    // Floating-point data types MUST be stored in IEEE 754 format.
    // Complex64 elements must be written as two consecutive FLOAT values,
    // real component first. Complex128 elements must be written as two
    // consecutive DOUBLE values, real component first. Boolean type MUST be
    // written one byte per tensor element (00000001 for true, 00000000 for false).
    bool m_has_raw_data = false;
    std::string m_raw_data;

    // For very large tensors, we may want to store them in chunks, in which
    // case the following fields will specify the segment that is tored in
    // the current Tensor.
    bool m_has_segment = false;
    int64_t m_segment_begin = 0;
    int64_t m_segment_end = 0;

public:
    TensorData() = default;

    TensorData(std::string name, DataType type, Dims dims)
        : m_name(std::move(name)), m_type(type), m_dims(std::move(dims))
    {
        m_data.init(type);
    }

    template <typename T>
    TensorData(std::string name, DataType type, Dims dims, std::initializer_list<T> init)
        : TensorData(name, type, dims)
    {
        set_data(init.begin(), init.end());
    }

    template <typename Iterator>
    TensorData(std::string name, Dims dims, Iterator first, Iterator last)
        : TensorData(std::move(name),
                     DataTypeTrait<typename std::iterator_traits<Iterator>::value_type>,
                     std::move(dims))
    {
        set_data(first, last);
    }

    template <typename Container>
    TensorData(std::string name, Dims dims, Container& c)
        : TensorData(std::move(name), std::move(dims), std::begin(c), std::end(c))
    {}

    template <typename T>
    TensorData(std::string name, const Tensor<T>& tensor)
        : TensorData(std::move(name), tensor.shape().extents(), tensor)
    {}

    bool has_name() const noexcept {
        return !m_name.empty();
    }

    const std::string& name() const noexcept {
        return m_name;
    }

    void set_name(std::string name) noexcept {
        m_name = std::move(name);
    }

    DataType type() const noexcept {
        return m_type;
    }

    void set_type(DataType type) {
        m_type = type;
        m_data.reset(type);
    }

    const Dims& dims() const noexcept {
        return m_dims;
    }

    void set_dims(Dims dims) noexcept {
        m_dims = std::move(dims);
    }

    size_t size() const noexcept {
        return m_dims.size();
    }

    std::vector<float>& float_data() noexcept {
        return m_data.get<float>();
    }

    const std::vector<float>& float_data() const noexcept {
        return m_data.get<float>();
    }

    std::vector<double>& double_data() noexcept {
        return m_data.get<double>();
    }

    const std::vector<double>& double_data() const noexcept {
        return m_data.get<double>();
    }

    std::vector<int32_t>& int32_data() noexcept {
        return m_data.get<int32_t>();
    }

    const std::vector<int32_t>& int32_data() const noexcept {
        return m_data.get<int32_t>();
    }

    std::vector<int64_t>& int64_data() noexcept {
        return m_data.get<int64_t>();
    }

    const std::vector<int64_t>& int64_data() const noexcept {
        return m_data.get<int64_t>();
    }

    std::vector<uint64_t>& uint64_data() noexcept {
        return m_data.get<uint64_t>();
    }

    const std::vector<uint64_t>& uint64_data() const noexcept {
        return m_data.get<uint64_t>();
    }

    std::vector<std::string>& string_data() noexcept {
        return m_data.get<std::string>();
    }

    const std::vector<std::string>& string_data() const noexcept {
        return m_data.get<std::string>();
    }

    bool has_raw_data() const noexcept {
        return m_has_raw_data;
    }

    const std::string& raw_data() const noexcept {
        return m_raw_data;
    }

    void set_raw_data(std::string raw_data) noexcept {
        m_has_raw_data = true;
        m_raw_data = std::move(raw_data);
    }

    template <typename Iterator>
    std::enable_if_t<
        std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value ||
        std::is_same<gblas::half, typename std::iterator_traits<Iterator>::value_type>::value>
    set_data(Iterator first, Iterator last) {
        if (size() != std::distance(first, last)) {
            throw std::invalid_argument("invalid tensor data size");
        }

        switch (type()) {
        case DataType::FLOAT:
            clear();
            float_data().assign(first, last);
            break;

        case DataType::DOUBLE:
            clear();
            double_data().assign(first, last);
            break;

        case DataType::FLOAT16: {
            clear();
            int32_data().resize(size());
            std::transform(first, last, int32_data().begin(), [](auto x) {
                gblas::half h = static_cast<float>(x);
                return static_cast<int32_t>(h.x);
            });
            break;
        }

        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::BOOL:
            clear();
            int32_data().assign(first, last);
            break;

        case DataType::INT64:
            clear();
            int64_data().assign(first, last);
            break;

        case DataType::UINT32:
        case DataType::UINT64:
            clear();
            uint64_data().assign(first, last);
            break;

        default:
            throw std::logic_error("unsupported tensor data type");
        }
    }

    template <typename Iterator>
    std::enable_if_t<std::is_same<std::string, typename std::iterator_traits<Iterator>::value_type>::value>
    set_data(Iterator first, Iterator last) {
        if (size() != std::distance(first, last))
            throw std::invalid_argument("invalid tensor data size");
        if (type() != DataType::STRING)
            throw std::invalid_argument("invalid tensor data type");
        clear();
        string_data().assign(first, last);
    }

    template <typename Iterator>
    std::enable_if_t<std::is_same<std::complex<float>, typename std::iterator_traits<Iterator>::value_type>::value>
    set_data(Iterator first, Iterator last) {
        if (size() != std::distance(first, last))
            throw std::invalid_argument("invalid tensor data size");
        if (type() != DataType::COMPLEX64)
            throw std::invalid_argument("invalid tensor data type");
        clear();
        float_data().resize(size()*2);
        std::copy(first, last, reinterpret_cast<std::complex<float>*>(float_data().data()));
    }

    template <typename Iterator>
    std::enable_if_t<std::is_same<std::complex<double>, typename std::iterator_traits<Iterator>::value_type>::value>
    set_data(Iterator first, Iterator last) {
        if (size() != std::distance(first, last))
            throw std::invalid_argument("invalid tensor data size");
        if (type() != DataType::COMPLEX128)
            throw std::invalid_argument("invalid tensor data type");
        clear();
        double_data().resize(size()*2);
        std::copy(first, last, reinterpret_cast<std::complex<double>*>(double_data().data()));
    }

    void clear() {
        m_data.clear();
        m_raw_data.clear();
        m_has_raw_data = false;
    }

    template <typename T>
    Tensor<T> decode() const;
};

template <typename T>
Tensor<T> TensorData::decode() const {
    auto shape = m_dims.shape();

    if (has_raw_data()) {
        void* raw = const_cast<void*>(reinterpret_cast<const void*>(raw_data().data()));

        switch (type()) {
        case DataType::FLOAT:
            return Tensor<float>::wrap(shape, reinterpret_cast<float*>(raw)).cast<T>();
        case DataType::DOUBLE:
            return Tensor<double>::wrap(shape, reinterpret_cast<double*>(raw)).cast<T>();
        case DataType::FLOAT16:
            return Tensor<gblas::half>::wrap(shape, reinterpret_cast<gblas::half*>(raw)).cast<T>();
        case DataType::UINT8:
            return Tensor<uint8_t>::wrap(shape, reinterpret_cast<uint8_t*>(raw)).cast<T>();
        case DataType::INT8:
            return Tensor<int8_t>::wrap(shape, reinterpret_cast<int8_t*>(raw)).cast<T>();
        case DataType::UINT16:
            return Tensor<uint16_t>::wrap(shape, reinterpret_cast<uint16_t*>(raw)).cast<T>();
        case DataType::INT16:
            return Tensor<int16_t>::wrap(shape, reinterpret_cast<int16_t*>(raw)).cast<T>();
        case DataType::INT32:
            return Tensor<int32_t>::wrap(shape, reinterpret_cast<int32_t*>(raw)).cast<T>();
        case DataType::BOOL:
            return Tensor<bool>::wrap(shape, reinterpret_cast<bool*>(raw)).cast<T>();
        case DataType::INT64:
            return Tensor<int64_t>::wrap(shape, reinterpret_cast<int64_t*>(raw)).cast<T>();
        case DataType::UINT32:
            return Tensor<uint32_t>::wrap(shape, reinterpret_cast<uint32_t*>(raw)).cast<T>();
        case DataType::UINT64:
            return Tensor<uint64_t>::wrap(shape, reinterpret_cast<uint64_t*>(raw)).cast<T>();
        default:
            throw std::logic_error("invalid tensor data type");
        }
    } else {
        switch (type()) {
        case DataType::FLOAT:
            return Tensor<float>::wrap(shape, const_cast<float*>(float_data().data())).cast<T>();

        case DataType::DOUBLE:
            return Tensor<double>::wrap(shape, const_cast<double*>(double_data().data())).cast<T>();

        case DataType::FLOAT16: {
            Tensor<gblas::half> temp(shape);
            std::transform(int32_data().begin(), int32_data().end(), temp.begin(), [](auto x) {
                gblas::half h;
                h.x = static_cast<unsigned short>(x);
                return h;
            });
            return temp.cast<T>();
        }

        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::BOOL:
            return Tensor<int32_t>::wrap(shape, const_cast<int32_t*>(int32_data().data())).cast<T>();

        case DataType::INT64:
            return Tensor<int64_t>::wrap(shape, const_cast<int64_t*>(int64_data().data())).cast<T>();

        case DataType::UINT32:
        case DataType::UINT64:
            return Tensor<uint64_t>::wrap(shape, const_cast<uint64_t*>(uint64_data().data())).cast<T>();

        default:
            throw std::logic_error("invalid tensor data type");
        }
    }
}

template <>
inline Tensor<std::string> TensorData::decode() const {
    if (type() != DataType::STRING)
        throw std::logic_error("invalid tensor data type");
    return Tensor<std::string>{m_dims.shape(), string_data().begin(), string_data().end()};
}

template <>
inline Tensor<std::complex<float>> TensorData::decode() const {
    if (type() != DataType::COMPLEX64)
        throw std::logic_error("invalid tensor data type");

    const std::complex<float>* data;
    size_t size;
    if (has_raw_data()) {
        data = reinterpret_cast<const std::complex<float>*>(raw_data().data());
        size = raw_data().size() / sizeof(std::complex<float>);
    } else {
        data = reinterpret_cast<const std::complex<float>*>(float_data().data());
        size = float_data().size() / 2;
    }
    return Tensor<std::complex<float>>{m_dims.shape(), data, data+size};
}

template <>
inline Tensor<std::complex<double>> TensorData::decode() const {
    if (type() != DataType::COMPLEX128)
        throw std::logic_error("invalid tensor data type");

    const std::complex<double>* data;
    size_t size;
    if (has_raw_data()) {
        data = reinterpret_cast<const std::complex<double>*>(raw_data().data());
        size = raw_data().size() / sizeof(std::complex<double>);
    } else {
        data = reinterpret_cast<const std::complex<double>*>(float_data().data());
        size = float_data().size() / 2;
    }
    return Tensor<std::complex<double>>{m_dims.shape(), data, data+size};
}

//==-------------------------------------------------------------------------
// Node Attributes
//==-------------------------------------------------------------------------

enum class AttributeKind : uint8_t {
    UNDEFINED = 0,
    FLOAT,
    INT,
    STRING,
    TENSOR,
    GRAPH,

    FLOATS,
    INTS,
    STRINGS,
    TENSORS,
    GRAPHS,
};

template <AttributeKind kind>
struct AttributeKindTrait { using type = void; };

template <AttributeKind kind>
using AttributeType = typename AttributeKindTrait<kind>::type;

#define DEFINE_ATTRIBUTE_TYPE(K, T) \
template <> struct AttributeKindTrait<AttributeKind::K> { \
    using type = T; \
}; \
template <> struct AttributeKindTrait<AttributeKind::K##S> { \
    using type = std::vector<T>; \
};

DEFINE_ATTRIBUTE_TYPE(FLOAT, float)
DEFINE_ATTRIBUTE_TYPE(INT, int64_t)
DEFINE_ATTRIBUTE_TYPE(STRING, std::string)
DEFINE_ATTRIBUTE_TYPE(TENSOR, TensorData)
DEFINE_ATTRIBUTE_TYPE(GRAPH, std::shared_ptr<Graph>)
#undef DEFINE_ATTRIBUTE_TYPE

#ifdef HAS_VARIANT

using bad_variant_access = std::bad_variant_access;

// CRTP so that Node which inherits Attributes can be return for
// method chaining e.g:
// Node* n = g->create(kSelect)->set_i(kOffset, 3)->set_f(kValue, 3.5);
// we return Derived* pointers because Nodes are normally held as pointers.
template <typename Derived>
class Attributes {
private:
    // Keep the same order in AttributeKind enum
    using AttributeVariant = std::variant<
        AttributeType<AttributeKind::FLOAT>,
        AttributeType<AttributeKind::INT>,
        AttributeType<AttributeKind::STRING>,
        AttributeType<AttributeKind::TENSOR>,
        AttributeType<AttributeKind::GRAPH>,
        AttributeType<AttributeKind::FLOATS>,
        AttributeType<AttributeKind::INTS>,
        AttributeType<AttributeKind::STRINGS>,
        AttributeType<AttributeKind::TENSORS>,
        AttributeType<AttributeKind::GRAPHS>
    >;

    class AttributeValue {
        Symbol m_name;
        AttributeVariant m_value;

    public:
        AttributeValue(Symbol name, AttributeVariant&& value)
            : m_name(name), m_value(std::move(value))
        {}

        Symbol name() const noexcept {
            return m_name;
        }

        AttributeKind kind() const noexcept {
            return static_cast<AttributeKind>(m_value.index() + 1);
        }

        template <typename T>
        const T& value() const {
            return std::get<T>(m_value);
        }

        template <typename T>
        void value(T&& value) noexcept {
            m_value = std::forward<T>(value);
        }
    };

    // NB: For determinism, we use a vector rather than a hash map. This does
    // mean that lookups are O(n), so you shouldn't use Attributes to store
    // a big pile of messages.
    std::vector<AttributeValue> m_values;

public:
    Attributes() = default;

    bool hasAttributes() const noexcept {
        return m_values.size() > 0;
    }

    bool hasAttribute(Symbol name) const noexcept {
        return find(name, false) != m_values.end();
    }

    std::vector<Symbol> attributeNames() const noexcept {
        std::vector<Symbol> names(m_values.size());
        std::transform(m_values.begin(), m_values.end(), names.begin(), std::mem_fn(&AttributeValue::name));
        return names;
    }

    AttributeKind attributeKind(Symbol name) const noexcept {
        auto it = find(name, false);
        if (it != m_values.end())
            return it->kind();
        return AttributeKind::UNDEFINED;
    }

    bool removeAttribute(Symbol name) noexcept {
        auto it = find(name, false);
        if (it != m_values.end()) {
            m_values.erase(it);
            return true;
        }
        return false;
    }

    #define AT(k) AttributeType<AttributeKind::k>
    #define CREATE_ACCESSOR(kind, method) \
    Derived& set_##method(Symbol name, AT(kind) v) noexcept \
        { return set<AT(kind)>(name, std::move(v)); } \
    const AT(kind)& get_##method(Symbol name) const \
        { return get<AT(kind)>(name); } \
    AT(kind) get_##method(Symbol name, AT(kind) default_value) const \
        { return get_or_default<AT(kind)>(name, std::move(default_value)); }

    CREATE_ACCESSOR(FLOAT, f)
    CREATE_ACCESSOR(FLOATS, fs)
    CREATE_ACCESSOR(INT, i)
    CREATE_ACCESSOR(INTS, is)
    CREATE_ACCESSOR(STRING, s);
    CREATE_ACCESSOR(STRINGS, ss)
    CREATE_ACCESSOR(TENSOR, t)
    CREATE_ACCESSOR(TENSORS, ts)
    CREATE_ACCESSOR(GRAPH, g)
    CREATE_ACCESSOR(GRAPHS, gs)

    #undef CREATE_ACCESSOR
    #undef AT

private:
    Derived* This() noexcept {
        return static_cast<Derived*>(this);
    }

    auto find(Symbol name, bool required) noexcept {
        auto it = std::find_if(m_values.begin(), m_values.end(),
            [&](const auto& v) { return v.name() == name; });
        assert(!required || it != m_values.end());
        return it;
    }

    auto find(Symbol name, bool required) const noexcept {
        auto it = std::find_if(m_values.begin(), m_values.end(),
            [&](const auto& v) { return v.name() == name; });
        assert(!required || it != m_values.end());
        return it;
    }

    template <typename T>
    Derived& set(Symbol name, T&& value) noexcept {
        auto it = find(name, false);
        if (it == m_values.end()) {
            m_values.emplace_back(name, std::forward<T>(value));
        } else {
            it->template value<T>(std::forward<T>(value));
        }
        return *This();
    }

    template <typename T>
    const T& get(Symbol name) const {
        return find(name, true)->template value<T>(); // may throw bad_variant_access
    }

    template <typename T>
    T get_or_default(Symbol name, T&& default_value) const {
        auto it = find(name, false);
        if (it == m_values.end()) {
            return std::forward<T>(default_value);
        } else {
            return it->template value<T>(); // may throw bad_variant_access
        }
    }
};

#else // !HAS_VARIANT

class bad_variant_access : public std::exception {
public:
    const char* what() const noexcept override {
        return "bad_variant_access";
    }
};

template <typename Derived>
class Attributes {
private:
    struct AttributeValue {
        virtual ~AttributeValue() = default;
        virtual Symbol name() const noexcept = 0;
        virtual AttributeKind kind() const noexcept  = 0;
    };

    template <AttributeKind K>
    struct GenericAttributeValue : AttributeValue {
        static constexpr AttributeKind Kind = K;
        using ValueType = AttributeType<Kind>;

        Symbol m_name;
        ValueType m_value;

        GenericAttributeValue(Symbol name, ValueType&& value)
            : m_name(name), m_value(std::move(value)) {}

        Symbol name() const noexcept override {
            return m_name;
        }

        AttributeKind kind() const noexcept override {
            return Kind;
        }

        ValueType& value() noexcept {
            return m_value;
        }
    };

    using FloatAttribute   = GenericAttributeValue<AttributeKind::FLOAT>;
    using FloatsAttribute  = GenericAttributeValue<AttributeKind::FLOATS>;
    using IntAttribute     = GenericAttributeValue<AttributeKind::INT>;
    using IntsAttribute    = GenericAttributeValue<AttributeKind::INTS>;
    using StringAttribute  = GenericAttributeValue<AttributeKind::STRING>;
    using StringsAttribute = GenericAttributeValue<AttributeKind::STRINGS>;
    using TensorAttribute  = GenericAttributeValue<AttributeKind::TENSOR>;
    using TensorsAttribute = GenericAttributeValue<AttributeKind::TENSORS>;
    using GraphAttribute   = GenericAttributeValue<AttributeKind::GRAPH>;
    using GraphsAttribute  = GenericAttributeValue<AttributeKind::GRAPHS>;

    std::vector<std::unique_ptr<AttributeValue>> m_values;

public:
    Attributes() = default;

    bool hasAttributes() const noexcept {
        return m_values.size() > 0;
    }

    bool hasAttribute(Symbol name) const noexcept {
        return find(name, false) != m_values.end();
    }

    std::vector<Symbol> attributeNames() const noexcept {
        std::vector<Symbol> names(m_values.size());
        std::transform(m_values.begin(), m_values.end(), names.begin(), std::mem_fn(&AttributeValue::name));
        return names;
    }

    AttributeKind attributeKind(Symbol name) const noexcept {
        auto it = find(name, false);
        if (it != m_values.end())
            return (*it)->kind();
        return AttributeKind::UNDEFINED;
    }

    bool removeAttribute(Symbol name) noexcept {
        auto it = find(name, false);
        if (it != m_values.end()) {
            m_values.erase(it);
            return true;
        }
        return false;
    }

    #define AT(kind) typename kind##Attribute::ValueType
    #define CREATE_ACCESSOR(kind, method) \
    Derived& set_##method(Symbol name, AT(kind) v) noexcept { \
        return set<kind##Attribute>(name, std::move(v)); \
    } \
    const AT(kind)& get_##method(Symbol name) const { \
        return get<kind##Attribute>(name, kind##Attribute::Kind); \
    } \
    AT(kind) get_##method(Symbol name, AT(kind) default_value) const { \
        return get_or_default<kind##Attribute>(name, kind##Attribute::Kind, std::move(default_value)); \
    }

    CREATE_ACCESSOR(Float, f)
    CREATE_ACCESSOR(Floats, fs)
    CREATE_ACCESSOR(Int, i)
    CREATE_ACCESSOR(Ints, is)
    CREATE_ACCESSOR(String, s)
    CREATE_ACCESSOR(Strings, ss)
    CREATE_ACCESSOR(Tensor, t)
    CREATE_ACCESSOR(Tensors, ts)
    CREATE_ACCESSOR(Graph, g)
    CREATE_ACCESSOR(Graphs, gs)

    #undef CREATE_ACCESSOR
    #undef AT

private:
    Derived* This() noexcept {
        return static_cast<Derived*>(this);
    }

    auto find(Symbol name, bool required) noexcept {
        auto it = std::find_if(m_values.begin(), m_values.end(),
            [&](const auto& v) { return v->name() == name; });
        assert(!required || it != m_values.end());
        return it;
    }

    auto find(Symbol name, bool required) const noexcept {
        auto it = std::find_if(m_values.begin(), m_values.end(),
            [&](const auto& v) { return v->name() == name; });
        assert(!required || it != m_values.end());
        return it;
    }

    template <typename AT, typename V = typename AT::ValueType>
    Derived& set(Symbol name, V&& v) {
        auto it = find(name, false);
        if (it == m_values.end()) {
            m_values.emplace_back(new AT(name, std::move(v)));
        } else {
            *it = std::make_unique<AT>(name, std::move(v));
        }
        return *This();
    }

    template <typename AT, typename V = typename AT::ValueType>
    const V& get(Symbol name, AttributeKind kind) const {
        auto a = find(name, true)->get();
        if (a->kind() != kind)
            throw bad_variant_access();
        return static_cast<AT*>(a)->value();
    }

    template <typename AT, typename V = typename AT::ValueType>
    V get_or_default(Symbol name, AttributeKind kind, V&& default_value) const {
        auto it = find(name, false);
        if (it == m_values.end()) {
            return std::forward<V>(default_value);
        } else if ((*it)->kind() != kind) {
            throw bad_variant_access();
        } else {
            return static_cast<AT*>(it->get())->value();
        }
    }
};

#endif // HAS_VARIANT

//==-------------------------------------------------------------------------

/**
 * Each use is represented by this type, see Node::uses().
 * 'user' is the consumer of the value, offset is the index into
 * 'user's input where the produces will be found.
 */
struct Use final {
    Node* user;
    size_t offset;

    Use(Node* user, size_t offset) : user(user), offset(offset) {}

    bool operator==(const Use& b) const noexcept {
        return user == b.user && offset == b.offset;
    }
};

// The list types are intentionally simple, but we type-def
// them here so if we need to change them, refactoring will be easier.
using ValueList = std::vector<Value*>;
using UseList = std::vector<Use>;
using NodeKind = Symbol;

using GraphNodeList = GenericNodeList<Node>;
using ConstGraphNodeList = GenericNodeList<const Node>;
using GraphNodeListIterator = GenericNodeListIterator<Node>;
using ConstGraphNodeListIterator = GenericNodeListIterator<const Node>;

//==-------------------------------------------------------------------------

class Value final {
    friend class Node;
    friend class Graph;

    Node*           m_node;
    size_t          m_offset;
    std::string     m_name;
    DataType        m_type;
    Dims            m_dims;
    UseList         m_uses;
    bool            m_has_initializer = false;
    TensorData      m_initializer;

public:
    Value(Node* node, size_t offset, std::string name);

    Value(const Value&) = delete;
    Value& operator=(const Value&) = delete;

    Node* node() noexcept {
        return m_node;
    }

    const Node* node() const noexcept {
        return m_node;
    }

    Graph* owningGraph() noexcept;
    const Graph* owningGraph() const noexcept;

    size_t offset() const noexcept {
        return m_offset;
    }

    std::string name() const noexcept {
        return m_name;
    }

    DataType type() const noexcept {
        return m_type;
    }

    Value* set_type(DataType type) noexcept {
        m_type = type;
        return this;
    }

    const Dims& dims() const noexcept {
        return m_dims;
    }

    Dims& dims() noexcept {
        return m_dims;
    }

    size_t dim(size_t i) const noexcept {
        return m_dims[i];
    }

    Value* set_dims(Dims dims) noexcept {
        m_dims = std::move(dims);
        return this;
    }

    const UseList& uses() const noexcept {
        return m_uses;
    }

    UseList& uses() noexcept {
        return m_uses;
    }

    bool has_initializer() const noexcept;
    const TensorData& initializer() const noexcept;
    TensorData& initializer() noexcept;
    Value* set_initializer(TensorData initializer) noexcept;

    // Replaces all uses of this node with 'newValue'.
    //
    // Given:   %3 = f(%1, %2)
    //          %4 = g(%3)
    //          %5 = h(%3, %3)
    // Execute: %3.replaceAllUsesWith(%6)
    // Result:  %3 = f(%1, %2)
    //          %4 = g(%6)
    //          %5 = h(%6, %6)
    void replaceAllUsesWith(Value* newValue);
};

//==-------------------------------------------------------------------------

// Forward declaration of all operators. These operators are defined in "model/operators.h"
#define FORWARD_DECLARE(op) class op;
FORALL_OPERATORS(FORWARD_DECLARE)
#undef FORWARD_DECLARE

/**
 * The generic visitor for the graph node.
 */
class Visitor {
public:
#define DEFINE_VISITOR(op) virtual void visit(op*) = 0;
    FORALL_OPERATORS(DEFINE_VISITOR)
#undef DEFINE_VISITOR

    virtual void visit(Node*) = 0;
    virtual ~Visitor() = default;
};

class NodeFactory {
public:
    virtual Node* createNode(Graph* graph, NodeKind kind) = 0;
    virtual Value* createValue(Node* node, size_t offset, std::string&& name) = 0;
    virtual void freeNode(Node* node) = 0;
    virtual void freeValue(Value* value) = 0;
    virtual ~NodeFactory() = default;

    static std::unique_ptr<NodeFactory> newInstance();
};

class ShapeInference {
public:
    virtual void infer(Node* n) = 0;
    virtual void infer(Graph& g) = 0;
    virtual ~ShapeInference() = default;

    static std::unique_ptr<ShapeInference> newInstance(
        const std::unordered_map<std::string, size_t>& env = {});
};

class ShapeInferenceError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    ShapeInferenceError(const std::string& message) : std::runtime_error(message) {}
};

//==-------------------------------------------------------------------------

class Node : public Attributes<Node> {
    friend class Graph;
    friend class Value;
    friend GraphNodeList;
    friend ConstGraphNodeList;
    friend GraphNodeListIterator;
    friend ConstGraphNodeListIterator;

    // Each node but Return/Param is associated with exactly one place in the
    // node list of the graph. This circular is a doubly-linked list, the Return
    // node is used as the sentinel for the beginning and end of the list such
    // that the list never has null pointers.
    //   next_in_graph[0] is next pointer
    //   next_in_graph[1] is prev pointer
    // using an array to allow the same iterator class for forward and reverse
    // node list.
    //
    // This list represents a topological sort.
    Node* next_in_graph[2] = { nullptr, nullptr };
    Node*& next() noexcept { return next_in_graph[kNextDirection]; }
    Node*& prev() noexcept { return next_in_graph[kPrevDirection]; }
    Node* const& next() const noexcept { return next_in_graph[kNextDirection]; }
    Node* const& prev() const noexcept { return next_in_graph[kPrevDirection]; }

    Graph* m_graph;
    const NodeKind m_kind;

    ValueList m_inputs;
    ValueList m_outputs;

    std::string m_name;
    std::string m_domain;
    std::string m_doc_string;

public:
    Node(Graph* graph, NodeKind kind);

    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;

    Graph* owningGraph() noexcept {
        return m_graph;
    }

    const Graph* owningGraph() const noexcept {
        return m_graph;
    }

    NodeKind kind() const noexcept {
        return m_kind;
    }

    bool has_name() const noexcept {
        return !m_name.empty();
    }

    const std::string& name() const noexcept {
        return m_name;
    }

    void set_name(std::string name) noexcept {
        m_name = std::move(name);
    }

    bool has_domain() const noexcept {
        return !m_domain.empty();
    }

    const std::string& domain() const noexcept {
        return m_domain;
    }

    void set_domain(std::string domain) noexcept {
        m_domain = std::move(domain);
    }

    bool has_doc_string() const noexcept {
        return !m_doc_string.empty();
    }

    const std::string& doc_string() const noexcept {
        return m_doc_string;
    }

    void set_doc_string(std::string doc_string) noexcept {
        m_doc_string = std::move(doc_string);
    }

    // Note: This returns an array_ref; that means that it will
    // get invalidated if you resize inputs (e.g., using addInput)
    // We can't return a std::vector<Node*>& because there's no
    // way to soundly cast to std::vector<const Node*> (an insane
    // implementation of std::vector could make this representationally
    // different.)

    cxx::array_ref<Value*> inputs() noexcept {
        return m_inputs;
    }

    cxx::array_ref<const Value*> inputs() const noexcept {
        // Vectors are not convertible in const-ness of elements, but
        // raw pointers are.
        return cxx::array_ref<const Value*>(m_inputs.data(), m_inputs.size());
    }

    cxx::array_ref<Value*> outputs() noexcept {
        return m_outputs;
    }

    cxx::array_ref<const Value*> outputs() const noexcept {
        return cxx::array_ref<const Value*>(m_outputs.data(), m_outputs.size());
    }

    bool hasUses() const noexcept {
        for (auto o : outputs())
            if (!o->uses().empty())
                return true;
        return false;
    }

    void replaceAllUsesWith(Node* n) noexcept {
        assert(outputs().size() == n->m_outputs.size());
        size_t nOutputs = m_outputs.size();
        for (size_t i = 0; i < nOutputs; i++) {
            m_outputs[i]->replaceAllUsesWith(n->m_outputs[i]);
        }
    }

    // Lots of things like chunk have a single input or single output,
    // so we have a helper to make accessing it easier.

    Value* input() noexcept {
        return m_inputs.at(0);
    }

    Value* output() noexcept {
        return m_outputs.at(0);
    }

    const Value* input() const noexcept {
        return m_inputs.at(0);
    }

    const Value* output() const noexcept {
        return m_outputs.at(0);
    }

    // Access a particular input or output. Null is returned if no such value.

    Value* input(size_t i) noexcept {
        if (i < m_inputs.size() && m_inputs[i]->node()->kind() != kUndefined)
            return m_inputs[i];
        return nullptr;
    }

    const Value* input(size_t i) const noexcept {
        if (i < m_inputs.size() && m_inputs[i]->node()->kind() != kUndefined)
            return m_inputs[i];
        return nullptr;
    }

    Value* output(size_t i) noexcept {
        return i < m_outputs.size() ? m_outputs[i] : nullptr;
    }

    const Value* output(size_t i) const noexcept {
        return i < m_outputs.size() ? m_outputs[i] : nullptr;
    }

    // Graphs

    // Note [Topological invariant]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We always maintain an up-to-date topological ordering of all nodes
    // via the next()/prev() links.  All transformations to graphs must
    // preserve this topological ordering: for example, it is only valid
    // to 'addInput' with an input which is topologically before the
    // current node.
    //
    // Usually, it is obvious whether or not topological order is maintained;
    // for example, if you are adding nodes to the end of the topsort, it's
    // impossible for them to refer to inputs that are not in the topsort.
    // If it is not obvious, please comment accordingly.

    /**
     * Add 'node' as an input to 'this' at the end of existing arguments.
     * Returns the added node for ease of chaining.
     *
     * Given:   %3 = f(%1, %2)
     * Execute: %3.addInput(%4)
     * Result:  %3 = f(%1, %2, %4)
     */
    Value* addInput(Value* node) noexcept {
        node->uses().emplace_back(this, m_inputs.size());
        m_inputs.push_back(node);
        return node;
    }

    /**
     * Replace the input of 'this' at position 'i' with
     * 'newValue', returning the old node.
     *
     * Given:   %3 = f(%1, %2)
     * Execute: %3.replaceInput(1, %4)
     * Result:  %3 = f(%1, %4)
     */
    Value* replaceInput(size_t i, Value* newValue) noexcept {
        Value* old = dropInput(i);
        m_inputs[i] = newValue;
        newValue->uses().emplace_back(this, i);
        return old;
    }

    /**
     * Replace all occurrences of 'from' in the inputs of this
     * node with 'to'.
     *
     * Given:   %3 = f(%1, %2, %1)
     * Execute: %3.replaceInputWith(%1, %4)
     * Result:  %3 = f(%4, %2, %4)
     */
    void replaceInputWith(Value* from, Value* to) noexcept {
        size_t i = 0;
        for (auto input : inputs()) {
            if (input == from)
                replaceInput(i, to);
            i++;
        }
    }

    /**
     * Remove the input at 'i' from this node.
     *
     * WARNING: This is O(n) in the number of inputs, so avoid repeatedly calling
     * removeInput.
     *
     * Given:   %3 = f(%1, %2)
     * Execute: %3.eraseInput(1)
     * Result:  %3 = f(%1)
     */
    void eraseInput(size_t i) noexcept {
        dropInput(i);
        // everything after this input shifts left, so we need to update
        // their use offsets to match
        for (size_t j = i+1; j < m_inputs.size(); j++) {
            auto it = findUseForInput(j);
            it->offset--;
        }
        m_inputs.erase(m_inputs.begin() + i);
    }

    /**
     * Remove all inputs from a node.
     *
     * Given:   %3 = f(%1, %2)
     * Execute: %3.removeAllInputs()
     * Result:  %3 = f()
     */
    void eraseAllInputs() noexcept {
        for (size_t i = 0; i < inputs().size(); ++i)
            dropInput(i);
        m_inputs.clear();
    }

    Value* addOutput(std::string name) noexcept;

    Value* addOutput(std::string name, DataType type, Dims dims) noexcept {
        return addOutput(std::move(name))->set_type(type)->set_dims(std::move(dims));
    }

    void eraseOutput(size_t i) noexcept;

    /**
     * Insert unattached 'this' node after 'n' in the topological order.
     * Returns this (for chaining).
     *
     * Given:   %3 = f(%1, %2)
     *          %4 = g(%3)
     * and unattached: %5 = h(%1)
     * Execute: %5.insertBefore(%4)
     * Result:  %3 = f(%1, %2)
     *          %5 = h(%1)
     *          %4 = g(%3)
     */
    Node* insertBefore(Node* n) noexcept {
        assert(n->inGraphList());
        insertAfter(n->prev());
        return this;
    }

    /**
     * Insert unattached 'this' node after 'n' in the topological order.
     * Returns this (for chaining).
     *
     * Given:   %3 = f(%1, %2)
     *          %4 = g(%3)
     * and unattached: %5 = h(%1)
     * Execute: %5.insertAfter(%4)
     * Result:  %3 = f(%1, %2)
     *          %4 = g(%3)
     *          %5 = h(%1)
     */
    Node* insertAfter(Node* n) noexcept {
        assert(!inGraphList() && n->inGraphList());
        Node* next = n->next();
        n->next() = this;
        this->prev() = n;
        this->next() = next;
        next->prev() = this;
        return this;
    }

    /**
     * Move 'this' (already in the graph) before 'n' in the topological order.
     *
     * Given:   %2 = f(%1)
     *          %3 = g(%1)
     * Execute: %3.moveBefore(%2)
     * Result:  %3 = g(%1)
     *          %2 = f(%1)
     */
    void moveBefore(Node* n) noexcept {
        removeFromList();
        insertBefore(n);
    }

    /**
     * Move 'this' (already in the graph) after 'n' in the topological order.
     *
     * Given:   %2 = f(%1)
     *          %3 = g(%1)
     * Execute: %2.moveAfter(%3)
     * Result:  %3 = g(%1)
     *          %2 = f(%1)
     */
    void moveAfter(Node* n) noexcept {
        removeFromList();
        insertAfter(n);
    }

    /**
     * Check whether this node is before node n in the graph.
     */
    bool isBefore(Node* n) noexcept;

    // Iterators of the node list starting at this node.
    // Useful for resuming a search starting at this node.
    GraphNodeListIterator iterator() noexcept {
        return {this, kNextDirection};
    }

    GraphNodeListIterator reverse_iterator() noexcept {
        return iterator().reverse();
    }

    ConstGraphNodeListIterator iterator() const noexcept {
        return {this, kNextDirection};
    }

    ConstGraphNodeListIterator reverse_iterator() const noexcept {
        return iterator().reverse();
    }

    /**
     * Remove 'this' from the instruction list and deallocate it.
     *
     * Invariant: no outputs of 'this' may have any uses.
     *
     * Given:   %2 = f(%1)
     *          %3 = g(%1)
     * Execute: %2.destroy()
     * Result:  %3 = g(%1)
     */
    void destroy() noexcept;

    /**
     * Dynamically cast this node to the subclass indicated by the
     * template variable, returning nullptr if the cast is invalid.
     *
     * Example usage: if (auto s = n.cast<Select>()) { ... }
     */
    template <typename T>
    T* cast() noexcept {
        if (T::Kind == kind())
            return static_cast<T*>(this);
        return nullptr;
    }

    template <typename T>
    T* expect() {
        assert(T::Kind == kind());
        return static_cast<T*>(this);
    }

    virtual void accept(Visitor& visitor) {
        visitor.visit(this);
    }

    virtual ~Node() = default;

private:
    /**
     * Lookup iterator in use list of i that corresponds to its use of this.
     */
    UseList::iterator findUseForInput(size_t i) noexcept {
        auto& input_uses = m_inputs[i]->m_uses;
        auto use_it = std::find(input_uses.begin(), input_uses.end(), Use(this, i));
        assert(use_it != input_uses.end());
        return use_it;
    }

    /**
     * Remove the use of input i, this sets input i to nullptr, but
     * is only used internally to Node before setting it to a new value
     * or erasing the entry from the list.
     */
    Value* dropInput(size_t i) noexcept {
        assert(i < m_inputs.size());
        auto input_node = m_inputs[i];
        auto use_it = findUseForInput(i);
        input_node->m_uses.erase(use_it);
        m_inputs[i] = nullptr;
        return input_node;
    }

    bool inGraphList() const noexcept {
        assert(next() != nullptr || prev() == nullptr);
        return next() != nullptr;
    }

    void removeFromList() noexcept {
        assert(inGraphList());
        Node* next = this->next();
        Node* prev = this->prev();
        prev->next() = next;
        next->prev() = prev;
        this->next() = nullptr;
        this->prev() = nullptr;
    }
};

//==-------------------------------------------------------------------------

class Graph {
    friend class Node;
    friend class Value;

    std::unique_ptr<NodeFactory> m_factory;

    // Holds outputs in a way that can be reflected as a Use object.
    // Also used as the beginning/end of the circular node list to avoid
    // having corner cases where the list is empty.
    Node* const m_output;
    Node* const m_input;
    Node* const m_undefined;

    std::string m_name;
    std::string m_doc_string;

public:
    Graph(std::unique_ptr<NodeFactory> factory = NodeFactory::newInstance()) :
        m_factory(std::move(factory)),
        m_output(initOutput(createNode(kReturn))),
        m_input(createNode(kParam)),
        m_undefined(createNode(kUndefined))
    {
        m_undefined->addOutput("");
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    bool has_name() const noexcept {
        return !m_name.empty();
    }

    const std::string& name() const noexcept {
        return m_name;
    }

    void set_name(std::string name) noexcept {
        m_name = std::move(name);
    }

    bool has_doc_string() const noexcept {
        return !m_doc_string.empty();
    }

    const std::string& doc_string() const noexcept {
        return m_doc_string;
    }

    void set_doc_string(std::string doc_string) noexcept {
        m_doc_string = std::move(doc_string);
    }

    cxx::array_ref<Value*> inputs() noexcept {
        return m_input->outputs();
    }

    cxx::array_ref<const Value*> inputs() const noexcept {
        return static_cast<const Node*>(m_input)->outputs();
    }

    Value* input(size_t id) noexcept {
        for (auto v : inputs())
            if (!v->has_initializer() && id-- == 0)
                return v;
        return nullptr;
    }

    const Value* input(size_t id) const noexcept {
        for (auto v : inputs())
            if (!v->has_initializer() && id-- == 0)
                return v;
        return nullptr;
    }

    Value* input(const std::string& name) noexcept {
        for (auto v : inputs())
            if (v->name() == name)
                return v;
        return nullptr;
    }

    const Value* input(const std::string& name) const noexcept {
        for (auto v : inputs())
            if (v->name() == name)
                return v;
        return nullptr;
    }

    cxx::array_ref<Value*> outputs() noexcept {
        return m_output->inputs();
    }

    cxx::array_ref<const Value*> outputs() const noexcept {
        return static_cast<const Node*>(m_output)->inputs();
    }

    Value* output(size_t id) noexcept {
        return m_output->input(id);
    }

    const Value* output(size_t id) const noexcept {
        return m_output->input(id);
    }

    Value* output(const std::string& name) noexcept {
        for (auto v : outputs())
            if (v->name() == name)
                return v;
        return nullptr;
    }

    const Value* output(const std::string& name) const noexcept {
        for (auto v : outputs())
            if (v->name() == name)
                return v;
        return nullptr;
    }

    GraphNodeList nodes() noexcept {
        return GraphNodeList(m_output, kNextDirection);
    }

    ConstGraphNodeList nodes() const noexcept {
        return ConstGraphNodeList(m_output, kNextDirection);
    }

    GraphNodeListIterator      begin()        noexcept { return nodes().begin();  }
    ConstGraphNodeListIterator begin()  const noexcept { return nodes().begin();  }
    GraphNodeListIterator      end()          noexcept { return nodes().end();    }
    ConstGraphNodeListIterator end()    const noexcept { return nodes().end();    }
    GraphNodeListIterator      rbegin()       noexcept { return nodes().rbegin(); }
    ConstGraphNodeListIterator rbegin() const noexcept { return nodes().rbegin(); }
    GraphNodeListIterator      rend()         noexcept { return nodes().rend();   }
    ConstGraphNodeListIterator rend()   const noexcept { return nodes().rend();   }

    Node* returnNode() noexcept {
        return m_output;
    }

    const Node* returnNode() const noexcept {
        return m_output;
    }

    Value* undefinedValue() noexcept {
        return m_undefined->output();
    }

    const Value* undefinedValue() const noexcept {
        return m_undefined->output();
    }

    Value* addInput(std::string name) noexcept {
        return m_input->addOutput(std::move(name));
    }

    Value* addInput(std::string name, DataType type, Dims dims) noexcept {
        return m_input->addOutput(std::move(name), type, std::move(dims));
    }

    Value* addInitializer(TensorData data) noexcept {
        Value* v = addInput(data.name(), data.type(), data.dims());
        v->set_initializer(std::move(data));
        return v;
    }

    void eraseInput(size_t i) noexcept {
        m_input->eraseOutput(i);
    }

    void eraseInput(Value* v) {
        eraseInput(v->offset());
    }

    Value* addOutput(Value* n) noexcept {
        return m_output->addInput(n);
    }

    void eraseOutput(size_t i) noexcept {
        m_output->eraseInput(i);
    }

    Node* createNode(NodeKind kind) {
        return m_factory->createNode(this, kind);
    }

    Node* appendNode(Node* n) noexcept {
        assert(n->owningGraph() == this && !n->inGraphList());
        n->insertBefore(m_output);
        return n;
    }

    Node* prependNode(Node* n) noexcept {
        assert(n->owningGraph() == this && !n->inGraphList());
        n->insertAfter(m_output);
        return n;
    }

    template <typename T>
    std::enable_if_t<std::is_base_of<Node, T>::value, T*> create() noexcept {
        return static_cast<T*>(createNode(T::Kind));
    }

    template <typename T>
    std::enable_if_t<std::is_base_of<Node, T>::value, T*> append() noexcept {
        return static_cast<T*>(appendNode(create<T>()));
    }

    template <typename T>
    std::enable_if_t<std::is_base_of<Node, T>::value, T*> prepend() noexcept {
        return static_cast<T*>(prependNode(create<T>()));
    }

private:
    // should only be called in the constructor
    static Node* initOutput(Node* p) {
        return p->next() = p->prev() =  p;
    }
};

//==-------------------------------------------------------------------------
// Implementation
//==-------------------------------------------------------------------------

inline Value::Value(Node* node, size_t offset, std::string name)
    : m_node(node), m_offset(offset), m_name(std::move(name)),
      m_type(DataType::UNDEFINED)
{}

inline Graph* Value::owningGraph() noexcept {
    return node()->owningGraph();
}

inline const Graph* Value::owningGraph() const noexcept {
    return node()->owningGraph();
}

inline bool Value::has_initializer() const noexcept {
    return m_has_initializer || m_node->kind() == kConstant;
}

inline const TensorData& Value::initializer() const noexcept {
    if (m_node->kind() == kConstant)
        return m_node->get_t(kvalue);
    return m_initializer;
}

inline TensorData& Value::initializer() noexcept {
    if (m_node->kind() == kConstant)
        return const_cast<TensorData&>(m_node->get_t(kvalue));
    return m_initializer;
}

inline Value* Value::set_initializer(TensorData initializer) noexcept {
    m_has_initializer = true;
    m_initializer = std::move(initializer);
    return this;
}

inline void Value::replaceAllUsesWith(Value* newValue) {
    assert(owningGraph() == newValue->owningGraph());
    for (auto u : uses()) {
        u.user->m_inputs[u.offset] = newValue;
        newValue->m_uses.push_back(u);
    }
    m_uses.clear();
}

inline Node::Node(Graph* graph, NodeKind kind) :
    m_graph(graph), m_kind(kind)
{}

inline Value* Node::addOutput(std::string name) noexcept {
    auto v = owningGraph()->m_factory->createValue(this, m_outputs.size(), std::move(name));
    m_outputs.push_back(v);
    return v;
}

inline void Node::eraseOutput(size_t i) noexcept {
    assert(i < m_outputs.size());
    assert(m_outputs[i]->uses().empty());
    Value* n = m_outputs[i];
    m_outputs.erase(m_outputs.begin() + i);
    owningGraph()->m_factory->freeValue(n);
    for (size_t j = i; j < m_outputs.size(); ++j) {
        m_outputs[j]->m_offset--;
    }
}

inline bool Node::isBefore(Node* n) noexcept {
    if (n == nullptr || this == n) {
        // Bail out early
        return false;
    }
    if (kind() == kParam) {
        // return true if node is Param (in initializers)
        return true;
    }
    if (n->kind() == kParam) {
        // return false if target node is Param (in initializers)
        return false;
    }
    assert(n->inGraphList());
    for (Node* p = next(); p != *m_graph->end(); p = p->next()) {
        if (p == n)
            return true;
    }
    return false;
}

inline void Node::destroy() noexcept {
    assert(inGraphList());
    while (!outputs().empty())
        eraseOutput(outputs().size() - 1);
    eraseAllInputs();
    removeFromList();
    m_graph->m_factory->freeNode(this);
}

}} // namespace dlf::model
