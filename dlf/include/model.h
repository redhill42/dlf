#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>
#include <complex>
#include <cassert>

#if __cplusplus >= 201703L
#define HAS_VARIANT
#include <variant>
#endif

#include "model/intern.h"
#include "tensor.h"

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

class TensorData {
private:
    // The shape of the tensor.
    std::vector<size_t> m_dims;

    // The data type of the tensor.
    DataType m_data_type = DataType::UNDEFINED;

    // Optionally, a name for the tensor.
    bool m_has_name = false;
    std::string m_name;

    // For very large tensors, we may want to store them in chunks, in which
    // case the following fields will specify the segment that is tored in
    // the current Tensor.
    bool m_has_segment = false;
    int64_t m_segment_begin = 0;
    int64_t m_segment_end = 0;

    // Tensor content must be organized in row-major order.
    //
    // Depending on the data_type field, exactly one of the fields below with
    // name ending in _data is used to store the elements of the tensor.

    // For float and complex64 values:
    // Complex64 tensors are encoded as a single array of floats,
    // with the real components appearing in odd numbered positions,
    // and corresponding imaginary component appearing in the
    // subsequent event numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    // is encoded as [1.0, 2.0, 3.0, 4.0]
    // When this field is present, the data_type field MUST be FLOAT or COMPLEX64.
    std::vector<float> m_float_data;

    // For double
    // Complex128 tensors are encoded as a single array of doubles,
    // with the real components appearing in odd numbered positions,
    // and the corresponding imaginary component appearing in the
    // subsequent event numbered position. (e.g., [1.0 + 2.0i, 3.0 + 4.0i]
    // is encoded as [1.0, 2.0, 3.0, 4.0]
    // When this field is present, the data_type field MUST be DOUBLE or COMPLEX128.
    std::vector<double> m_double_data;

    // For int32, uint8, int8, uint16, int16, bool, and float16 values:
    // float16 values must be bit-wise covered to an uint16_t prior
    // to writing to the buffer.
    // When this field is present, the data_type field must be
    // INT32, INT16, INT8, UINT16, UINT8, BOOL, or FLOAT16.
    std::vector<int32_t> m_int32_data;

    // For int64 values.
    // When this field is present, the data_type field MUST be INT64
    std::vector<int64_t> m_int64_data;

    // For uint64 and uint32 values.
    // When this field is present, the data_type field MUST be
    // UINT32 or UINT64.
    std::vector<uint64_t> m_uint64_data;

    // For strings.
    // Each element of string_data is a UTF-8 encoded Unicode
    // string. No trailing null, no leading BOM.
    // When this field is present, the data_type field MUST be STRING
    std::vector<std::string> m_string_data;

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

public:
    TensorData() = default;

    TensorData(std::vector<size_t> dims, DataType type)
        : m_dims(std::move(dims)), m_data_type(type) {}

    template <typename Iterator>
    TensorData(std::vector<size_t> dims, Iterator first, Iterator last)
        : m_dims(dims)
    {
        m_data_type = DataTypeTrait<typename std::iterator_traits<Iterator>::value_type>;
        set_data(first, last);
    }

    template <typename T>
    TensorData(Tensor<T> tensor) {
        m_dims.resize(tensor.rank());
        for (size_t i = 0; i < tensor.rank(); i++) {
            m_dims[i] = tensor.extent(i);
        }

        m_data_type = DataTypeTrait<T>;
        set_data(tensor.begin(), tensor.end());
    }

    const std::vector<size_t>& dims() const noexcept {
        return m_dims;
    }

    void set_dims(std::vector<size_t> dims) noexcept {
        m_dims = std::move(dims);
    }

    size_t size() const noexcept {
        return std::accumulate(m_dims.begin(), m_dims.end(), 1, std::multiplies<size_t>());
    }

    DataType type() const noexcept {
        return m_data_type;
    }

    void set_type(DataType type) noexcept {
        m_data_type = type;
    }

    bool has_name() const noexcept {
        return m_has_name;
    }

    const std::string& name() const noexcept {
        return m_name;
    }

    void set_name(std::string name) noexcept {
        m_has_name = true;
        m_name = std::move(name);
    }

    template <typename Iterator>
    void set_data(Iterator first, Iterator last) {
        size_t size = std::distance(first, last);
        if (size != this->size())
            throw std::invalid_argument("invalid tensor data size");

        if (DataTypeTrait<typename std::iterator_traits<Iterator>::value_type> != type())
            throw std::invalid_argument("invalid tensor data type");

        switch (type()) {
        case DataType::FLOAT:
            m_float_data.resize(size);
            std::copy(first, last, m_float_data.begin());
            break;

        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::BOOL:
            m_int32_data.resize(size);
            std::transform(first, last, m_int32_data.begin(),
                [](auto x) { return static_cast<int32_t>(x); });
            break;

        case DataType::INT64:
            m_int64_data.resize(size);
            std::copy(first, last, m_int64_data.begin());
            break;

        case DataType::STRING:
            m_string_data.resize(size);
            std::copy(first, last, m_string_data.begin());
            break;

        case DataType::DOUBLE:
            m_double_data.resize(size);
            std::copy(first, last, m_double_data.begin());
            break;

        case DataType::UINT32:
            m_uint64_data.resize(size);
            std::transform(first, last, m_uint64_data.begin(),
                [](auto x) { return static_cast<uint64_t>(x); });
            break;

        case DataType::UINT64:
            m_uint64_data.resize(size);
            std::copy(first, last, m_uint64_data.begin());
            break;

        case DataType::COMPLEX64:
            m_float_data.resize(size*2);
            std::copy(first, last, reinterpret_cast<std::complex<float>*>(m_float_data.data()));
            break;

        case DataType::COMPLEX128:
            m_double_data.resize(size*2);
            std::copy(first, last, reinterpret_cast<std::complex<double>*>(m_double_data.data()));
            break;

        default:
            throw std::logic_error("unsupported tensor data type");
        }
    }

    std::vector<float>& float_data() noexcept {
        return m_float_data;
    }

    const std::vector<float>& float_data() const noexcept {
        return m_float_data;
    }

    std::vector<double>& double_data() noexcept {
        return m_double_data;
    }

    const std::vector<double>& double_data() const noexcept {
        return m_double_data;
    }

    std::vector<int32_t>& int32_data() noexcept {
        return m_int32_data;
    }

    const std::vector<int32_t>& int32_data() const noexcept {
        return m_int32_data;
    }

    std::vector<int64_t>& int64_data() noexcept {
        return m_int64_data;
    }

    const std::vector<int64_t>& int64_data() const noexcept {
        return m_int64_data;
    }

    std::vector<uint64_t>& uint64_data() noexcept {
        return m_uint64_data;
    }

    const std::vector<uint64_t>& uint64_data() const noexcept {
        return m_uint64_data;
    }

    std::vector<std::string>& string_data() noexcept {
        return m_string_data;
    }

    const std::vector<std::string>& string_data() const noexcept {
        return m_string_data;
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

    void clear() {
        m_float_data.clear();
        m_double_data.clear();
        m_int32_data.clear();
        m_int64_data.clear();
        m_uint64_data.clear();
        m_string_data.clear();
        m_raw_data.clear();
        m_has_raw_data = false;
    }

    template <typename T>
    Tensor<T> decode();
};

template <typename T>
Tensor<T> TensorData::decode() {
    Shape shape(m_dims);

    switch (type()) {
    case DataType::FLOAT:
        return Tensor<float>::wrap(shape, m_float_data.data()).cast<T>();

    case DataType::DOUBLE:
        return Tensor<double>::wrap(shape, m_double_data.data()).cast<T>();

    case DataType::UINT8:
    case DataType::INT8:
    case DataType::UINT16:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::BOOL:
        return Tensor<int32_t>::wrap(shape, m_int32_data.data()).cast<T>();

    case DataType::INT64:
        return Tensor<int64_t>::wrap(shape, m_int64_data.data()).cast<T>();

    case DataType::UINT32:
    case DataType::UINT64:
        return Tensor<uint64_t>::wrap(shape, m_uint64_data.data()).cast<T>();

    default:
        throw std::logic_error("invalid tensor data type");
    }
}

template <>
inline Tensor<std::string> TensorData::decode() {
    if (type() != DataType::STRING)
        throw std::logic_error("invalid tensor data type");
    return {Shape(m_dims), m_string_data.begin(), m_string_data.end()};
}

template <>
inline Tensor<std::complex<float>> TensorData::decode() {
    if (type() != DataType::COMPLEX64)
        throw std::logic_error("invalid tensor data type");

    std::complex<float>* data = reinterpret_cast<std::complex<float>*>(m_float_data.data());
    size_t size = m_float_data.size() / 2;
    return {Shape(m_dims), data, data+size};
}

template <>
inline Tensor<std::complex<double>> TensorData::decode() {
    if (type() != DataType::COMPLEX128)
        throw std::logic_error("invalid tensor data type");

    std::complex<double>* data = reinterpret_cast<std::complex<double>*>(m_float_data.data());
    size_t size = m_float_data.size() / 2;
    return {Shape(m_dims), data, data+size};
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

namespace detail {
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
} // namespace detail

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
        detail::AttributeType<AttributeKind::FLOAT>,
        detail::AttributeType<AttributeKind::INT>,
        detail::AttributeType<AttributeKind::STRING>,
        detail::AttributeType<AttributeKind::TENSOR>,
        detail::AttributeType<AttributeKind::GRAPH>,
        detail::AttributeType<AttributeKind::FLOATS>,
        detail::AttributeType<AttributeKind::INTS>,
        detail::AttributeType<AttributeKind::STRINGS>,
        detail::AttributeType<AttributeKind::TENSORS>,
        detail::AttributeType<AttributeKind::GRAPHS>
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

    AttributeKind kindOf(Symbol name) const noexcept {
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

    #define AT(k) detail::AttributeType<AttributeKind::k>
    #define CREATE_ACCESSOR(kind, method) \
    Derived& set_##method(Symbol name, AT(kind) v) noexcept { \
        return set<AT(kind)>(name, std::move(v)); \
    } \
    const AT(kind)& get_##method(Symbol name) const { \
        return get<AT(kind)>(name); \
    } \
    const AT(kind)& get_##method(Symbol name, const AT(kind)& default_value) const { \
        return get_or_default<AT(kind)>(name, default_value); \
    }

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
    const T& get_or_default(Symbol name, const T& default_value) const {
        auto it = find(name, false);
        if (it == m_values.end()) {
            return default_value;
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
        using ValueType = detail::AttributeType<Kind>;

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
        std::transform(m_values.begin(), m_values.end(), names.begin(),
            [](auto const& a) { return a->name(); });
        return names;
    }

    AttributeKind kindOf(Symbol name) const noexcept {
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
    const AT(kind)& get_##method(Symbol name, const AT(kind)& d) const { \
        return get_or_default<kind##Attribute>(name, kind##Attribute::Kind, d); \
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
        auto nv = std::make_unique<AT>(name, std::move(v));
        if (it == m_values.end()) {
            m_values.push_back(std::move(nv));
        } else {
            *it = std::move(nv);
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
    const V& get_or_default(Symbol name, AttributeKind kind, const V& default_value) const {
        auto it = find(name, false);
        if (it == m_values.end()) {
            return default_value;
        } else if ((*it)->kind() != kind) {
            throw bad_variant_access();
        } else {
            return static_cast<AT*>(it->get())->value();
        }
    }
};

#endif // HAS_VARIANT

}} // namespace dlf::model
