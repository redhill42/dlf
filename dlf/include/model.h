#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <cassert>

#if __cplusplus >= 201703L
#define HAS_VARIANT
#include <variant>
#endif

#include "model/intern.h"

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
DEFINE_ATTRIBUTE_TYPE(TENSOR, void*) // FIXME
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
        return find(name, true)->kind();
    }

    Derived* removeAttribute(Symbol name) noexcept {
        m_values.erase(find(name, true));
        return This();
    }

    #define AT(k) detail::AttributeType<AttributeKind::k>
    #define CREATE_ACCESSOR(kind, method) \
    Derived* set_##method(Symbol name, AT(kind) v) noexcept { \
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
    Derived* set(Symbol name, T&& value) noexcept {
        auto it = find(name, false);
        if (it == m_values.end()) {
            m_values.emplace_back(name, std::forward<T>(value));
        } else {
            it->template value<T>(std::forward<T>(value));
        }
        return This();
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
        return (*find(name, true))->kind();
    }

    Derived* removeAttribute(Symbol name) noexcept {
        m_values.erase(find(name, true));
        return This();
    }

    #define AT(kind) typename kind##Attribute::ValueType
    #define CREATE_ACCESSOR(kind, method) \
    Derived* set_##method(Symbol name, AT(kind) v) noexcept { \
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
    Derived* set(Symbol name, V&& v) {
        auto it = find(name, false);
        auto nv = std::make_unique<AT>(name, std::move(v));
        if (it == m_values.end()) {
            m_values.push_back(std::move(nv));
        } else {
            *it = std::move(nv);
        }
        return This();
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