#include <string>
#include <unordered_map>
#include <mutex>
#include <cassert>

#include "model/intern.h"

namespace dlf { namespace model {

class InternedStrings {
    std::unordered_map<std::string, uint32_t> m_string_to_sym;
    std::unordered_map<uint32_t, std::string> m_sym_to_string;
    uint32_t m_next_sym;
    std::mutex m_mutex;

public:
    InternedStrings() : m_next_sym(kLastSymbol) {
        #define REGISTER_SYMBOL(s) \
            m_string_to_sym[#s] = k##s; \
            m_sym_to_string[k##s] = #s;
        FORALL_BUILTIN_SYMBOLS(REGISTER_SYMBOL)
        #undef REGISTER_SYMBOL
    }

    uint32_t symbol(const std::string& s) {
        std::lock_guard<std::mutex> guard(m_mutex);
        auto it = m_string_to_sym.find(s);
        if (it != m_string_to_sym.end())
            return it->second;
        uint32_t k = m_next_sym++;
        m_string_to_sym[s] = k;
        m_sym_to_string[k] = s;
        return k;
    }

    const char* string(Symbol sym) {
        // Builtin Symbols are also in the maps, but we can bypass the
        // need to acquire a lock to read the map for Builtins because
        // we already know their string value
        switch (sym) {
            #define DEFINE_CASE(s) \
            case k##s: return #s;
            FORALL_BUILTIN_SYMBOLS(DEFINE_CASE)
            #undef DEFINE_CASE
        default:
            return custom_string(sym);
        }
    }

private:
    const char* custom_string(Symbol sym) {
        std::lock_guard<std::mutex> guard(m_mutex);
        auto it = m_sym_to_string.find(sym);
        assert(it != m_sym_to_string.end());
        return it->second.c_str();
    }
};

static InternedStrings& global_strings() {
    static InternedStrings s;
    return s;
}

Symbol::Symbol(const std::string& s)
    : m_value(global_strings().symbol(s))
{}

const char* Symbol::str() const noexcept {
    return global_strings().string(*this);
}

}} // namespace dlf::model
