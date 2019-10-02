#include "model.h"

namespace dlf { namespace model {

class DefaultNodeFactory final : public NodeFactory {
    std::unordered_set<const Node*> all_nodes;
    std::unordered_set<const Value*> all_values;

public:
    Node* createNode(Graph* g, NodeKind kind) override {
        Node* node;
        switch (kind) {
            #define DEFINE_FACTORY(op) \
            case k##op: node = new op(g); break;
            FORALL_OPERATORS(DEFINE_FACTORY)
            #undef DEFINE_FACTORY
        default:
            node = new Node(g, kind);
            break;
        }

        all_nodes.emplace(node);
        return node;
    }

    Value* createValue(Node* node, size_t offset, std::string&& name) override {
        Value* value = new Value(node, offset, std::move(name));
        all_values.emplace(value);
        return value;
    }

    void freeNode(Node* node) override {
        auto it = all_nodes.find(node);
        assert(it != all_nodes.end());
        all_nodes.erase(it);
        delete node;
    }

    void freeValue(Value* value) override {
        auto it = all_values.find(value);
        assert(it != all_values.end());
        all_values.erase(it);
        delete value;
    }

    ~DefaultNodeFactory() {
        for (auto n : all_nodes)
            delete n;
        for (auto v : all_values)
            delete v;
    }
};

std::unique_ptr<NodeFactory> NodeFactory::newInstance() {
    return std::make_unique<DefaultNodeFactory>();
}

#define DEFINE_KIND(op) constexpr NodeKind op::Kind;
FORALL_OPERATORS(DEFINE_KIND)
#undef DEFINE_KIND

}} // namespace dlf::model
