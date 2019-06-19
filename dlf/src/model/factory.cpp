#include "model.h"

namespace dlf { namespace model {

class DefaultNodeFactory : public NodeFactory {
public:
    Node* createNode(Graph* g, NodeKind kind) const override {
        switch (kind) {
            #define DEFINE_FACTORY(op) \
            case k##op: return new op(g);
            FORALL_OPERATORS(DEFINE_FACTORY)
            #undef DEFINE_FACTORY
        default:
            return new Node(g, kind);
        }
    }
};

const NodeFactory& NodeFactory::Instance() {
    static DefaultNodeFactory instance;
    return instance;
}

#define DEFINE_KIND(op) constexpr NodeKind op::Kind;
FORALL_OPERATORS(DEFINE_KIND)
#undef DEFINE_KIND

}} // namespace dlf::model
