#include "model/operators.h"

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

}} // namespace dlf::model
