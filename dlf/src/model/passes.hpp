#pragma once

namespace dlf { namespace model {

class EliminateIdentity final : public PredicateBasedPass {
    std::string getPassName() const override {
        return "eliminate_identity";
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kIdentity || node->kind() == kDropout;
    }

    bool runTransform(Node* node, Graph&, NodeDestroyType& destroyType) override {
        node->output()->replaceAllUsesWith(node->input());
        destroyType = NodeDestroyType::DestroyOne;
        return true;
    }
};

}} // namespace dlf::model
