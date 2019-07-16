#include "model.h"
#include "passes.hpp"

namespace dlf { namespace model {

int Pass::descendOnGraphAttributesAndCount(
    Node* n, std::function<int(Graph&)> fn)
{
    int num_changes = 0;
    for (auto name : n->attributeNames()) {
        auto kind = n->attributeKind(name);
        if (kind == AttributeKind::GRAPH) {
            num_changes += fn(*n->get_g(name));
        }
        if (kind == AttributeKind::GRAPHS) {
            for (auto& g : n->get_gs(name)) {
                num_changes += fn(*g);
            }
        }
    }
    return num_changes;
}

int PredicateBasedPass::runPassInternal(Graph& graph) {
    int num_changes = 0;
    for (auto it = graph.begin(); it != graph.end(); ++it) {
        auto* n = *it;
        num_changes += descendOnGraphAttributesAndCount(
            n, [this](Graph& g) { return runPassInternal(g); });
        if (patternMatchPredicate(n)) {
            NodeDestroyType destroyType = NodeDestroyType::DestroyZero;
            num_changes += runTransform(n, graph, destroyType);
            if (destroyType == NodeDestroyType::DestroyOne) {
                it.destroyCurrent();
            }
            if (destroyType == NodeDestroyType::DestroyTwo) {
                it.destroyCurrent();
                it.destroyCurrent();
            }
        }
    }
    return num_changes;
}

bool PredicateBasedPass::runPass(Graph& graph) {
    return runPassInternal(graph) > 0;
}

//==-------------------------------------------------------------------------

class GlobalPassRegistryImpl : public GlobalPassRegistry {
    std::unordered_map<std::string, std::shared_ptr<Pass>> m_passes;

public:
    GlobalPassRegistryImpl();

    std::vector<std::string> getAvailablePasses() override {
        std::vector<std::string> names;
        for (const auto& pass : m_passes)
            names.push_back(pass.first);
        return names;
    }

    void registerPass(std::shared_ptr<Pass> pass) override {
        m_passes[pass->getPassName()] = pass;
    }

    std::shared_ptr<Pass> find(const std::string& name) override {
        auto it = m_passes.find(name);
        assert(it != m_passes.end());
        return it->second;
    }

    using GlobalPassRegistry::registerPass;
};

GlobalPassRegistryImpl::GlobalPassRegistryImpl() {
    registerPass<EliminateIdentity>();
}

GlobalPassRegistry& GlobalPassRegistry::Instance() {
    static GlobalPassRegistryImpl TheInstance;
    return TheInstance;
}

Optimizer::Optimizer(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        m_passes.push_back(GlobalPassRegistry::Instance().find(name));
    }
}

void Optimizer::optimize(Graph& graph) {
    bool done;
    do {
        done = true;
        for (auto pass : m_passes) {
            while (pass->runPass(graph)) {
                if (pass->isComplete())
                    break;
                done = false;
            }
        }
    } while (!done);
}

}} // namespace dlf::model
