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
    // Use vector here to ensure the order of the passes.
    std::vector<std::shared_ptr<Pass>> m_passes;

public:
    GlobalPassRegistryImpl();

    std::vector<std::string> getAvailablePasses() override {
        std::vector<std::string> names;
        for (const auto& pass : m_passes)
            names.push_back(pass->getPassName());
        return names;
    }

    void registerPass(std::shared_ptr<Pass> pass) override {
        m_passes.push_back(pass);
    }

    std::shared_ptr<Pass> find(const std::string& name) override {
        for (const auto& pass : m_passes)
            if (pass->getPassName() == name)
                return pass;
        return nullptr;
    }

    using GlobalPassRegistry::registerPass;
};

GlobalPassRegistryImpl::GlobalPassRegistryImpl() {
    registerPass<EliminateIdentity>();
    registerPass<EliminateDeadEnd>();
    registerPass<EliminateNopPad>();
    registerPass<EliminateNopTranspose>();
    registerPass<ReshapeInitializer>();
    registerPass<FuseBnIntoConv>();
    registerPass<FuseScaleIntoConv>();
    registerPass<FuseAddBiasIntoConv>();
}

GlobalPassRegistry& GlobalPassRegistry::Instance() {
    static GlobalPassRegistryImpl TheInstance;
    return TheInstance;
}

class OptimizerImpl final : public Optimizer {
    // Use vector here to ensure the order of the passes.
    std::vector<std::shared_ptr<Pass>> m_passes;

public:
    OptimizerImpl(const std::vector<std::string>& names);
    void optimize(Graph& graph) override;
};

OptimizerImpl::OptimizerImpl(const std::vector<std::string>& names) {
    for (const auto& name : names) {
        m_passes.push_back(GlobalPassRegistry::Instance().find(name));
    }
}

std::unique_ptr<Optimizer> Optimizer::newInstance(
    const std::vector<std::string>& names)
{
    return std::make_unique<OptimizerImpl>(names);
}

std::unique_ptr<Optimizer> Optimizer::newInstance() {
    return newInstance(GlobalPassRegistry::Instance().getAvailablePasses());
}

void OptimizerImpl::optimize(Graph& graph) {
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
