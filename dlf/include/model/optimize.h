#pragma once

namespace dlf { namespace model {

enum class NodeDestroyType {
    // Does not destroy node.
    DestroyZero,

    // Destroy one node.
    DestroyOne,

    // Destroy two nodes.
    DestroyTwo
};

class Pass {
public:
    virtual ~Pass() = default;

    virtual std::string getPassName() const = 0;
    virtual bool isComplete() const { return true; }
    virtual bool runPass(Graph& graph) = 0;

protected:
    // Iterate through the elements in the graph and counts the number
    // of times the transform is successfully run.
    static int descendOnGraphAttributesAndCount(
        Node* n, std::function<int(Graph&)> fn);
};

/**
 * A pass that is based on pattern matching. The majority of passes will
 * implement this pss. In order for the pass to work the patternMatchPredicate
 * function must be implemented which matches a subgraph to the respective
 * optimization pass. Lastly the runTransform method must also be implemented
 * which simply implements the pass on any node which passes
 * patternMatchPredicate.
 */
class PredicateBasedPass : public Pass {
public:
    virtual bool patternMatchPredicate(Node* node) = 0;

    /**
     * Run transform is given the current node in the iterator, a reference
     * to the current graph as well as a reference describing how to treat
     * the current node in the iterator post transform. Run transform is then
     * responsible for running the actual transform as well as describing how
     * to treat the iterator node. By default the current node will not call
     * destroy. Do not internally delete node instead set the correct
     * destroyCurrent type.
     */
    virtual bool runTransform(
        Node* node, Graph& graph, NodeDestroyType& destroyCurrent) = 0;

    bool runPass(Graph& graph) override;

private:
    int runPassInternal(Graph& graph);
};

/**
 * Registry containing all passes.
 */
class GlobalPassRegistry {
public:
    virtual ~GlobalPassRegistry() = default;

    static GlobalPassRegistry& Instance();

    virtual std::vector<std::string> getAvailablePasses() = 0;
    virtual void registerPass(std::shared_ptr<Pass> pass) = 0;
    virtual std::shared_ptr<Pass> find(const std::string& name) = 0;

    template <typename T>
    void registerPass() {
        static_assert(std::is_base_of<Pass, T>::value, "T must inherit from Pass");
        registerPass(std::make_shared<T>());
    }
};

class Optimizer {
public:
    virtual ~Optimizer() = default;

    static std::unique_ptr<Optimizer> newInstance(
        const std::vector<std::string>& names);
    static std::unique_ptr<Optimizer> newInstance();

    virtual void optimize(Graph& graph) = 0;
};

}} // namespace dlf::model
