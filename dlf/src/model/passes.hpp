#pragma once

namespace dlf { namespace model {

class EliminateDeadEnd final : public Pass {
public:
    std::string getPassName() const override {
        return "eliminate_deadend";
    }

    bool runPass(Graph& graph) override {
        int nodes_removed = 0;
        auto nodes = graph.nodes().reverse();
        for (auto it = nodes.begin(); it != nodes.end(); it++) {
            auto node = *it;
            if (!node->hasUses()) {
                nodes_removed++;
                it.destroyCurrent();
            }
        }
        return nodes_removed > 0;
    }
};

class EliminateIdentity final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "eliminate_identity";
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kIdentity || node->kind() == kDropout;
    }

    bool runTransform(Node* node, Graph&, NodeDestroyType& destroyCurrent) override {
        node->output()->replaceAllUsesWith(node->input());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

class EliminateNopPad final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "eliminate_nop_pad";
    }

    static bool is_nop_pad(const std::vector<int64_t>& pads) {
        return std::all_of(pads.begin(), pads.end(), [](auto x){ return x == 0; });
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kPad && node->hasAttribute(kpads) &&
            is_nop_pad(node->get_is(kpads));
    }

    bool runTransform(Node* node, Graph&, NodeDestroyType& destroyCurrent) override {
        node->output()->replaceAllUsesWith(node->input());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

class EliminateNopTranspose final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "eliminate_nop_transpose";
    }

    static bool is_nop_transpose(const std::vector<int64_t>& perm) {
        for (size_t i = 0; i < perm.size(); i++)
            if (perm[i] != static_cast<int64_t>(i))
                return false;
        return true;
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kTranspose && node->hasAttribute(kperm) &&
            is_nop_transpose(node->get_is(kperm));
    }

    bool runTransform(Node* node, Graph&, NodeDestroyType& destroyCurrent) override {
        node->output()->replaceAllUsesWith(node->input());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

class ReshapeInitializer final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "reshape_initializer";
    }

    bool patternMatchPredicate(Node* node) override {
        return node->input()->has_initializer()
            && node->input()->uses().size() == 1
            && (node->kind() == kReshape
             || node->kind() == kSqueeze
             || node->kind() == kUnsqueeze
             || node->kind() == kFlatten);
    }

    bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroyCurrent) override {
        node->input()->set_dims(node->output()->dims());
        node->input()->initializer().set_dims(node->output()->dims());
        node->output()->replaceAllUsesWith(node->input());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

/**
 * Fuse batch normalization into convolution.
 *
 * Before:
 *   conv = Conv()
 *     bn = BatchNormalization()
 *
 * After:
 *   bn is deleted
 *   new inputs/initializers to conv are added to graph
 *   any no longer used inputs/initializers are erased from graph
 *
 * This pass can handle the case satisfy all following conditions:
 *   condition 1: Run in testing mode
 *   condition 2: Inputs 1 - 4 of bn are all initializer_size
 *   condition 3: Output of initial conv has no other uses
 *   condition 4: Currently work for only DOUBLE, FLOAT32 tensor types
 *
 * Formula for transformation
 *   $$ X_{bn} = \frac{s(X - m)}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
 *   $$ X_{conv} = X * W + b_{conv} $$
 * thus, substituting $X$ with $X_{conv}$ in the BN equation we get:
 *   $$ X_{bn} = X * \frac{s W}{\sqrt{\sigma + \epsilon}}
 *             + \frac{s(b_{conv} - m)}{\sqrt{\sigma + \epsilon}}
 *             + b_{bn}$$
 * or
 *   $$ W' = W\frac{s}{\sqrt{\sigma + \epsilon}}$$
 *   $$ b' = (b_{conv} - m)\frac{s}{\sqrt{\sigma + \epsilon}} + b_{bn}$$
 */
class FuseBnIntoConv final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "fuse_bn_into_conv";
    }

    template <typename T>
    bool modify_conv(Conv* conv, BatchNormalization* bn, Graph& graph) {
        if (bn->X()->uses().size() > 1 || bn->outputs().size() > 1)
            return false;

        if (   !bn->scale()->has_initializer()
            || !bn->mean()->has_initializer()
            || !bn->var()->has_initializer()
            || !bn->B()->has_initializer()
            || !conv->W()->has_initializer()
            || (conv->B() != nullptr && !conv->B()->has_initializer()))
            return false;

        auto scale = bn->scale()->initializer().decode<T>();
        auto bbn   = bn->B()->initializer().decode<T>();
        auto mean  = bn->mean()->initializer().decode<T>();
        auto var   = bn->var()->initializer().decode<T>();
        auto W     = conv->W()->initializer().decode<T>();

        assert(scale.rank() == 1);
        assert(bbn.shape() == scale.shape());
        assert(mean.shape() == scale.shape());
        assert(var.shape() == scale.shape());
        assert(W.rank() > 2 && W.extent(0) == scale.extent(0));

        Tensor<T> bc;
        if (conv->B() != nullptr) {
            bc = conv->B()->initializer().decode<T>();
            assert(bc.shape() == scale.shape());
        } else {
            bc = Tensor<T>(scale.shape());
            std::fill(bc.begin(), bc.end(), T{});
        }

        // do computation
        T epsilon = static_cast<T>(bn->epsilon());
        scale.apply(var, [=](auto s, auto v) { return s / std::sqrt(v + epsilon); });
        bc = (bc - mean) * scale + bbn;
        transformChannel(W, scale, W, 0, xfn::multiplies<>());

        // replace inputs
        conv->W()->initializer().set_data(W.begin(), W.end());
        if (conv->B() != nullptr) {
            conv->B()->initializer().set_data(bc.begin(), bc.end());
        } else {
            auto b_data = TensorData(conv->Y()->name() + ":bias", bc); // FIXME
            conv->addInput(graph.addInitializer(std::move(b_data)));
        }

        return true;
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kBatchNormalization
            && node->input(0)->node()->kind() == kConv;
    }

    bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroyCurrent) override {
        auto bn = static_cast<BatchNormalization*>(n);
        auto conv = static_cast<Conv*>(bn->input(0)->node());

        switch (conv->X()->type()) {
        case DataType::FLOAT:
            if (!modify_conv<float>(conv, bn, graph))
                return false;
            break;

        case DataType::DOUBLE:
            if (!modify_conv<double>(conv, bn, graph))
                return false;
            break;

        default:
            return false;
        }

        for (int i = 4; i >= 1; --i) {
            if (bn->input(i)->uses().size() == 1) {
                auto input = bn->input(i);
                bn->eraseInput(i);
                graph.eraseInput(input);
            }
        }

        bn->Y()->replaceAllUsesWith(bn->X());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

}} // namespace dlf::model
