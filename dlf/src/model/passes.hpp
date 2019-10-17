#pragma once

namespace dlf { namespace model {

class ExtractConstantToInitializer final : public PredicateBasedPass {
    std::string getPassName() const override {
        return "extract_constant_to_initializer";
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kConstant;
    }

    bool runTransform(Node* node, Graph& graph, NodeDestroyType& destroyCurrent) override {
        auto t = node->get_t(kvalue);
        t.set_name(node->output()->name());
        Value* new_init = graph.addInitializer(std::move(t));
        node->output()->replaceAllUsesWith(new_init);
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

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
        return (node->kind() == kReshape
             || node->kind() == kSqueeze
             || node->kind() == kUnsqueeze
             || node->kind() == kFlatten)
             && node->input()->has_initializer()
             && node->input()->uses().size() == 1
             && node->output()->has_dims();
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
        auto Bbn   = bn->B()->initializer().decode<T>();
        auto mean  = bn->mean()->initializer().decode<T>();
        auto var   = bn->var()->initializer().decode<T>();
        auto W     = conv->W()->initializer().decode<T>();

        assert(scale.rank() == 1);
        assert(Bbn.shape() == scale.shape());
        assert(mean.shape() == scale.shape());
        assert(var.shape() == scale.shape());
        assert(W.rank() > 2 && W.extent(0) == scale.extent(0));

        Tensor<T> Bc;
        if (conv->B() != nullptr) {
            Bc = conv->B()->initializer().decode<T>();
            assert(Bc.shape() == scale.shape());
        } else {
            Bc = Tensor<T>(scale.shape(), T{});
        }

        // do computation
        auto epsilon = static_cast<T>(bn->epsilon());
        scale /= sqrt(var + epsilon);
        Bc = (Bc - mean) * scale + Bbn;
        transformChannel(W, scale, W, 0, xfn::multiplies<>());

        // replace inputs
        conv->W()->initializer().set_data(W.begin(), W.end());
        if (conv->B() != nullptr) {
            conv->B()->initializer().set_data(Bc.begin(), Bc.end());
        } else {
            auto b_data = TensorData(conv->Y()->name() + ":bias", Bc); // FIXME
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

class FuseScaleIntoConv final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "fuse_scale_into_conv";
    }

    bool isComplete() const override {
        return false;
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kMul && node->input(0)->node()->kind() == kConv;
    }

    bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroyCurrent) override {
        auto conv = static_cast<Conv*>(n->input(0)->node());
        auto val = n->input(1);

        if (conv->Y()->uses().size() != 1 || !conv->Y()->has_dims())
            return false;
        if (!val->has_initializer())
            return false;
        if (!conv->W()->has_initializer())
            return false;
        if (conv->B() != nullptr && !conv->B()->has_initializer())
            return false;

        // try to get output channel M
        auto M = conv->W()->dim(0);

        // for output shape (N,M,H,W...), the scale must have shape (1,M,1,1...) or (M,1,1,...)
        std::vector<int> axes(val->dims().rank());
        std::iota(axes.begin(), axes.end(), 0);
        if (val->dims().rank() == conv->Y()->dims().rank()) {
            if (val->dim(0) != 1 || val->dim(1) != M)
                return false;
            for (int i = 2; i < val->dims().rank(); i++)
                if (val->dim(i) != 1)
                    return false;
            axes.erase(axes.begin() + 1);
        } else if (val->dims().rank() == conv->Y()->dims().rank() - 1) {
            if (val->dim(0) != M)
                return false;
            for (int i = 1; i < val->dims().rank(); i++)
                if (val->dim(i) != 1)
                    return false;
            axes.erase(axes.begin());
        } else {
            return false;
        }

        auto scale = val->initializer().decode<float>();
        scale.squeeze(axes);

        auto W = conv->W()->initializer().decode<float>();
        transformChannel(W, scale, W, 0, xfn::multiplies<>());
        conv->W()->initializer().set_data(W.begin(), W.end());

        if (conv->B() != nullptr) {
            auto B = conv->B()->initializer().decode<float>() * scale;
            conv->B()->initializer().set_data(B.begin(), B.end());
        }

        if (val->uses().size() == 1) {
            n->eraseInput(1);
            graph.eraseInput(val);
        }

        n->output()->replaceAllUsesWith(conv->Y());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

class FuseAddBiasIntoConv final : public PredicateBasedPass {
public:
    std::string getPassName() const override {
        return "fuse_add_bias_into_conv";
    }

    bool isComplete() const override {
        return false;
    }

    bool patternMatchPredicate(Node* node) override {
        return node->kind() == kAdd && node->input(0)->node()->kind() == kConv;
    }

    bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroyCurrent) override {
        auto conv = static_cast<Conv*>(n->input(0)->node());
        auto val = n->input(1);

        if (conv->Y()->uses().size() != 1 || !conv->Y()->has_dims())
            return false;
        if (!val->has_initializer())
            return false;
        if (!conv->W()->has_dims())
            return false;
        if (conv->B() != nullptr && !conv->B()->has_initializer())
            return false;

        // try to get output channel M
        auto M = conv->W()->dim(0);

        // for output shape (N,M,H,W...), the bias must have shape (1,M,1,1...) or (M,1,1,...)
        std::vector<int> axes(val->dims().rank());
        std::iota(axes.begin(), axes.end(), 0);
        if (val->dims().rank() == conv->Y()->dims().rank()) {
            if (val->dim(0) != 1 || val->dim(1) != M)
                return false;
            for (int i = 2; i < val->dims().rank(); i++)
                if (val->dim(i) != 1)
                    return false;
            axes.erase(axes.begin() + 1);
        } else if (val->dims().rank() == conv->Y()->dims().rank() - 1) {
            if (val->dim(0) != M)
                return false;
            for (int i = 1; i < val->dims().rank(); i++)
                if (val->dim(i) != 1)
                    return false;
            axes.erase(axes.begin());
        } else {
            return false;
        }

        auto bias = val->initializer().decode<float>();
        bias.squeeze(axes);

        if (conv->B() == nullptr) {
            conv->addInput(graph.addInitializer(TensorData(conv->Y()->name() + ":bias", bias)));
        } else {
            auto B = conv->B()->initializer().decode<float>() + bias;
            conv->B()->initializer().set_data(B.begin(), B.end());
        }

        if (val->uses().size() == 1) {
            n->eraseInput(1);
            graph.eraseInput(val);
        }

        n->output()->replaceAllUsesWith(conv->Y());
        destroyCurrent = NodeDestroyType::DestroyOne;
        return true;
    }
};

}} // namespace dlf::model
