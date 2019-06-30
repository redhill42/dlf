#pragma once

#include "model.h"

namespace dlf { namespace eval {

template <typename TensorT>
struct TensorAllocator {
    using element_type = typename TensorT::value_type;
    std::shared_ptr<TensorT> alloc(const Shape& shape);
    std::shared_ptr<TensorT> alloc(const Tensor<element_type>& tensor);
};

template <typename T>
struct TensorAllocator<Tensor<T>> {
    std::shared_ptr<Tensor<T>> alloc(const Shape& shape) {
        return std::make_shared<Tensor<T>>(shape);
    }

    std::shared_ptr<Tensor<T>> alloc(const Tensor<T>& tensor) {
        return std::make_shared<Tensor<T>>(tensor);
    }
};

template <typename T>
struct TensorAllocator<DevTensor<T>> {
    TensorAllocator(const gpgpu::Queue& queue = gpgpu::current::queue())
        : queue(queue) {}

    std::shared_ptr<DevTensor<T>> alloc(const Shape& shape) {
        return std::make_shared<DevTensor<T>>(shape, queue);
    }

    std::shared_ptr<DevTensor<T>> alloc(const Tensor<T>& tensor) {
        return std::make_shared<DevTensor<T>>(tensor, queue);
    }

private:
    const gpgpu::Queue& queue;
};

template <typename TensorT>
class Operator {
    std::vector<std::shared_ptr<TensorT>> m_inputs;
    std::vector<std::shared_ptr<TensorT>> m_outputs;

public:
    virtual ~Operator() = default;

    void addInput(std::shared_ptr<TensorT> input) {
        m_inputs.push_back(input);
    }

    void addOutput(std::shared_ptr<TensorT> output) {
        m_outputs.push_back(output);
    }

    const TensorT& input(size_t i) const {
        return *m_inputs.at(i);
    }

    TensorT& output(size_t i) {
        return *m_outputs.at(i);
    }

    virtual void evaluate() = 0;
};

template <typename TensorT>
class OperatorFactory : model::DefaultVisitor {
private:
    TensorAllocator<TensorT> m_alloc;
    std::unordered_map<model::Value*, std::shared_ptr<TensorT>> m_tensors;
    std::unique_ptr<Operator<TensorT>> result;

    using Element = typename TensorT::value_type;

public:
    OperatorFactory(TensorAllocator<TensorT> alloc) : m_alloc(alloc) {}

    std::shared_ptr<TensorT> alloc(model::Value* value) {
        auto it = m_tensors.find(value);
        if (it == m_tensors.end()) {
            std::shared_ptr<TensorT> tensor;
            if (value->has_initializer()) {
                tensor = m_alloc.alloc(value->initializer().decode<Element>());
            } else {
                tensor = m_alloc.alloc(Shape(value->dims()));
            }
            it = m_tensors.emplace(value, std::move(tensor)).first;
        }
        return it->second;
    }

    std::unique_ptr<Operator<TensorT>> createOperator(model::Node* node) {
        node->accept(*this);
        for (auto v : node->inputs())
            result->addInput(alloc(v));
        for (auto v : node->outputs())
            result->addOutput(alloc(v));
        return std::move(result);
    }

private:
    void visitNode(model::Node* n) override {
        throw std::runtime_error(cxx::concat("Unsupported operator ", n->kind().str()));
    }

    #define DEFINE_UNARY_OPERATOR(Name, op) \
    void visit(model::Name* n) override { \
        struct Name##Op : public Operator<TensorT> { \
            void evaluate() override { \
                dlf::op(this->input(0), this->output(0)); \
            } \
        }; \
        result = std::make_unique<Name##Op>(); \
    }
    
    DEFINE_UNARY_OPERATOR(Abs, abs)
    DEFINE_UNARY_OPERATOR(Neg, neg)
    DEFINE_UNARY_OPERATOR(Sign, sign)
    DEFINE_UNARY_OPERATOR(Reciprocal, reciprocal)
    DEFINE_UNARY_OPERATOR(Floor, floor)
    DEFINE_UNARY_OPERATOR(Ceil, ceil)
    DEFINE_UNARY_OPERATOR(Round, round)
    DEFINE_UNARY_OPERATOR(Sqrt, sqrt)
    DEFINE_UNARY_OPERATOR(Exp, exp)
    DEFINE_UNARY_OPERATOR(Log, log)
    DEFINE_UNARY_OPERATOR(Sin, sin)
    DEFINE_UNARY_OPERATOR(Cos, cos)
    DEFINE_UNARY_OPERATOR(Tan, tan)
    DEFINE_UNARY_OPERATOR(Asin, asin)
    DEFINE_UNARY_OPERATOR(Acos, acos)
    DEFINE_UNARY_OPERATOR(Atan, atan)
    DEFINE_UNARY_OPERATOR(Sinh, sinh)
    DEFINE_UNARY_OPERATOR(Cosh, cosh)
    DEFINE_UNARY_OPERATOR(Tanh, tanh)
    DEFINE_UNARY_OPERATOR(Asinh, asinh)
    DEFINE_UNARY_OPERATOR(Acosh, acosh)
    DEFINE_UNARY_OPERATOR(Atanh, atanh)
    DEFINE_UNARY_OPERATOR(Erf, erf)
    DEFINE_UNARY_OPERATOR(Sigmoid, sigmoid)
    #undef DEFINE_UNARY_OPERATOR
    
    #define DEFINE_BINARY_OPERATOR(Name, op) \
    void visit(model::Name* n) override { \
        struct Name##Op : public Operator<TensorT> { \
            void evaluate() override { \
                dlf::op(this->input(0), this->input(1), this->output(0)); \
            } \
        }; \
        result = std::make_unique<Name##Op>(); \
    }

    DEFINE_BINARY_OPERATOR(Add, addTo)
    DEFINE_BINARY_OPERATOR(Sub, subTo)
    DEFINE_BINARY_OPERATOR(Mul, mulTo)
    DEFINE_BINARY_OPERATOR(Div, divTo)
    #undef DEFINE_BINARY_OPERATOR

    struct Gemm : public Operator<TensorT> {
        Element alpha, beta;
        bool transA, transB;

        Gemm(model::Gemm* n) {
            alpha  = Element(n->get_f(model::kalpha, 1.f));
            beta   = Element(n->get_f(model::kbeta, 1.f));
            transA = !!n->get_i(model::ktransA, 0);
            transB = !!n->get_i(model::ktransB, 0);
        }

        void evaluate() override {
            gemm(alpha, this->input(0), this->input(1), beta, this->input(2),
                 this->output(0), transA, transB);
        }
    };

    void visit(model::Gemm* n) override {
        result = std::make_unique<Gemm>(n);
    }
};

template <typename TensorT>
class Evaluator {
    TensorAllocator<TensorT> m_alloc;
    std::vector<std::shared_ptr<TensorT>> m_inputs;
    std::vector<std::shared_ptr<TensorT>> m_outputs;
    std::vector<std::unique_ptr<Operator<TensorT>>> m_operators;

public:
    Evaluator() = default;
    Evaluator(TensorAllocator<TensorT> alloc) : m_alloc(alloc) {}

    void load(model::Graph& graph);

    TensorT& input(size_t i) { return *m_inputs.at(i); }
    TensorT& output(size_t i) { return *m_outputs.at(i); }

    void evaluate();
};

template <typename TensorT>
void Evaluator<TensorT>::load(model::Graph& graph) {
    m_inputs.clear();
    m_outputs.clear();
    m_operators.clear();

    OperatorFactory<TensorT> factory(m_alloc);
    model::ShapeInference::Instance().infer(graph);

    for (auto v : graph.inputs()) {
        if (!v->has_initializer())
            m_inputs.push_back(factory.alloc(v));
    }
    for (auto v : graph.outputs()) {
        m_outputs.push_back(factory.alloc(v));
    }
    for (auto n : graph.nodes()) {
        m_operators.push_back(factory.createOperator(n));
    }
}

template <typename TensorT>
void Evaluator<TensorT>::evaluate() {
    for (auto& op : m_operators) {
        op->evaluate();
    }
}

}} // namespace dlf::eval
