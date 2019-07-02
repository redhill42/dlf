#pragma once

#include "model.h"

namespace dlf { namespace eval {

struct CPU {
    template <typename T>
    using TensorT = Tensor<T>;
};

struct GPU {
    template <typename T>
    using TensorT = DevTensor<T>;
};

template <class Context> struct Datum {};

template <> struct Datum<CPU> {
    const model::DataType dtype;
    const Shape shape;

    Datum(model::DataType dtype, Shape shape)
        : dtype(dtype), shape(std::move(shape))
    {}

    template <typename T>
    Tensor<T> get() {
        assert(dtype == model::DataTypeTrait<T>);
        if (data.empty())
            data.resize(shape.size() * sizeof(T));
        return Tensor<T>::wrap(shape, reinterpret_cast<T*>(&data[0]));
    }

    template <typename T>
    Tensor<T> read() {
        return get<T>();
    }

    template <typename T>
    void set(const Tensor<T>& val) {
        auto dst = get<T>();
        dlf::copy(val, dst);
    }

private:
    std::vector<char> data;
};

template <> struct Datum<GPU> {
    const model::DataType dtype;
    const Shape shape;

    Datum(model::DataType dtype, Shape shape)
        : dtype(dtype), shape(std::move(shape))
    {}

    template <typename T>
    DevTensor<T> get() {
        assert(dtype == model::DataTypeTrait<T>);
        if (handle == nullptr)
            handle = gpgpu::current::context().createBuffer<T>(shape.size()).handle();
        return DevTensor<T>(shape, gpgpu::Buffer<T>(handle, shape.size()));
    }

    template <typename T>
    Tensor<T> read() {
        return get<T>().read();
    }

    template <typename T>
    void set(const Tensor<T>& val) {
        get<T>().write(val);
    }

private:
    std::shared_ptr<gpgpu::raw::Buffer> handle;
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual void evaluate() = 0;
};

template <typename Context, typename T>
class Evaluator {
public:
    void load(model::Graph& graph);

    void evaluate() {
        std::for_each(m_operators.begin(), m_operators.end(), [](auto& op){ op->evaluate(); });
    }

    void set(size_t i, Tensor<T> data) {
        m_inputs.at(i)->template set<T>(data);
    }

    Tensor<T> get(size_t i) {
        return m_outputs.at(i)->template read<T>();
    }

private:
    std::vector<std::shared_ptr<Datum<Context>>> m_dataset;
    std::vector<std::shared_ptr<Datum<Context>>> m_inputs;
    std::vector<std::shared_ptr<Datum<Context>>> m_outputs;
    std::vector<std::unique_ptr<Operator>> m_operators;
};

template <typename Context, typename T>
class OperatorFactory : model::DefaultVisitor {
private:
    using DatumPtr = std::shared_ptr<Datum<Context>>;

    std::vector<DatumPtr>& m_dataset;
    std::unordered_map<const model::Value*, DatumPtr> m_datamap;
    std::unique_ptr<Operator> result;

    template <typename U>
    using TensorT = typename Context::template TensorT<U>;

public:
    OperatorFactory(std::vector<DatumPtr>& dataset)
        : m_dataset(dataset) {}

    template <typename U = T>
    DatumPtr allocDatum(const model::Value* value) {
        auto it = m_datamap.find(value);
        if (it != m_datamap.end()) {
            return it->second;
        }

        auto datum = std::make_shared<Datum<Context>>(
            model::DataTypeTrait<U>, Shape(value->dims()));
        m_dataset.push_back(datum);
        m_datamap.emplace(value, datum);
        if (value->has_initializer())
            datum->template set<U>(value->initializer().decode<U>());
        return datum;
    }

    template <typename U = T>
    TensorT<U> alloc(const model::Value* value) {
        return allocDatum<U>(value)->template get<U>();
    }

    std::unique_ptr<Operator> createOperator(model::Node* node) {
        node->accept(*this);
        return std::move(result);
    }

private:
    void visitNode(model::Node* n) override {
        throw std::runtime_error(cxx::concat("Unsupported operator ", n->kind().str()));
    }

    #define DEFINE_UNARY_OPERATOR(Name, op) \
    void visit(model::Name* n) override { \
        struct Name##Op : Operator { \
            TensorT<T> X, Y; \
            Name##Op(TensorT<T>&& X, TensorT<T>&& Y) \
                : X(std::move(X)), Y(std::move(Y)) {} \
            void evaluate() override { \
                dlf::op(X, Y); \
            } \
        }; \
        result = std::make_unique<Name##Op>(alloc(n->input()), alloc(n->output())); \
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

    void visit(model::Relu* n) override {
        struct ReluOp : Operator {
            TensorT<T> X, Y;
            ReluOp(TensorT<T>&& X, TensorT<T>&& Y)
                : X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::relu(X, Y);
            }
        };

        result = std::make_unique<ReluOp>(
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::PRelu* n) override {
        struct PReluOp : Operator {
            TensorT<T> X, slope, Y;
            PReluOp(TensorT<T>&& X, TensorT<T>&& slope, TensorT<T>&& Y)
                : X(std::move(X)), slope(std::move(slope)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::prelu(X, slope, Y);
            }
        };

        result = std::make_unique<PReluOp>(
            alloc(n->input()), alloc(n->slope()), alloc(n->output()));
    }

    void visit(model::LeakyRelu* n) override {
        struct LeakyReluOp : Operator {
            T alpha;
            TensorT<T> X, Y;
            LeakyReluOp(T alpha, TensorT<T>&& X, TensorT<T>&& Y)
                : alpha(alpha), X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::leaky_relu(alpha, X, Y);
            }
        };

        result = std::make_unique<LeakyReluOp>(
            T(n->get_f(model::kalpha, 0.01f)),
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::ThresholdedRelu* n) override {
        struct ThresholdedReluOp : Operator {
            T alpha;
            TensorT<T> X, Y;
            ThresholdedReluOp(T alpha, TensorT<T>&& X, TensorT<T>&& Y)
                : alpha(alpha), X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::thresholded_relu(alpha, X, Y);
            }
        };

        result = std::make_unique<ThresholdedReluOp>(
            T(n->get_f(model::kalpha, 1.f)),
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Selu* n) override {
        struct SeluOp : Operator {
            T alpha, gamma;
            TensorT<T> X, Y;
            SeluOp(T alpha, T gamma, TensorT<T>&& X, TensorT<T>&& Y)
                : alpha(alpha), gamma(gamma), X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::selu(alpha, gamma, X, Y);
            }
        };

        result = std::make_unique<SeluOp>(
            T(n->get_f(model::kalpha, 1.67326319217681884765625f)),
            T(n->get_f(model::kgamma, 1.05070102214813232421875f)),
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Elu* n) override {
        struct EluOp : Operator {
            T alpha;
            TensorT<T> X, Y;
            EluOp(T alpha, TensorT<T>&& X, TensorT<T>&& Y)
                : alpha(alpha), X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::elu(alpha, X, Y);
            }
        };

        result = std::make_unique<EluOp>(
            T(n->get_f(model::kalpha, 1.f)),
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::HardSigmoid* n) override {
        struct HardSigmoidOp : Operator {
            T alpha, beta;
            TensorT<T> X, Y;
            HardSigmoidOp(T alpha, T beta, TensorT<T>&& X, TensorT<T>&& Y)
                : alpha(alpha), beta(beta), X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::hard_sigmoid(alpha, beta, X, Y);
            }
        };

        result = std::make_unique<HardSigmoidOp>(
            T(n->get_f(model::kalpha, 0.2f)),
            T(n->get_f(model::kbeta, 0.5f)),
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Softsign* n) override {
        struct SoftsignOp : Operator {
            TensorT<T> X, Y;
            SoftsignOp(TensorT<T>&& X, TensorT<T>&& Y)
                : X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::softsign(X, Y);
            }
        };

        result = std::make_unique<SoftsignOp>(
            alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Softplus* n) override {
        struct SoftplusOp : Operator {
            TensorT<T> X, Y;
            SoftplusOp(TensorT<T>&& X, TensorT<T>&& Y)
                : X(std::move(X)), Y(std::move(Y)) {}
            void evaluate() override {
                dlf::softplus(X, Y);
            }
        };

        result = std::make_unique<SoftplusOp>(
            alloc(n->input()), alloc(n->output()));
    }

    #define DEFINE_BINARY_OPERATOR(Name, op) \
    void visit(model::Name* n) override { \
        struct Name##Op : Operator { \
            TensorT<T> A, B, C; \
            Name##Op(TensorT<T>&& A, TensorT<T>&& B, TensorT<T>&& C) \
                : A(std::move(A)), B(std::move(B)), C(std::move(C)) {} \
            void evaluate() override { \
                dlf::op(A, B, C); \
            } \
        }; \
        result = std::make_unique<Name##Op>(alloc(n->A()), alloc(n->B()), alloc(n->C())); \
    }

    DEFINE_BINARY_OPERATOR(Add, addTo)
    DEFINE_BINARY_OPERATOR(Sub, subTo)
    DEFINE_BINARY_OPERATOR(Mul, mulTo)
    DEFINE_BINARY_OPERATOR(Div, divTo)
    #undef DEFINE_BINARY_OPERATOR

    void visit(model::Gemm* n) override {
        struct GemmOp : Operator {
            T alpha, beta;
            bool transA, transB;
            TensorT<T> A, B, C, Y;

            GemmOp(T alpha, T beta, bool transA, bool transB,
                   TensorT<T>&& A, TensorT<T>&& B, TensorT<T>&& C, TensorT<T>&& Y)
                : alpha(alpha), beta(beta), transA(transA), transB(transB),
                  A(std::move(A)), B(std::move(B)), C(std::move(C)),
                  Y(std::move(Y)) {}

            void evaluate() override {
                gemm(alpha, A, B, beta, C, Y, transA, transB);
            }
        };

        result = std::make_unique<GemmOp>(
            T(n->get_f(model::kalpha, 1.f)),
            T(n->get_f(model::kbeta, 1.f)),
            !!n->get_i(model::ktransA, 0),
            !!n->get_i(model::ktransB, 0),
            alloc(n->A()), alloc(n->B()), alloc(n->C()), alloc(n->Y()));
    }
};

template <typename Context, typename T>
void Evaluator<Context, T>::load(model::Graph& graph) {
    m_dataset.clear();
    m_inputs.clear();
    m_outputs.clear();
    m_operators.clear();

    OperatorFactory<Context, T> factory(m_dataset);
    model::ShapeInference::Instance().infer(graph);

    for (auto v : graph.inputs()) {
        if (!v->has_initializer())
            m_inputs.push_back(factory.allocDatum(v));
    }
    for (auto v : graph.outputs()) {
        m_outputs.push_back(factory.allocDatum(v));
    }
    for (auto n : graph.nodes()) {
        m_operators.push_back(factory.createOperator(n));
    }
}

}} // namespace dlf::eval
