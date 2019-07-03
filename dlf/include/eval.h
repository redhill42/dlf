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
    explicit Evaluator(model::Graph& graph);

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

    template <typename U = T>
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

    #define DEFINE_UNARY_OPERATOR(Name, fn) \
    void visit(model::Name* n) override { \
        struct Name##Op : Operator { \
            TensorT<> X, Y; \
            Name##Op(TensorT<>&& X, TensorT<>&& Y) \
                : X(std::move(X)), Y(std::move(Y)) {} \
            void evaluate() override \
                { dlf::transformTo(X, Y, ::dlf::xfn::fn<T>()); } \
        }; \
        result = std::make_unique<Name##Op>(alloc(n->input()), alloc(n->output())); \
    }
    
    DEFINE_UNARY_OPERATOR(Abs, abs)
    DEFINE_UNARY_OPERATOR(Neg, negate)
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

    struct ClipOp : Operator {
        xfn::clip<T> op; TensorT<> X, Y;
        ClipOp(const T& min, const T& max, TensorT<>&& X, TensorT<>&& Y)
            : op(min, max), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override {
            dlf::transformTo(X, Y, op);
        }
    };

    void visit(model::Clip* n) override {
        result = std::make_unique<ClipOp>(
            n->min(), n->max(), alloc(n->input()), alloc(n->output()));
    }

    struct ReluOp : Operator {
        TensorT<> X, Y;
        ReluOp(TensorT<>&& X, TensorT<>&& Y)
            : X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override {
            dlf::transformTo(X, Y, xfn::relu<T>());
        }
    };

    void visit(model::Relu* n) override {
        result = std::make_unique<ReluOp>(alloc(n->input()), alloc(n->output()));
    }

    struct PReluOp : Operator {
        TensorT<> X, slope, Y;
        PReluOp(TensorT<>&& X, TensorT<>&& slope, TensorT<>&& Y)
            : X(std::move(X)), slope(std::move(slope)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, slope, Y, xfn::prelu<T>()); }
    };

    void visit(model::PRelu* n) override {
        result = std::make_unique<PReluOp>(alloc(n->input()), alloc(n->slope()), alloc(n->output()));
    }

    struct LeakyReluOp : Operator {
        xfn::leaky_relu<T> op; TensorT<> X, Y;
        LeakyReluOp(float alpha, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::LeakyRelu* n) override {
        result = std::make_unique<LeakyReluOp>(n->alpha(), alloc(n->input()), alloc(n->output()));
    }

    struct ThresholdedReluOp : Operator {
        xfn::thresholded_relu<T> op; TensorT<> X, Y;
        ThresholdedReluOp(float alpha, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::ThresholdedRelu* n) override {
        result = std::make_unique<ThresholdedReluOp>(n->alpha(), alloc(n->input()), alloc(n->output()));
    }

    struct SeluOp : Operator {
        xfn::selu<T> op; TensorT<> X, Y;
        SeluOp(float alpha, float gamma, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha, gamma), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::Selu* n) override {
        result = std::make_unique<SeluOp>(
            n->alpha(), n->gamma(), alloc(n->input()), alloc(n->output()));
    }

    struct EluOp : Operator {
        xfn::elu<T> op; TensorT<> X, Y;
        EluOp(float alpha, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::Elu* n) override {
        result = std::make_unique<EluOp>(n->alpha(), alloc(n->input()), alloc(n->output()));
    }

    struct HardSigmoidOp : Operator {
        xfn::hard_sigmoid<T> op; TensorT<> X, Y;
        HardSigmoidOp(T alpha, T beta, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha, beta), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::HardSigmoid* n) override {
        result = std::make_unique<HardSigmoidOp>(
            n->alpha(), n->beta(), alloc(n->input()), alloc(n->output()));
    }

    struct SoftsignOp : Operator {
        TensorT<> X, Y;
        SoftsignOp(TensorT<>&& X, TensorT<>&& Y)
            : X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, xfn::softsign<T>()); }
    };

    void visit(model::Softsign* n) override {
        result = std::make_unique<SoftsignOp>(alloc(n->input()), alloc(n->output()));
    }

    struct SoftplusOp : Operator {
        TensorT<> X, Y;
        SoftplusOp(TensorT<>&& X, TensorT<>&& Y)
            : X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, xfn::softplus<T>()); }
    };

    void visit(model::Softplus* n) override {
        result = std::make_unique<SoftplusOp>(alloc(n->input()), alloc(n->output()));
    }

    #define DEFINE_BINARY_OPERATOR(Name, fn) \
    void visit(model::Name* n) override { \
        struct Name##Op : Operator { \
            TensorT<> A, B, C; \
            Name##Op(TensorT<>&& A, TensorT<>&& B, TensorT<>&& C) \
                : A(std::move(A)), B(std::move(B)), C(std::move(C)) {} \
            void evaluate() override { dlf::transformTo(A, B, C, ::dlf::xfn::fn<T>()); } \
        }; \
        result = std::make_unique<Name##Op>(alloc(n->A()), alloc(n->B()), alloc(n->C())); \
    }

    DEFINE_BINARY_OPERATOR(Add, plus)
    DEFINE_BINARY_OPERATOR(Sub, minus)
    DEFINE_BINARY_OPERATOR(Mul, multiplies)
    DEFINE_BINARY_OPERATOR(Div, divides)
    DEFINE_BINARY_OPERATOR(Pow, power)
    #undef DEFINE_BINARY_OPERATOR

    struct GemmOp : Operator {
        T alpha, beta;
        bool transA, transB;
        TensorT<> A, B, C, Y;

        GemmOp(T alpha, T beta, bool transA, bool transB,
               TensorT<>&& A, TensorT<>&& B, TensorT<>&& C, TensorT<>&& Y)
            : alpha(alpha), beta(beta), transA(transA), transB(transB),
              A(std::move(A)), B(std::move(B)), C(std::move(C)),
              Y(std::move(Y)) {}

        void evaluate() override {
            gemm(alpha, A, B, beta, C, Y, transA, transB);
        }
    };

    void visit(model::Gemm* n) override {
        result = std::make_unique<GemmOp>(
            T(n->alpha()), T(n->beta()), n->transA(), n->transB(),
            alloc(n->A()), alloc(n->B()), alloc(n->C()), alloc(n->Y()));
    }
};

template <typename Context, typename T>
Evaluator<Context, T>::Evaluator(model::Graph& graph) {
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
