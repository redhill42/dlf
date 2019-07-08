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
        if (val.shape() != dst.shape())
            throw shape_error("incompatible shape");
        flat_copy(val, dst);
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
        auto dst = get<T>();
        if (val.shape() != dst.shape())
            throw shape_error("incompatible shape");
        dst.write(val);
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

    template <typename U = T>
    std::list<TensorT<U>> alloc_all(cxx::array_ref<model::Value*> values) {
        std::list<TensorT<U>> inputs;
        for (auto v : values)
            inputs.push_back(alloc<U>(v));
        return inputs;
    }

    std::unique_ptr<Operator> createOperator(model::Node* node) {
        node->accept(*this);
        return std::move(result);
    }

private:
    void visitNode(model::Node* n) override {
        throw std::runtime_error(cxx::string_concat("Unsupported operator ", n->kind().str()));
    }

    template <typename Fn>
    struct UnaryOp : Operator {
        TensorT<> X, Y;
        UnaryOp(TensorT<>&& X, TensorT<>&& Y)
            : X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { transformTo(X, Y, Fn{}); }
    };

    #define DEFINE_UNARY_OPERATOR(Name, fn) \
    void visit(model::Name* n) override { \
        result = std::make_unique<UnaryOp<xfn::fn<T>>>( \
            alloc(n->input()), alloc(n->output())); \
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
        result = std::make_unique<PReluOp>(
            alloc(n->input()), alloc(n->slope()), alloc(n->output()));
    }

    struct LeakyReluOp : Operator {
        xfn::leaky_relu<T> op; TensorT<> X, Y;
        LeakyReluOp(float alpha, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::LeakyRelu* n) override {
        result = std::make_unique<LeakyReluOp>(
            n->alpha(), alloc(n->input()), alloc(n->output()));
    }

    struct ThresholdedReluOp : Operator {
        xfn::thresholded_relu<T> op; TensorT<> X, Y;
        ThresholdedReluOp(float alpha, TensorT<>&& X, TensorT<>&& Y)
            : op(alpha), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::ThresholdedRelu* n) override {
        result = std::make_unique<ThresholdedReluOp>(
            n->alpha(), alloc(n->input()), alloc(n->output()));
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
        result = std::make_unique<EluOp>(
            n->alpha(), alloc(n->input()), alloc(n->output()));
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

    template <typename Fn>
    struct BinaryOp : Operator {
        TensorT<> A, B, C;
        BinaryOp(TensorT<>&& A, TensorT<>&& B, TensorT<>&& C)
            : A(std::move(A)), B(std::move(B)), C(std::move(C)) {}
        void evaluate() override { transformTo(A, B, C, Fn{}); }
    };

    void visit(model::Add* n) override {
        result = std::make_unique<BinaryOp<xfn::plus<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    void visit(model::Sub* n) override {
        result = std::make_unique<BinaryOp<xfn::minus<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    void visit(model::Mul* n) override {
        result = std::make_unique<BinaryOp<xfn::multiplies<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    void visit(model::Div* n) override {
        result = std::make_unique<BinaryOp<xfn::divides<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    void visit(model::Mod* n) override {
        result = std::make_unique<BinaryOp<xfn::modulus<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    void visit(model::Pow* n) override {
        result = std::make_unique<BinaryOp<xfn::power<T>>>(
            alloc(n->A()), alloc(n->B()), alloc(n->C()));
    }

    template <typename Fn>
    struct AggregateOp : Operator {
        std::list<TensorT<>> inputs;
        TensorT<> output;

        AggregateOp(std::list<TensorT<>>&& inputs, TensorT<>&& output)
            : inputs(std::move(inputs)), output(std::move(output)) {}

        void evaluate() override {
            if (inputs.size() == 0)
                return; // FIXME: fill with zero?
            if (inputs.size() == 1) {
                assert(inputs.front().shape() == output.shape());
                flat_copy(inputs.front(), output);
                return;
            }

            auto iterator = inputs.begin();
            auto& a = *iterator++;
            auto& b = *iterator++;
            transformTo(a, b, output, Fn{});
            while (iterator != inputs.end()) {
                transformTo(output, *iterator, output, Fn{});
                ++iterator;
            }
        }
    };

    void visit(model::Max* n) override {
        result = std::make_unique<AggregateOp<xfn::max<T>>>(
            alloc_all(n->inputs()), alloc(n->output()));
    }

    void visit(model::Min* n) override {
        result = std::make_unique<AggregateOp<xfn::min<T>>>(
            alloc_all(n->inputs()), alloc(n->output()));
    }

    void visit(model::Sum* n) override {
        result = std::make_unique<AggregateOp<xfn::plus<T>>>(
            alloc_all(n->inputs()), alloc(n->output()));
    }

    struct MeanOp : AggregateOp<xfn::plus<T>> {
        TensorT<> count;

        MeanOp(std::list<TensorT<>>&& inputs, TensorT<>&& output)
            : AggregateOp<xfn::plus<T>>(std::move(inputs), std::move(output))
        {
            count = TensorT<>::scalar(static_cast<T>(this->inputs.size()));
        }

        void evaluate() override {
            AggregateOp<xfn::plus<T>>::evaluate();
            transformTo(this->output, count, this->output, xfn::divides<T>());
        }
    };

    void visit(model::Mean* n) override {
        result = std::make_unique<MeanOp>(
            alloc_all(n->inputs()), alloc(n->output()));
    }

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

    struct ReshapeOp : Operator {
        TensorT<> X, Y;
        ReshapeOp(TensorT<>&& X, TensorT<>&& Y)
            : X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override { reshape(X, Y); }
    };

    void visit(model::Reshape* n) override {
        result = std::make_unique<ReshapeOp>(alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Flatten* n) override {
        result = std::make_unique<ReshapeOp>(alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Squeeze* n) override {
        result = std::make_unique<ReshapeOp>(alloc(n->input()), alloc(n->output()));
    }

    void visit(model::Unsqueeze* n) override {
        result = std::make_unique<ReshapeOp>(alloc(n->input()), alloc(n->output()));
    }

    struct ConcatOp : Operator {
        const int axis;
        std::list<TensorT<>> inputs;
        TensorT<> output;
        ConcatOp(int axis, std::list<TensorT<>>&& inputs, TensorT<>&& output)
            : axis(axis), inputs(std::move(inputs)), output(std::move(output)) {}
        void evaluate() override {
            std::vector<const TensorT<>*> tmp;
            for (const auto& t : inputs)
                tmp.push_back(&t);
            concat(axis, tmp, output);
        }
    };

    void visit(model::Concat* n) override {
        result = std::make_unique<ConcatOp>(
            n->axis(), alloc_all(n->inputs()), alloc(n->output()));
    }

    struct SplitOp : Operator {
        const int axis;
        TensorT<> input;
        std::list<TensorT<>> outputs;
        SplitOp(int axis, TensorT<>&& input, std::list<TensorT<>>&& outputs)
            : axis(axis), input(std::move(input)), outputs(std::move(outputs)) {}
        void evaluate() override {
            std::vector<TensorT<>*> tmp;
            for (auto& t : outputs)
                tmp.push_back(&t);
            split(axis, input, tmp);
        }
    };

    void visit(model::Split* n) override {
        result = std::make_unique<SplitOp>(
            n->axis(), alloc(n->input()), alloc_all(n->outputs()));
    }

    struct TransposeOp : Operator {
        std::vector<size_t> perm;
        TensorT<> X, Y;
        TransposeOp(std::vector<size_t>&& perm, TensorT<>&& X, TensorT<>&& Y)
            : perm(std::move(perm)), X(std::move(X)), Y(std::move(Y)) {}
        void evaluate() override {
            transpose(X, perm, Y);
        }
    };

    void visit(model::Transpose* n) override {
        std::vector<size_t> perm;
        if (n->has_perm()) {
            auto& x_perm = n->perm();
            perm.resize(x_perm.size());
            std::copy(x_perm.begin(), x_perm.end(), perm.begin());
        } else {
            perm.resize(n->input()->dims().size());
            std::iota(perm.begin(), perm.end(), 0);
            std::reverse(perm.begin(), perm.end());
        }
        result = std::make_unique<TransposeOp>(
            std::move(perm), alloc(n->input()), alloc(n->output()));
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
