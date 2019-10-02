#pragma once

#include "model.h"

namespace dlf { namespace predict {

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
        : dtype(dtype), shape(std::move(shape)) {}

    template <typename T>
    Tensor<T> get() {
        assert(dtype == model::DataTypeTrait<T>);
        if (data == nullptr) {
            data.reset(new char[shape.size() * sizeof(T)],
                       std::default_delete<char[]>());
        }
        return Tensor<T>::wrap(shape, reinterpret_cast<T*>(data.get()));
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

    std::shared_ptr<Datum<CPU>> makeShared(Shape dims) {
        assert(dims.size() == this->shape.size());
        return std::shared_ptr<Datum<CPU>>(
            new Datum<CPU>(this->dtype, std::move(dims), this->data));
    }

private:
    std::shared_ptr<char> data;

    Datum(model::DataType dtype, Shape&& shape, std::shared_ptr<char> data)
        : dtype(dtype), shape(std::move(shape)), data(std::move(data)) {}
};

template <> struct Datum<GPU> {
    const model::DataType dtype;
    const Shape shape;

    Datum(model::DataType dtype, Shape shape)
        : dtype(dtype), shape(std::move(shape)) {}

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

    std::shared_ptr<Datum<GPU>> makeShared(Shape dims) {
        assert(dims.size() == this->shape.size());
        return std::shared_ptr<Datum<GPU>>(
            new Datum(this->dtype, std::move(dims), this->handle));
    }

private:
    std::shared_ptr<gpgpu::rawBuffer> handle;

    Datum(model::DataType dtype, Shape&& shape, std::shared_ptr<gpgpu::rawBuffer> handle)
        : dtype(dtype), shape(std::move(shape)), handle(std::move(handle)) {}
};

class Operator {
public:
    virtual ~Operator() = default;
    virtual void evaluate() = 0;
};

template <typename Context, typename T>
class Predictor {
public:
    explicit Predictor(model::Graph& graph,
        const std::unordered_map<std::string, size_t>& env = {});

    explicit Predictor(std::unique_ptr<model::Graph> graph,
        const std::unordered_map<std::string, size_t>& env = {})
        : Predictor(*graph, env) {}

    void predict() {
        for (auto& op : m_operators) {
            op->evaluate();
        }
    }

    template <typename U>
    void set(size_t i, const Tensor<U>& data) {
        m_inputs.at(i)->template set<U>(data);
    }

    template <typename U = T>
    Tensor<U> get(size_t i) {
        return m_outputs.at(i)->template read<U>();
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
        if (value == nullptr)
            return nullptr;

        auto it = m_datamap.find(value);
        if (it != m_datamap.end()) {
            return it->second;
        }

        if (value->type() == model::DataType::UNDEFINED)
            throw std::runtime_error(cxx::string_concat("undefined value '",
                value->name(), "' in node ", value->node()->kind().str()));

        auto datum = std::make_shared<Datum<Context>>(
            model::DataTypeTrait<U>, value->dims().shape());
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
    std::list<TensorT<U>> allocAll(cxx::array_ref<model::Value*> values) {
        std::list<TensorT<U>> inputs;
        for (auto v : values)
            inputs.push_back(alloc<U>(v));
        return inputs;
    }

    template <typename U = T>
    DatumPtr allocDatumInplace(const model::Value* input, const model::Value* output) {
        if (input->uses().size() > 1 || input->has_initializer())
            return allocDatum<U>(output);

        assert(m_datamap.find(output) == m_datamap.end());
        auto input_datum = allocDatum<U>(input);
        auto output_datum = input_datum->makeShared(output->dims().shape());
        m_dataset.push_back(output_datum);
        m_datamap.emplace(output, output_datum);
        return output_datum;
    }

    template <typename U = T>
    TensorT<U> allocInplace(const model::Value* input, const model::Value* output) {
        auto datum = allocDatumInplace(input, output);
        return datum->template get<U>();
    }

    std::unique_ptr<Operator> createOperator(model::Node* node) {
        node->accept(*this);
        return std::move(result);
    }

private:
    void visitNode(model::Node* n) override {
        throw std::runtime_error(cxx::string_concat("Unsupported operator ", n->kind().str()));
    }

    struct IfOp : Operator {
        DatumPtr cond;
        std::list<TensorT<>> outputs;
        std::vector<std::unique_ptr<Operator>> then_branch;
        std::vector<std::unique_ptr<Operator>> else_branch;
        std::vector<std::shared_ptr<Datum<Context>>> then_outputs;
        std::vector<std::shared_ptr<Datum<Context>>> else_outputs;

        IfOp(OperatorFactory* of, model::If* n)
            : cond(of->allocDatum<bool>(n->input())),
              outputs(of->allocAll(n->outputs()))
        {
            for (auto x : n->then_branch()->nodes())
                then_branch.push_back(of->createOperator(x));
            for (auto v : n->then_branch()->outputs())
                then_outputs.push_back(of->allocDatum(v));

            for (auto x : n->else_branch()->nodes())
                else_branch.push_back(of->createOperator(x));
            for (auto v : n->else_branch()->outputs())
                else_outputs.push_back(of->allocDatum(v));
        }

        void evaluate() override {
            auto b = cond->template read<bool>();
            assert(b.rank() == 0);
            auto& branch = *b ? then_branch : else_branch;
            auto& results = *b ? then_outputs : else_outputs;

            for (auto& op : branch)
                op->evaluate();
            auto it = outputs.begin();
            for (auto& v : results) {
                flat_copy(v->template get<T>(), *it);
                ++it;
            }
        }
    };

    void visit(model::If* n) override {
        result = std::make_unique<IfOp>(this, n);
    }

    struct LoopOp : Operator {
        std::vector<std::shared_ptr<Datum<Context>>> inputs;
        std::vector<std::shared_ptr<Datum<Context>>> outputs;

        std::vector<std::unique_ptr<Operator>> body;
        std::vector<std::shared_ptr<Datum<Context>>> body_inputs;
        std::vector<std::shared_ptr<Datum<Context>>> body_outputs;

        LoopOp(OperatorFactory* of, model::Loop* n) {
            assert(n->inputs().size() == n->body()->inputs().size());
            assert(n->outputs().size() == n->body()->outputs().size() - 1);

            for (auto x : n->body()->nodes())
                body.push_back(of->createOperator(x));

            body_inputs.push_back(of->allocDatum(n->body()->input(0)));         // iteration
            body_inputs.push_back(of->allocDatum<bool>(n->body()->input(1)));   // cond
            for (int i = 2; i < n->body()->inputs().size(); ++i)                // state vars
                body_inputs.push_back(of->allocDatum(n->body()->input(i)));

            body_outputs.push_back(of->allocDatum<bool>(n->body()->output(0))); // output cond
            for (int i = 1; i < n->body()->outputs().size(); ++i)               // final outputs
                body_outputs.push_back(of->allocDatum(n->body()->output(i)));

            inputs.push_back(of->allocDatum<int64_t>(n->M()));  // max trip count
            inputs.push_back(of->allocDatum<bool>(n->cond()));  // initial condition
            for (int i = 2; i < n->inputs().size(); ++i)
                inputs.push_back(of->allocDatum(n->input(i)));
            for (size_t i = 0; i < n->outputs().size(); ++i)
                outputs.push_back(of->allocDatumInplace(n->body()->output(i+1), n->output(i)));
        }

        void evaluate() override {
            bool cond = (inputs[1] == nullptr) ? true : *(inputs[1]->template read<bool>());
            if (inputs[0] == nullptr) {
                cond = run_body(0, true, cond);
                for (int64_t i = 1; cond; ++i) {
                    cond = run_body(i, false, cond);
                }
            } else if (inputs[1] == nullptr) {
                auto trip_count = *(inputs[0]->template read<int64_t>());
                for (int64_t i = 0; i < trip_count; ++i) {
                    run_body(i, i==0, true);
                }
            } else {
                auto trip_count = *(inputs[0]->template read<int64_t>());
                for (int64_t i = 0; i < trip_count && cond; ++i) {
                    cond = run_body(i, i==0, cond);
                }
            }

            for (int i = 0; i < outputs.size(); ++i) {
                copy_data(body_outputs[i+1], outputs[i]);
            }
        }

    private:
        bool run_body(int64_t iteration, bool first, bool cond) {
            body_inputs[0]->set(Scalar<T>(iteration));
            body_inputs[1]->set(Scalar<bool>(cond));

            if (first) {
                for (int i = 2; i < inputs.size(); ++i) {
                    copy_data(inputs[i], body_inputs[i]);
                }
            } else {
                for (int i = 2; i < inputs.size(); ++i) {
                    copy_data(body_outputs[i-1], body_inputs[i]);
                }
            }

            for (auto& op : body) {
                op->evaluate();
            }

            return *(body_outputs[0]->template read<bool>());
        }

        void copy_data(DatumPtr src, DatumPtr dst) {
            auto src_data = src->template get<T>();
            auto dst_data = dst->template get<T>();
            flat_copy(src_data, dst_data);
        }
    };

    void visit(model::Loop* n) override {
        result = std::make_unique<LoopOp>(this, n);
    }

    struct EyeLikeOp : Operator {
        TensorT<> output;
        int k;
        EyeLikeOp(OperatorFactory* of, model::EyeLike* n)
            : output(of->alloc(n->output())), k(n->get_i("k", 0)) {}
        void evaluate() override {
            output.fill(0);
            output.diagonal(k).fill(1);
        }
    };

    void visit(model::EyeLike* n) override {
        result = std::make_unique<EyeLikeOp>(this, n);
    }

    struct RandomNormalOp : Operator {
        TensorT<> output;
        T mean, scale;
        RandomNormalOp(OperatorFactory* of, model::Node* n, T mean, T scale)
            : output(of->alloc(n->output())), mean(mean), scale(scale) {}
        void evaluate() override {
            output.random(std::normal_distribution<T>(mean, scale));
        }
    };

    void visit(model::RandomNormal* n) override {
        result = std::make_unique<RandomNormalOp>(this, n, n->mean(), n->scale());
    }

    void visit(model::RandomNormalLike* n) override {
        result = std::make_unique<RandomNormalOp>(this, n, n->mean(), n->scale());
    }

    struct RandomUniformOp : Operator {
        TensorT<> output;
        T low, high;
        RandomUniformOp(OperatorFactory* of, model::Node* n, T low, T high)
            : output(of->alloc(n->output())), low(low), high(high) {}
        void evaluate() override {
            output.random(low, high);
        }
    };

    void visit(model::RandomUniform* n) override {
        result = std::make_unique<RandomUniformOp>(this, n, n->low(), n->high());
    }

    void visit(model::RandomUniformLike* n) override {
        result = std::make_unique<RandomUniformOp>(this, n, n->low(), n->high());
    }

    template <typename Fn>
    struct UnaryOp : Operator {
        TensorT<> X, Y;
        UnaryOp(OperatorFactory* of, model::Node* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { transformTo(X, Y, Fn{}); }
    };

    #define DEFINE_UNARY_OPERATOR(Name, fn) \
    void visit(model::Name* n) override { \
        result = std::make_unique<UnaryOp<xfn::fn<T>>>(this, n); \
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
        ClipOp(OperatorFactory* of, model::Clip* n)
            : op(n->min(), n->max()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(X, Y, op);
        }
    };

    void visit(model::Clip* n) override {
        result = std::make_unique<ClipOp>(this, n);
    }

    struct ShrinkOp : Operator {
        xfn::shrink<T> op; TensorT<> X, Y;
        ShrinkOp(OperatorFactory* of, model::Shrink* n)
            : op(n->lambd(), n->bias()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(X, Y, op);
        }
    };

    void visit(model::Shrink* n) override {
        result = std::make_unique<ShrinkOp>(this, n);
    }

    struct ReluOp : Operator {
        TensorT<> X, Y;
        ReluOp(OperatorFactory* of, model::Relu* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(X, Y, xfn::relu<T>());
        }
    };

    void visit(model::Relu* n) override {
        result = std::make_unique<ReluOp>(this, n);
    }

    struct PReluOp : Operator {
        TensorT<> X, slope, Y;
        PReluOp(OperatorFactory* of, model::PRelu* n)
            : X(of->alloc(n->input())),
              slope(of->alloc(n->slope())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(X, slope, Y, xfn::prelu<T>());
        }
    };

    void visit(model::PRelu* n) override {
        result = std::make_unique<PReluOp>(this, n);
    }

    struct LeakyReluOp : Operator {
        xfn::leaky_relu<T> op; TensorT<> X, Y;
        LeakyReluOp(OperatorFactory* of, model::LeakyRelu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::LeakyRelu* n) override {
        result = std::make_unique<LeakyReluOp>(this, n);
    }

    struct ThresholdedReluOp : Operator {
        xfn::thresholded_relu<T> op; TensorT<> X, Y;
        ThresholdedReluOp(OperatorFactory* of, model::ThresholdedRelu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::ThresholdedRelu* n) override {
        result = std::make_unique<ThresholdedReluOp>(this, n);
    }

    struct SeluOp : Operator {
        xfn::selu<T> op; TensorT<> X, Y;
        SeluOp(OperatorFactory* of, model::Selu* n)
            : op(n->alpha(), n->gamma()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::Selu* n) override {
        result = std::make_unique<SeluOp>(this, n);
    }

    struct EluOp : Operator {
        xfn::elu<T> op; TensorT<> X, Y;
        EluOp(OperatorFactory* of, model::Elu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::Elu* n) override {
        result = std::make_unique<EluOp>(this, n);
    }

    struct HardSigmoidOp : Operator {
        xfn::hard_sigmoid<T> op; TensorT<> X, Y;
        HardSigmoidOp(OperatorFactory* of, model::HardSigmoid* n)
            : op(n->alpha(), n->beta()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, op); }
    };

    void visit(model::HardSigmoid* n) override {
        result = std::make_unique<HardSigmoidOp>(this, n);
    }

    struct SoftmaxOp : Operator {
        TensorT<> X, Y; int axis;
        SoftmaxOp(OperatorFactory* of, model::Softmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override { dnn::softmax(X, Y, axis); }
    };

    void visit(model::Softmax* n) override {
        result = std::make_unique<SoftmaxOp>(this, n);
    }

    struct LogSoftmaxOp : Operator {
        TensorT<> X, Y; int axis;
        LogSoftmaxOp(OperatorFactory* of, model::LogSoftmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override { dnn::logsoftmax(X, Y, axis); }
    };

    void visit(model::LogSoftmax* n) override {
        result = std::make_unique<LogSoftmaxOp>(this, n);
    }

    struct HardmaxOp : Operator {
        TensorT<> X, Y; int axis;
        HardmaxOp(OperatorFactory* of, model::Hardmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override { dnn::hardmax(X, Y, axis); }
    };

    void visit(model::Hardmax* n) override {
        result = std::make_unique<HardmaxOp>(this, n);
    }

    struct SoftsignOp : Operator {
        TensorT<> X, Y;
        SoftsignOp(OperatorFactory* of, model::Softsign* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, xfn::softsign<T>()); }
    };

    void visit(model::Softsign* n) override {
        result = std::make_unique<SoftsignOp>(this, n);
    }

    struct SoftplusOp : Operator {
        TensorT<> X, Y;
        SoftplusOp(OperatorFactory* of, model::Softplus* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { dlf::transformTo(X, Y, xfn::softplus<T>()); }
    };

    void visit(model::Softplus* n) override {
        result = std::make_unique<SoftplusOp>(this, n);
    }

    template <typename Fn>
    struct BinaryOp : Operator {
        TensorT<> A, B, C;

        BinaryOp(OperatorFactory* of, model::Node* n)
            : A(of->alloc(n->input(0))), B(of->alloc(n->input(1)))
        {
            auto shape = n->output()->dims().shape();
            if (A.shape() == shape &&
                    n->input(0)->uses().size() == 1 &&
                   !n->input(0)->has_initializer()) {
                C = of->allocInplace(n->input(0), n->output());
            } else if (B.shape() == shape &&
                    n->input(1)->uses().size() == 1 &&
                   !n->input(1)->has_initializer()) {
                C = of->allocInplace(n->input(1), n->output());
            } else {
                C = of->alloc(n->output());
            }
        }

        void evaluate() override { transformTo(A, B, C, Fn{}); }
    };

    void visit(model::Add* n) override {
        result = std::make_unique<BinaryOp<xfn::plus<T>>>(this, n);
    }

    void visit(model::Sub* n) override {
        result = std::make_unique<BinaryOp<xfn::minus<T>>>(this, n);
    }

    void visit(model::Mul* n) override {
        result = std::make_unique<BinaryOp<xfn::multiplies<T>>>(this, n);
    }

    void visit(model::Div* n) override {
        result = std::make_unique<BinaryOp<xfn::divides<T>>>(this, n);
    }

    void visit(model::Mod* n) override {
        result = std::make_unique<BinaryOp<xfn::modulus<T>>>(this, n);
    }

    void visit(model::Pow* n) override {
        result = std::make_unique<BinaryOp<xfn::power<T>>>(this, n);
    }

    template <typename Fn>
    struct RelationOp : Operator {
        TensorT<> A, B;
        TensorT<bool> C;
        RelationOp(OperatorFactory* of, model::Node* n)
            : A(of->alloc(n->input(0))), B(of->alloc(n->input(1))),
              C(of->alloc<bool>(n->output())) {}
        void evaluate() override { transformTo(A, B, C, Fn{}); }
    };

    void visit(model::Greater* n) override {
        result = std::make_unique<RelationOp<xfn::greater<T>>>(this, n);
    }

    void visit(model::Less* n) override {
        result = std::make_unique<RelationOp<xfn::less<T>>>(this, n);
    }

    void visit(model::Equal* n) override {
        result = std::make_unique<RelationOp<xfn::equal_to<T>>>(this, n);
    }

    template <typename Fn>
    struct AggregateOp : Operator {
        std::list<TensorT<>> inputs;
        TensorT<> output;

        AggregateOp(OperatorFactory* of, model::Node* n)
            : inputs(of->allocAll(n->inputs())),
              output(of->alloc(n->output())) {}

        void evaluate() override {
            if (inputs.size() == 0) {
                output.fill(T{}); // fill with zero
                return;
            }
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
        result = std::make_unique<AggregateOp<xfn::max<T>>>(this, n);
    }

    void visit(model::Min* n) override {
        result = std::make_unique<AggregateOp<xfn::min<T>>>(this, n);
    }

    void visit(model::Sum* n) override {
        result = std::make_unique<AggregateOp<xfn::plus<T>>>(this, n);
    }

    struct MeanOp : AggregateOp<xfn::plus<T>> {
        TensorT<> count;

        MeanOp(OperatorFactory* of, model::Node* n)
            : AggregateOp<xfn::plus<T>>(of, n)
        {
            count = TensorT<>::scalar(static_cast<T>(this->inputs.size()));
        }

        void evaluate() override {
            AggregateOp<xfn::plus<T>>::evaluate();
            transformTo(this->output, count, this->output, xfn::divides<T>());
        }
    };

    void visit(model::Mean* n) override {
        result = std::make_unique<MeanOp>(this, n);
    }

    template <typename Reducer>
    struct ReductionOp : Operator {
        TensorT<> X, Y;
        std::vector<int> axes;
        bool keepdims;

        ReductionOp(OperatorFactory* of, model::Node* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output()))
        {
            auto v = n->get_is(model::kaxes, {});
            axes.assign(v.begin(), v.end());
            keepdims = n->get_i(model::kkeepdims, 1) != 0;
        }

        void evaluate() override {
            dlf::reduce<Reducer>(X, Y, axes, keepdims);
        }
    };

    void visit(model::ReduceMax* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_max<T>>>(this, n);
    }

    void visit(model::ReduceMin* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_min<T>>>(this, n);
    }

    void visit(model::ReduceSum* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_sum<T>>>(this, n);
    }

    void visit(model::ReduceSumSquare* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_sum_square<T>>>(this, n);
    }

    void visit(model::ReduceMean* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_mean<T>>>(this, n);
    }

    void visit(model::ReduceProd* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_prod<T>>>(this, n);
    }

    void visit(model::ReduceLogSum* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_log_sum<T>>>(this, n);
    }

    void visit(model::ReduceLogSumExp* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_log_sum_exp<T>>>(this, n);
    }

    void visit(model::ReduceL1* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_asum<T>>>(this, n);
    }

    void visit(model::ReduceL2* n) override {
        result = std::make_unique<ReductionOp<xfn::reduce_nrm2<T>>>(this, n);
    }

    struct ArgMaxOp : Operator {
        TensorT<> X;
        TensorT<int> Y;
        int axis;
        bool keepdims;

        ArgMaxOp(OperatorFactory* of, model::ArgMax* n)
            : X(of->alloc(n->input())),
              Y(of->alloc<int>(n->output())),
              axis(n->axis()), keepdims(n->keepdims()) {}

        void evaluate() override {
            argmax(X, Y, axis, keepdims);
        }
    };

    void visit(model::ArgMax* n) override {
        result = std::make_unique<ArgMaxOp>(this, n);
    }

    struct ArgMinOp : Operator {
        TensorT<> X;
        TensorT<int> Y;
        int axis;
        bool keepdims;

        ArgMinOp(OperatorFactory* of, model::ArgMin* n)
            : X(of->alloc(n->input())),
              Y(of->alloc<int>(n->output())),
              axis(n->axis()), keepdims(n->keepdims()) {}

        void evaluate() override {
            argmin(X, Y, axis, keepdims);
        }
    };

    void visit(model::ArgMin* n) override {
        result = std::make_unique<ArgMinOp>(this, n);
    }

    struct CumSumOp : Operator {
        TensorT<> input, output;
        bool exclusive, reverse;
        DatumPtr axis;

        CumSumOp(OperatorFactory* of, model::CumSum* n)
            : input(of->alloc(n->input())),
              output(of->allocInplace(n->input(), n->output())),
              exclusive(n->exclusive()),
              reverse(n->reverse()),
              axis(of->allocDatum<int>(n->axis())) {}

        void evaluate() override {
            auto a = axis->template read<int>();
            assert(a.rank() == 0);
            cumsum(input, output, *a, exclusive, reverse);
        }
    };

    void visit(model::CumSum* n) override {
        result = std::make_unique<CumSumOp>(this, n);
    }

    struct GemmOp : Operator {
        T alpha, beta;
        cblas::Transpose transA, transB;
        TensorT<> A, B, C, Y;

        GemmOp(OperatorFactory* of, model::Gemm* n)
            : alpha(n->alpha()), beta(n->beta()),
              transA(n->transA() ? cblas::Transpose::Trans : cblas::Transpose::NoTrans),
              transB(n->transB() ? cblas::Transpose::Trans : cblas::Transpose::NoTrans),
              A(of->alloc(n->A())),
              B(of->alloc(n->B())),
              C(of->alloc(n->C())),
              Y(of->alloc(n->Y())) {}

        void evaluate() override {
            gemm(transA, transB, alpha, A, B, beta, C, Y);
        }
    };

    void visit(model::Gemm* n) override {
        result = std::make_unique<GemmOp>(this, n);
    }

    struct MatMulOp : Operator {
        TensorT<> A, B, C;
        MatMulOp(OperatorFactory* of, model::MatMul* n)
            : A(of->alloc(n->A())), B(of->alloc(n->B())), C(of->alloc(n->C())) {}
        void evaluate() override {
            matmul(A, B, C);
        }
    };

    void visit(model::MatMul* n) override {
        result = std::make_unique<MatMulOp>(this, n);
    }

    struct ConvOp : Operator {
        TensorT<> X, W, B, Y;
        dnn::Filter2D filter;

        ConvOp(OperatorFactory* of, model::Conv* n)
            : X(of->alloc(n->X())),
              W(of->alloc(n->W())),
              B(n->B() ? of->alloc(n->B()) : TensorT<>()),
              Y(of->alloc(n->Y())),
              filter(dnn::Filter2D(X.shape(), W.shape(), n->group())
                .pads(n->pads())
                .strides(n->strides())
                .dilations(n->dilations())) {}

        void evaluate() override {
            dnn::conv2d(X, W, Y, filter);
            if (!B.empty()) {
                transformChannel(Y, B, Y, 1, xfn::plus<T>());
            }
        }
    };

    void visit(model::Conv* n) override {
        result = std::make_unique<ConvOp>(this, n);
    }

    struct MaxPoolOp : Operator {
        TensorT<> X, Y; dnn::Filter2D filter;
        MaxPoolOp(OperatorFactory* of, model::MaxPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X.shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())
                .dilations(n->dilations())) {}
        void evaluate() override { dnn::max_pooling(X, Y, filter); }
    };

    void visit(model::MaxPool* n) override {
        result = std::make_unique<MaxPoolOp>(this, n);
    }

    struct AveragePoolOp : Operator {
        TensorT<> X, Y; dnn::Filter2D filter;
        const bool count_include_pad;
        AveragePoolOp(OperatorFactory* of, model::AveragePool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X.shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())),
              count_include_pad(n->count_include_pad()) {}
        void evaluate() override { dnn::average_pooling(X, Y, filter, count_include_pad); }
    };

    void visit(model::AveragePool* n) override {
        result = std::make_unique<AveragePoolOp>(this, n);
    }

    struct LpPoolOp : Operator {
        TensorT<> X, Y; dnn::Filter2D filter; int p;
        LpPoolOp(OperatorFactory* of, model::LpPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X.shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())),
              p(n->get_i("p", 2)) {}
        void evaluate() override { dnn::lp_pooling(X, Y, filter, p); }
    };

    void visit(model::LpPool* n) override {
        result = std::make_unique<LpPoolOp>(this, n);
    }

    struct GlobalMaxPoolOp : Operator {
        TensorT<> X, Y;
        GlobalMaxPoolOp(OperatorFactory* of, model::GlobalMaxPool* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())) {}
        void evaluate() override { dnn::global_max_pooling(X, Y); }
    };

    void visit(model::GlobalMaxPool* n) override {
        result = std::make_unique<GlobalMaxPoolOp>(this, n);
    }

    struct GlobalAveragePoolOp : Operator {
        TensorT<> X, Y;
        GlobalAveragePoolOp(OperatorFactory* of, model::GlobalAveragePool* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())) {}
        void evaluate() override { dnn::global_average_pooling(X, Y); }
    };

    void visit(model::GlobalAveragePool* n) override {
        result = std::make_unique<GlobalAveragePoolOp>(this, n);
    }

    struct GlobalLpPoolOp : Operator {
        TensorT<> X, Y; int p;
        GlobalLpPoolOp(OperatorFactory* of, model::GlobalLpPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              p(n->get_i("p", 2)) {}
        void evaluate() override { dnn::global_lp_pooling(X, Y, p); }
    };

    void visit(model::GlobalLpPool* n) override {
        result = std::make_unique<GlobalLpPoolOp>(this, n);
    }

    struct BatchNormalizationOp : Operator {
        TensorT<> X, Y, S, B, M, V;
        T epsilon;
        BatchNormalizationOp(OperatorFactory* of, model::BatchNormalization* n)
            : X(of->alloc(n->X())),
              Y(of->allocInplace(n->X(), n->Y())),
              S(of->alloc(n->scale())),
              B(of->alloc(n->B())),
              M(of->alloc(n->mean())),
              V(of->alloc(n->var())),
              epsilon(n->epsilon()) {}
        void evaluate() override {
            dnn::batch_norm(X, Y, S, B, M, V, epsilon);
        }
    };

    void visit(model::BatchNormalization* n) override {
        result = std::make_unique<BatchNormalizationOp>(this, n);
    }

    struct LRNOp : Operator {
        TensorT<> X, Y; int n; T alpha, beta, bias;
        LRNOp(OperatorFactory* of, model::LRN* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              n(n->size()),
              alpha(n->alpha()), beta(n->beta()), bias(n->bias()) {}
        void evaluate() override {
            dnn::lrn(X, Y, n, alpha, beta, bias);
        }
    };

    void visit(model::LRN* n) override {
        result = std::make_unique<LRNOp>(this, n);
    }

    struct ReshapeOp : Operator {
        TensorT<> X, Y;
        ReshapeOp(OperatorFactory* of, model::Node* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override { reshape(X, Y); }
    };

    void visit(model::Reshape* n) override {
        result = std::make_unique<ReshapeOp>(this, n);
    }

    void visit(model::Flatten* n) override {
        result = std::make_unique<ReshapeOp>(this, n);
    }

    void visit(model::Squeeze* n) override {
        result = std::make_unique<ReshapeOp>(this, n);
    }

    void visit(model::Unsqueeze* n) override {
        result = std::make_unique<ReshapeOp>(this, n);
    }

    struct ConcatOp : Operator {
        std::list<TensorT<>> inputs;
        TensorT<> output;
        int axis;

        ConcatOp(OperatorFactory* of, model::Concat* n)
            : inputs(of->allocAll(n->inputs())),
              output(of->alloc(n->output())),
              axis(n->axis()) {}

        void evaluate() override {
            concat(axis, inputs.begin(), inputs.end(), output);
        }
    };

    void visit(model::Concat* n) override {
        result = std::make_unique<ConcatOp>(this, n);
    }

    struct SplitOp : Operator {
        TensorT<> input;
        std::list<TensorT<>> outputs;
        int axis;

        SplitOp(OperatorFactory* of, model::Split* n)
            : input(of->alloc(n->input())),
              outputs(of->allocAll(n->outputs())),
              axis(n->axis()) {}

        void evaluate() override {
            split(input, axis, outputs.begin(), outputs.end());
        }
    };

    void visit(model::Split* n) override {
        result = std::make_unique<SplitOp>(this, n);
    }

    struct GatherOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;
        int axis;

        GatherOp(OperatorFactory* of, model::Gather* n)
            : input(of->alloc(n->input())),
              output(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())),
              axis(n->axis()) {}

        void evaluate() override {
            gather(input, output, indices, axis);
        }
    };

    void visit(model::Gather* n) override {
        result = std::make_unique<GatherOp>(this, n);
    }

    // deprecated, same as ScatterElements
    struct ScatterOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;
        TensorT<> updates;
        int axis;

        ScatterOp(OperatorFactory* of, model::Scatter* n)
            : input(of->alloc(n->input())),
              output(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())),
              axis(n->axis()) {}

        void evaluate() override {
            flat_copy(input, output);
            scatter_elements(output, indices, updates, axis);
        }
    };

    void visit(model::Scatter* n) override {
        result = std::make_unique<ScatterOp>(this, n);
    }

    struct GatherElementsOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;
        int axis;

        GatherElementsOp(OperatorFactory* of, model::GatherElements* n)
            : input(of->alloc(n->input())),
              output(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())),
              axis(n->axis()) {}

        void evaluate() override {
            gather_elements(input, output, indices, axis);
        }
    };

    void visit(model::GatherElements* n) override {
        result = std::make_unique<GatherElementsOp>(this, n);
    }

    struct ScatterElementsOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;
        TensorT<> updates;
        int axis;

        ScatterElementsOp(OperatorFactory* of, model::ScatterElements* n)
            : input(of->alloc(n->input())),
              output(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())),
              axis(n->axis()) {}

        void evaluate() override {
            flat_copy(input, output);
            scatter_elements(output, indices, updates, axis);
        }
    };

    void visit(model::ScatterElements* n) override {
        result = std::make_unique<ScatterElementsOp>(this, n);
    }

    struct GatherNDOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;

        GatherNDOp(OperatorFactory* of, model::GatherND* n)
            : input(of->alloc(n->input())),
              output(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())) {}

        void evaluate() override {
            gather_nd(input, output, indices);
        }
    };

    void visit(model::GatherND* n) override {
        result = std::make_unique<GatherNDOp>(this, n);
    }

    struct ScatterNDOp : Operator {
        TensorT<> input, output;
        TensorT<int> indices;
        TensorT<> updates;

        ScatterNDOp(OperatorFactory* of, model::ScatterND* n)
            : input(of->alloc(n->input())),
              output(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())) {}

        void evaluate() override {
            flat_copy(input, output);
            scatter_nd(output, indices, updates);
        }
    };

    void visit(model::ScatterND* n) override {
        result = std::make_unique<ScatterNDOp>(this, n);
    }

    struct SliceOp : Operator {
        TensorT<> X, Y;
        DatumPtr starts, ends, axes, steps;

        SliceOp(OperatorFactory* of, model::Slice* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              starts(of->allocDatum<int>(n->starts())),
              ends(of->allocDatum<int>(n->ends())),
              axes(n->axes()==nullptr ? nullptr : of->allocDatum<int>(n->axes())),
              steps(n->steps()==nullptr ? nullptr : of->allocDatum<int>(n->steps())) {}

        void evaluate() override {
            slice(X, Y, read(starts), read(ends), read(axes), read(steps));
        }

        std::vector<int> read(DatumPtr datum) {
            if (datum == nullptr) {
                return {};
            } else {
                auto v = datum->template read<int>();
                assert(v.rank() == 1);
                return {v.begin(), v.end()};
            }
        }
    };

    void visit(model::Slice* n) override {
        result = std::make_unique<SliceOp>(this, n);
    }

    struct TransposeOp : Operator {
        TensorT<> X, Y;
        std::vector<size_t> perm;

        TransposeOp(OperatorFactory* of, model::Transpose* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output()))
        {
            if (n->has_perm()) {
                auto& x_perm = n->perm();
                perm.assign(x_perm.begin(), x_perm.end());
            } else {
                perm.resize(X.rank());
                std::iota(perm.begin(), perm.end(), 0);
                std::reverse(perm.begin(), perm.end());
            }
        }

        void evaluate() override { transpose(X, Y, perm); }
    };

    void visit(model::Transpose* n) override {
        result = std::make_unique<TransposeOp>(this, n);
    }

    struct PadOp : Operator {
        TensorT<> X, Y;
        std::vector<int> pads;
        PadMode mode = PadMode::Constant;
        T value{};

        PadOp(OperatorFactory* of, model::Pad* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              pads(n->pads().begin(), n->pads().end()),
              value(static_cast<T>(n->value()))
        {
            if (n->has_mode()) {
                auto smode = n->mode();
                if (smode == "constant") {
                    this->mode = PadMode::Constant;
                } else if (smode == "edge") {
                    this->mode = PadMode::Edge;
                } else if (smode == "reflect") {
                    this->mode = PadMode::Reflect;
                } else {
                    throw std::runtime_error(cxx::string_concat(
                        "Unsupported pad mode: ", smode));
                }
            }
        }

        void evaluate() override {
            pad(X, Y, pads, mode, value);
        }
    };

    void visit(model::Pad* n) override {
        result = std::make_unique<PadOp>(this, n);
    }

    struct TileOp : Operator {
        TensorT<> X, Y;
        DatumPtr reps;

        TileOp(OperatorFactory* of, model::Tile* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              reps(of->allocDatum<int>(n->repeats())) {}

        void evaluate() override {
            auto v = reps->template read<int>();
            assert(v.rank() == 1);
            tile(X, Y, std::vector<size_t>(v.begin(), v.end()));
        }
    };

    void visit(model::Tile* n) override {
        result = std::make_unique<TileOp>(this, n);
    }

    struct SpaceToDepthOp : Operator {
        TensorT<> X, Y;
        int blocksize;
        SpaceToDepthOp(OperatorFactory* of, model::SpaceToDepth* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              blocksize(n->blocksize()) {}
        void evaluate() override {
            dnn::space_to_depth(X, Y, blocksize);
        }
    };

    void visit(model::SpaceToDepth* n) override {
        result = std::make_unique<SpaceToDepthOp>(this, n);
    }

    struct DepthToSpaceOp : Operator {
        TensorT<> X, Y;
        int blocksize;
        std::string mode;
        DepthToSpaceOp(OperatorFactory* of, model::DepthToSpace* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              blocksize(n->blocksize()),
              mode(n->mode()) {}
        void evaluate() override {
            dnn::depth_to_space(X, Y, blocksize, mode);
        }
    };

    void visit(model::DepthToSpace* n) override {
        result = std::make_unique<DepthToSpaceOp>(this, n);
    }

    struct WhereOp : Operator {
        TensorT<bool> C;
        TensorT<> X, Y, Z;
        WhereOp(OperatorFactory* of, model::Where* n)
            : C(of->alloc<bool>(n->condition())),
              X(of->alloc(n->X())),
              Y(of->alloc(n->Y())),
              Z(of->alloc(n->Z())) {}
        void evaluate() override { where(C, X, Y, Z); }
    };

    void visit(model::Where* n) override {
        result = std::make_unique<WhereOp>(this, n);
    }
};

template <typename Context, typename T>
Predictor<Context, T>::Predictor(model::Graph& graph,
    const std::unordered_map<std::string, size_t>& env)
{
    OperatorFactory<Context, T> factory(m_dataset);

    model::ShapeInference::newInstance(env)->infer(graph);
    model::Optimizer::newInstance()->optimize(graph);

    for (auto n : graph.nodes()) {
        m_operators.push_back(factory.createOperator(n));
    }
    for (auto v : graph.inputs()) {
        if (!v->has_initializer())
            m_inputs.push_back(factory.allocDatum(v));
    }
    for (auto v : graph.outputs()) {
        m_outputs.push_back(factory.allocDatum(v));
    }
}

}} // namespace dlf::predict
