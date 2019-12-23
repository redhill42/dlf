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
private:
    model::DataType         m_dtype;
    Shape                   m_shape;
    std::vector<uint8_t>    m_data;

public:
    Datum(model::DataType dtype, Shape shape) :
        m_dtype(dtype), m_shape(std::move(shape)) {}

    model::DataType dtype() const noexcept {
        return m_dtype;
    }

    const Shape& shape() const noexcept {
        return m_shape;
    }

    template <typename T>
    void resize(Shape new_shape) {
        if (m_shape != new_shape) {
            m_data.resize(new_shape.size() * sizeof(T));
            m_shape = std::move(new_shape);
        }
    }

    template <typename T>
    Tensor<T> get() {
        assert(m_dtype == model::DataTypeTrait<T>); // FIXME
        if (m_data.empty() && m_shape.size() != 0)
            m_data.resize(m_shape.size() * sizeof(T));
        return Tensor<T>::wrap(m_shape, reinterpret_cast<T*>(m_data.data()));
    }

    template <typename T>
    void unget(const Tensor<T>& val) {
        resize<T>(val.shape());
        if (reinterpret_cast<const void*>(val.data()) != m_data.data()) {
            flat_copy(val, get<T>());
        }
    }

    template <typename T>
    Tensor<T> read() {
        return get<T>();
    }

    template <typename T>
    void set(const Tensor<T>& val) {
        resize<T>(val.shape());
        flat_copy(val, get<T>());
    }
};

template <> struct Datum<GPU> {
private:
    model::DataType         m_dtype;
    Shape                   m_shape;
    gpgpu::Buffer<uint8_t>  m_data;

public:
    Datum(model::DataType dtype, Shape shape)
        : m_dtype(dtype), m_shape(std::move(shape)) {}

    model::DataType dtype() const noexcept {
        return m_dtype;
    }

    const Shape& shape() const noexcept {
        return m_shape;
    }

    template <typename T>
    void resize(Shape new_shape) {
        auto new_size = new_shape.size() * sizeof(T);
        if (m_data.handle() != nullptr && m_data.data_size() < new_size) {
            auto new_data = gpgpu::current::context().createBuffer<uint8_t>(new_size);
            m_data.copyToAsync(gpgpu::current::queue(), new_data, m_shape.size() * sizeof(T));
            m_data = std::move(new_data);
        }
        m_shape = std::move(new_shape);
    }

    template <typename T>
    DevTensor<T> get() {
        assert(m_dtype == model::DataTypeTrait<T>);
        if (m_data.handle() == nullptr && m_shape.size() != 0)
            m_data = gpgpu::current::context().createBuffer<uint8_t>(m_shape.size() * sizeof(T));
        return DevTensor<T>(m_shape, gpgpu::Buffer<T>(m_data.handle(), m_data.size() / sizeof(T)));
    }

    template <typename T>
    void unget(const DevTensor<T>& val) {
        auto buf = val.data();
        if (m_data.handle() != buf.handle())
            m_data = gpgpu::Buffer<uint8_t>(buf.handle(), buf.data_size());
        m_shape = val.shape();
    }

    template <typename T>
    Tensor<T> read() {
        return get<T>().read();
    }

    template <typename T>
    void set(const Tensor<T>& val) {
        resize<T>(val.shape());
        get<T>().write(val);
    }
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

    void predict();

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
    using datum_ptr = std::shared_ptr<Datum<Context>>;
    using datum_list = std::vector<datum_ptr>;
    using op_list = std::vector<std::unique_ptr<Operator>>;

    std::vector<datum_ptr>& m_dataset;
    std::unordered_map<const model::Value*, datum_ptr> m_datamap;
    std::unique_ptr<Operator> result;

    template <typename U = T>
    using TensorT = typename Context::template TensorT<U>;

    template <typename U = T>
    struct DatumValue {
        U value;
        datum_ptr datum;

        DatumValue(OperatorFactory* of, model::Value* v, U deflt = 0) {
            if (v == nullptr)
                value = deflt;
            else if (v->has_initializer())
                value = *(v->initializer().decode<U>());
            else
                datum = of->alloc<U>(v);
        }

        U operator*() {
            if (datum == nullptr)
                return value;
            return *(datum->template read<U>());
        }
    };

    template <typename U, typename V = U>
    struct DatumValues {
        std::vector<U> values;
        datum_ptr datum;

        DatumValues(OperatorFactory* of, model::Value* v) {
            if (v != nullptr) {
                if (v->has_initializer()) {
                    auto t = v->initializer().decode<V>();
                    assert(t.rank() == 1);
                    values.assign(t.begin(), t.end());
                } else {
                    datum = of->alloc<V>(v);
                }
            }
        }

        std::vector<U> operator*() {
            if (datum == nullptr) {
                return values;
            } else {
                auto t = datum->template read<V>();
                assert(t.rank() == 1);
                return std::vector<U>(t.begin(), t.end());
            }
        }
    };

    struct DatumShape {
        DatumValues<size_t, int64_t> values;

        DatumShape(OperatorFactory* of, model::Value* v)
            : values(of, v) {}

        Shape operator*() {
            return Shape(*values);
        }
    };

public:
    OperatorFactory(std::vector<datum_ptr>& dataset)
        : m_dataset(dataset) {}

    std::unique_ptr<Operator> createOperator(model::Node* node) {
        node->accept(*this);
        return std::move(result);
    }

    template <typename U = T>
    datum_ptr alloc(const model::Value* value) {
        if (value == nullptr)
            return nullptr;

        auto it = m_datamap.find(value);
        if (it != m_datamap.end()) {
            return it->second;
        }

        if (value->type() == model::DataType::UNDEFINED)
            throw std::runtime_error(cxx::string_concat("undefined value '",
                value->name(), "' in node ", value->node()->kind().str()));

        auto datum = std::make_shared<Datum<Context>>(model::DataTypeTrait<U>, value->shape());
        m_dataset.push_back(datum);
        m_datamap.emplace(value, datum);
        if (value->has_initializer())
            datum->template set<U>(value->initializer().decode<U>());
        return datum;
    }

    template <typename U = T>
    datum_list allocAll(cxx::array_ref<model::Value*> values) {
        std::vector<datum_ptr> inputs;
        for (auto v : values)
            inputs.push_back(alloc<U>(v));
        return inputs;
    }

    template <typename U = T>
    datum_ptr allocInplace(const model::Value* input, const model::Value* output) {
        if (input->uses().size() > 1 || input->has_initializer())
            return alloc<U>(output);

        assert(m_datamap.find(output) == m_datamap.end());
        auto datum = alloc<U>(input);
        m_datamap.emplace(output, datum);
        return datum;
    }

    template <typename U = T>
    static TensorT<U> deref(datum_ptr datum) {
        return datum->template get<U>();
    }

    template <typename U = T>
    static TensorT<U> deref(datum_ptr datum, datum_ptr src) {
        if (datum != src)
            datum->template resize<U>(src->shape());
        return datum->template get<U>();
    }

    template <typename U = T>
    static std::list<TensorT<U>> derefAll(datum_list list) {
        std::list<TensorT<U>> res;
        for (auto p : list)
            res.push_back(deref<U>(p));
        return res;
    }

private:
    void visitNode(model::Node* n) override {
        throw std::runtime_error(cxx::string_concat("Unsupported operator ", n->kind().str()));
    }

    struct IfOp : Operator {
        datum_ptr cond;
        datum_list outputs;

        op_list then_branch, else_branch;
        datum_list then_outputs, else_outputs;

        IfOp(OperatorFactory* of, model::If* n)
            : cond(of->alloc<bool>(n->input())),
              outputs(of->allocAll(n->outputs()))
        {
            for (auto x : n->then_branch()->nodes())
                then_branch.push_back(of->createOperator(x));
            for (auto v : n->then_branch()->outputs())
                then_outputs.push_back(of->alloc(v));

            for (auto x : n->else_branch()->nodes())
                else_branch.push_back(of->createOperator(x));
            for (auto v : n->else_branch()->outputs())
                else_outputs.push_back(of->alloc(v));
        }

        void evaluate() override {
            auto b = cond->template read<bool>();
            assert(b.rank() == 0);
            auto& branch = *b ? then_branch : else_branch;
            auto& results = *b ? then_outputs : else_outputs;

            for (auto& op : branch) {
                op->evaluate();
            }

            auto it = outputs.begin();
            for (auto v : results) {
                flat_copy(deref(v), deref(*it, v));
                ++it;
            }
        }
    };

    void visit(model::If* n) override {
        result = std::make_unique<IfOp>(this, n);
    }

    struct LoopOp : Operator {
        datum_ptr  max_trip_var, initial_cond_var;
        datum_list initial_state_vars;
        datum_list final_state_vars, final_scan_vars;

        op_list    body;
        datum_ptr  iteration_var, body_cond_var;
        datum_list body_state_vars;
        datum_ptr  body_output_cond_var;
        datum_list body_output_state_vars;
        datum_list body_output_scan_vars;

        LoopOp(OperatorFactory* of, model::Loop* n) {
            assert(n->inputs().size() >= 2);
            assert(n->inputs().size() == n->body()->inputs().size());
            assert(n->outputs().size() == n->body()->outputs().size() - 1);
            auto num_state_vars = n->inputs().size() - 2;

            for (auto x : n->body()->nodes()) {
                body.push_back(of->createOperator(x));
            }

            iteration_var = of->alloc(n->body()->input(0));
            body_cond_var = of->alloc<bool>(n->body()->input(1));
            for (size_t i = 2; i < n->body()->inputs().size(); ++i)
                body_state_vars.push_back(of->alloc(n->body()->input(i)));

            body_output_cond_var = of->alloc<bool>(n->body()->output(0));
            for (size_t i = 1; i < num_state_vars+1; ++i)
                body_output_state_vars.push_back(of->alloc(n->body()->output(i)));
            for (size_t i = num_state_vars+1; i < n->body()->outputs().size(); ++i)
                body_output_scan_vars.push_back(of->alloc(n->body()->output(i)));

            max_trip_var = of->alloc<int64_t>(n->input(0));
            initial_cond_var = of->alloc<bool>(n->input(1));
            for (size_t i = 2; i < n->inputs().size(); ++i)
                initial_state_vars.push_back(of->alloc(n->input(i)));
            for (size_t i = 0; i < num_state_vars; ++i)
                final_state_vars.push_back(of->allocInplace(n->body()->output(i+1), n->output(i)));
            for (size_t i = num_state_vars; i < n->outputs().size(); ++i)
                final_scan_vars.push_back(of->alloc(n->output(i)));
        }

        void evaluate() override {
            bool cond = (initial_cond_var == nullptr) ? true : *(initial_cond_var->template read<bool>());
            if (max_trip_var == nullptr) {
                cond = run_body(0, true, cond);
                for (int64_t i = 1; cond; ++i) {
                    cond = run_body(i, false, cond);
                }
            } else if (initial_cond_var == nullptr) {
                auto trip_count = *(max_trip_var->template read<int64_t>());
                for (int64_t i = 0; i < trip_count; ++i) {
                    run_body(i, i==0, true);
                }
            } else {
                auto trip_count = *(max_trip_var->template read<int64_t>());
                for (int64_t i = 0; i < trip_count && cond; ++i) {
                    cond = run_body(i, i==0, cond);
                }
            }

            // Copying loop-carried dependencies to enclosing scope
            for (size_t i = 0; i < final_state_vars.size(); ++i) {
                copy_data(body_output_state_vars[i], final_state_vars[i]);
            }
        }

    private:
        bool run_body(int64_t iteration, bool first, bool cond) {
            iteration_var->set(Scalar<T>(iteration));
            body_cond_var->set(Scalar<bool>(cond));

            // Copying loop-carried dependencies to loop body
            auto& state_vars = first ? initial_state_vars : body_output_state_vars;
            for (size_t i = 0; i < state_vars.size(); ++i) {
                copy_data(state_vars[i], body_state_vars[i]);
            }

            // Execute the loop body
            for (auto& op : body) {
                op->evaluate();
            }

            // Concatenating the value of the specified output value at the end
            // of each iteration of the loop. It is an error if the dimensions
            // or data type of these scan_outputs change across loop iterations.
            for (size_t i = 0; i < body_output_scan_vars.size(); ++i) {
                auto dims = body_output_scan_vars[i]->shape().extents();
                dims.insert(dims.begin(), iteration+1);
                final_scan_vars[i]->template resize<T>(Shape(dims));

                auto body_out = deref(body_output_scan_vars[i]);
                auto scan_out = deref(final_scan_vars[i]);
                reorder(body_out, scan_out[iteration]);
            }

            return *(body_output_cond_var->template read<bool>());
        }

        void copy_data(datum_ptr src, datum_ptr dst) {
            if (src != dst) {
                flat_copy(deref(src), deref(dst, src));
            }
        }
    };

    void visit(model::Loop* n) override {
        result = std::make_unique<LoopOp>(this, n);
    }

    struct ScanOp : Operator {
        int              num_state_vars;
        std::vector<int> input_axes, output_axes;
        std::vector<int> input_dirs, output_dirs;
        datum_list       inputs, outputs;
        op_list          body;
        datum_list       body_inputs, body_outputs;

        ScanOp(OperatorFactory* of, model::Scan* n) {
            auto num_inputs       = static_cast<int>(n->inputs().size());
            auto num_outputs      = static_cast<int>(n->outputs().size());
            auto num_scan_inputs  = static_cast<int>(n->num_scan_inputs());
            num_state_vars        = num_inputs - num_scan_inputs;
            auto num_scan_outputs = num_outputs - num_state_vars;

            assert(num_scan_inputs > 0 && num_state_vars >= 0 && num_scan_outputs >= 0);
            assert(n->body()->inputs().size() == num_inputs);
            assert(n->body()->outputs().size() == num_outputs);

            if (n->has_scan_input_axes()) {
                auto& axes = n->scan_input_axes();
                assert(axes.size() == num_scan_inputs);
                input_axes.assign(axes.begin(), axes.end());
            } else {
                input_axes.insert(input_axes.end(), num_scan_inputs, 0);
            }

            if (n->has_scan_output_axes()) {
                auto& axes = n->scan_output_axes();
                assert(axes.size() == num_scan_outputs);
                output_axes.assign(axes.begin(), axes.end());
            } else {
                output_axes.insert(output_axes.end(), num_scan_outputs, 0);
            }

            if (n->has_scan_input_directions()) {
                auto& dirs = n->scan_input_directions();
                assert(dirs.size() == num_scan_inputs);
                input_dirs.assign(dirs.begin(), dirs.end());
            } else {
                input_dirs.insert(input_dirs.end(), num_scan_inputs, 0);
            }

            if (n->has_scan_output_directions()) {
                auto& dirs = n->scan_output_directions();
                assert(dirs.size() == num_scan_outputs);
                output_dirs.assign(dirs.begin(), dirs.end());
            } else {
                output_dirs.insert(output_dirs.end(), num_scan_outputs, 0);
            }

            for (auto x : n->body()->nodes())
                body.push_back(of->createOperator(x));
            for (int i = 0; i < num_inputs; ++i)
                body_inputs.push_back(of->alloc(n->body()->input(i)));
            for (int i = 0; i < num_outputs; ++i)
                body_outputs.push_back(of->alloc(n->body()->output(i)));

            for (int i = 0; i < num_inputs; ++i)
                inputs.push_back(of->alloc(n->input(i)));
            for (int i = 0; i < num_outputs; ++i) {
                if (i < num_state_vars)
                    outputs.push_back(of->allocInplace(n->body()->output(i), n->output(i)));
                else
                    outputs.push_back(of->alloc(n->output(i)));
            }
        }

        void evaluate() override {
            auto sequence_len = get_sequence_length(0);

            for (int iteration = 0; iteration < sequence_len; ++iteration) {
                // Copy loop-carried dependencies
                for (int i = 0; i < num_state_vars; ++i) {
                    copy_data(iteration==0 ? inputs[i] : body_outputs[i], body_inputs[i]);
                }

                // Copy scan elements to scan body inputs
                for (int i = num_state_vars; i < inputs.size(); ++i) {
                    auto src  = deref(inputs[i]);
                    auto dst  = deref(body_inputs[i]);
                    auto axis = input_axes[i - num_state_vars];

                    assert(sequence_len == get_sequence_length(i - num_state_vars));
                    auto start = iteration;
                    if (input_dirs[i - num_state_vars] != 0)
                        start = sequence_len - iteration - 1;
                    reorder(src.slice({start}, {start}, {axis}, {1}), unsqueeze(dst, axis));
                }

                // Execute the scan body
                for (auto& op : body) {
                    op->evaluate();
                }

                // Concatenating the body scan elements to final scan outputs
                for (int i = num_state_vars; i < outputs.size(); ++i) {
                    auto src  = deref(body_outputs[i]);
                    auto dst  = deref(outputs[i]);
                    auto axis = output_axes[i - num_state_vars];

                    auto start = iteration;
                    if (output_dirs[i - num_state_vars] != 0)
                        start = sequence_len - iteration - 1;
                    reorder(unsqueeze(src, axis), dst.slice({start}, {start}, {axis}, {1}));
                }
            }

            // Copy loop-carried dependencies to enclosing scope
            for (int i = 0; i < num_state_vars; ++i) {
                copy_data(body_outputs[i], outputs[i]);
            }
        }

        int get_sequence_length(int i) {
            const auto& shape = inputs[i + num_state_vars]->shape();
            return static_cast<int>(shape.extent(input_axes[i]));
        }

        void copy_data(datum_ptr src, datum_ptr dst) {
            if (src != dst) {
                flat_copy(deref(src), deref(dst, src));
            }
        }
    };

    void visit(model::Scan* n) override {
        result = std::make_unique<ScanOp>(this, n);
    }

    struct ConstantOfShapeOp : Operator {
        DatumShape shape;
        datum_ptr output;
        T value{};

        ConstantOfShapeOp(OperatorFactory* of, model::ConstantOfShape* n)
            : shape(of, n->input()), output(of->alloc(n->output()))
        {
            if (n->has_value()) {
                auto t = n->value().decode<int64_t>();
                assert(t.size() == 1);
                value = *t;
            }
        }

        void evaluate() override {
            output->template resize<T>(*shape);
            deref(output).fill(value);
        }
    };

    void visit(model::ConstantOfShape* n) override {
        result = std::make_unique<ConstantOfShapeOp>(this, n);
    }

    struct EyeLikeOp : Operator {
        datum_ptr X, Y;
        int k;

        EyeLikeOp(OperatorFactory* of, model::EyeLike* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              k(n->get_i("k", 0)) {}

        void evaluate() override {
            auto out = deref(Y, X);
            out.fill(0);
            out.diagonal(k).fill(1);
        }
    };

    void visit(model::EyeLike* n) override {
        result = std::make_unique<EyeLikeOp>(this, n);
    }

    struct RandomNormalOp : Operator {
        datum_ptr Y;
        T mean, scale;
        RandomNormalOp(OperatorFactory* of, model::RandomNormal* n)
            : Y(of->alloc(n->output())), mean(n->mean()), scale(n->scale()) {}
        void evaluate() override {
            deref(Y).random(std::normal_distribution<T>(mean, scale));
        }
    };

    void visit(model::RandomNormal* n) override {
        result = std::make_unique<RandomNormalOp>(this, n);
    }

    struct RandomNormalLikeOp : Operator {
        datum_ptr X, Y;
        T mean, scale;
        RandomNormalLikeOp(OperatorFactory* of, model::RandomNormalLike* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              mean(n->mean()), scale(n->scale()) {}
        void evaluate() override {
            deref(Y, X).random(std::normal_distribution<T>(mean, scale));
        }
    };

    void visit(model::RandomNormalLike* n) override {
        result = std::make_unique<RandomNormalLikeOp>(this, n);
    }

    struct RandomUniformOp : Operator {
        datum_ptr Y;
        T low, high;
        RandomUniformOp(OperatorFactory* of, model::RandomUniform* n)
            : Y(of->alloc(n->output())), low(n->low()), high(n->high()) {}
        void evaluate() override {
            deref(Y).random(low, high);
        }
    };

    void visit(model::RandomUniform* n) override {
        result = std::make_unique<RandomUniformOp>(this, n);
    }

    struct RandomUniformLikeOp : Operator {
        datum_ptr X, Y;
        T low, high;
        RandomUniformLikeOp(OperatorFactory* of, model::RandomUniformLike* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              low(n->low()), high(n->high()) {}
        void evaluate() override {
            deref(Y, X).random(low, high);
        }
    };

    void visit(model::RandomUniformLike* n) override {
        result = std::make_unique<RandomUniformLikeOp>(this, n);
    }

    template <typename Fn>
    struct UnaryOp : Operator {
        datum_ptr X, Y;
        UnaryOp(OperatorFactory* of, model::Node* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            transformTo(deref(X), deref(Y, X), Fn{});
        }
    };

    #define DEFINE_UNARY_OPERATOR(Name, fn) \
    void visit(model::Name* n) override { \
        result = std::make_unique<UnaryOp<xfn::fn<T>>>(this, n); \
    }

    DEFINE_UNARY_OPERATOR(Abs, abs)
    DEFINE_UNARY_OPERATOR(Neg, negate)
    DEFINE_UNARY_OPERATOR(Sign, sign)
    DEFINE_UNARY_OPERATOR(Reciprocal, recip)
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
    DEFINE_UNARY_OPERATOR(Softsign, softsign);
    DEFINE_UNARY_OPERATOR(Softplus, softplus);
    #undef DEFINE_UNARY_OPERATOR

    struct ClipOp : Operator {
        xfn::clip<T> op; datum_ptr X, Y;
        ClipOp(OperatorFactory* of, model::Clip* n)
            : op(n->min(), n->max()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::Clip* n) override {
        result = std::make_unique<ClipOp>(this, n);
    }

    struct ShrinkOp : Operator {
        xfn::shrink<T> op; datum_ptr X, Y;
        ShrinkOp(OperatorFactory* of, model::Shrink* n)
            : op(n->lambd(), n->bias()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::Shrink* n) override {
        result = std::make_unique<ShrinkOp>(this, n);
    }

    struct ReluOp : Operator {
        datum_ptr X, Y;
        ReluOp(OperatorFactory* of, model::Relu* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), xfn::relu<T>());
        }
    };

    void visit(model::Relu* n) override {
        result = std::make_unique<ReluOp>(this, n);
    }

    struct PReluOp : Operator {
        datum_ptr X, slope, Y;
        PReluOp(OperatorFactory* of, model::PRelu* n)
            : X(of->alloc(n->input())),
              slope(of->alloc(n->slope())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            auto out = deref(Y);
            dlf::transformTo(deref(X), deref(slope), out, xfn::prelu<T>());
            Y->unget(out);
        }
    };

    void visit(model::PRelu* n) override {
        result = std::make_unique<PReluOp>(this, n);
    }

    struct LeakyReluOp : Operator {
        xfn::leaky_relu<T> op; datum_ptr X, Y;
        LeakyReluOp(OperatorFactory* of, model::LeakyRelu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::LeakyRelu* n) override {
        result = std::make_unique<LeakyReluOp>(this, n);
    }

    struct ThresholdedReluOp : Operator {
        xfn::thresholded_relu<T> op; datum_ptr X, Y;
        ThresholdedReluOp(OperatorFactory* of, model::ThresholdedRelu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::ThresholdedRelu* n) override {
        result = std::make_unique<ThresholdedReluOp>(this, n);
    }

    struct SeluOp : Operator {
        xfn::selu<T> op; datum_ptr X, Y;
        SeluOp(OperatorFactory* of, model::Selu* n)
            : op(n->alpha(), n->gamma()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::Selu* n) override {
        result = std::make_unique<SeluOp>(this, n);
    }

    struct EluOp : Operator {
        xfn::elu<T> op; datum_ptr X, Y;
        EluOp(OperatorFactory* of, model::Elu* n)
            : op(n->alpha()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::Elu* n) override {
        result = std::make_unique<EluOp>(this, n);
    }

    struct HardSigmoidOp : Operator {
        xfn::hard_sigmoid<T> op; datum_ptr X, Y;
        HardSigmoidOp(OperatorFactory* of, model::HardSigmoid* n)
            : op(n->alpha(), n->beta()),
              X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())) {}
        void evaluate() override {
            dlf::transformTo(deref(X), deref(Y, X), op);
        }
    };

    void visit(model::HardSigmoid* n) override {
        result = std::make_unique<HardSigmoidOp>(this, n);
    }

    struct SoftmaxOp : Operator {
        datum_ptr X, Y; int axis;
        SoftmaxOp(OperatorFactory* of, model::Softmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override {
            dnn::softmax(deref(X), deref(Y, X), axis);
        }
    };

    void visit(model::Softmax* n) override {
        result = std::make_unique<SoftmaxOp>(this, n);
    }

    struct LogSoftmaxOp : Operator {
        datum_ptr X, Y; int axis;
        LogSoftmaxOp(OperatorFactory* of, model::LogSoftmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override {
            dnn::logsoftmax(deref(X), deref(Y, X), axis);
        }
    };

    void visit(model::LogSoftmax* n) override {
        result = std::make_unique<LogSoftmaxOp>(this, n);
    }

    struct HardmaxOp : Operator {
        datum_ptr X, Y; int axis;
        HardmaxOp(OperatorFactory* of, model::Hardmax* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}
        void evaluate() override {
            dnn::hardmax(deref(X), deref(Y, X), axis);
        }
    };

    void visit(model::Hardmax* n) override {
        result = std::make_unique<HardmaxOp>(this, n);
    }

    template <typename Fn>
    struct BinaryOp : Operator {
        datum_ptr A, B, C;

        BinaryOp(OperatorFactory* of, model::Node* n)
            : A(of->alloc(n->input(0))), B(of->alloc(n->input(1)))
        {
            if (!A->shape().empty() && !B->shape().empty()) {
                auto shape = Shape::broadcast(A->shape(), B->shape());
                if (A->shape() == shape &&
                        n->input(0)->uses().size() == 1 &&
                       !n->input(0)->has_initializer()) {
                    C = of->allocInplace(n->input(0), n->output());
                } else if (B->shape() == shape &&
                        n->input(1)->uses().size() == 1 &&
                       !n->input(1)->has_initializer()) {
                    C = of->allocInplace(n->input(1), n->output());
                } else {
                    C = of->alloc(n->output());
                }
            } else {
                C = of->alloc(n->output());
            }
        }

        void evaluate() override {
            auto out = deref(C);
            transformTo(deref(A), deref(B), out, Fn{});
            C->unget(out);
        }
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
        datum_ptr A, B, C;
        RelationOp(OperatorFactory* of, model::Node* n)
            : A(of->alloc(n->input(0))), B(of->alloc(n->input(1))),
              C(of->alloc<bool>(n->output())) {}
        void evaluate() override {
            auto out = deref<bool>(C);
            transformTo(deref(A), deref(B), out, Fn{});
            C->unget(out);
        }
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
        datum_list inputs;
        datum_ptr output;

        AggregateOp(OperatorFactory* of, model::Node* n)
            : inputs(of->allocAll(n->inputs())),
              output(of->alloc(n->output())) {}

        void evaluate() override {
            if (inputs.size() == 0) {
                deref(output).fill(T{}); // fill with zero
                return;
            }

            if (inputs.size() == 1) {
                flat_copy(deref(inputs.front()), deref(output, inputs.front()));
                return;
            }

            std::vector<Shape> shapes;
            shapes.reserve(inputs.size());
            for (auto in : inputs)
                shapes.push_back(in->shape());
            output->template resize<T>(Shape::broadcast(shapes));

            auto out = deref(output);
            auto it = inputs.begin();
            auto a = *it++;
            auto b = *it++;
            transformTo(deref(a), deref(b), out, Fn{});
            while (it != inputs.end()) {
                transformTo(out, deref(*it), out, Fn{});
                ++it;
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
            auto out = deref(this->output);
            transformTo(out, count, out, xfn::divides<T>());
        }
    };

    void visit(model::Mean* n) override {
        result = std::make_unique<MeanOp>(this, n);
    }

    template <typename Reducer>
    struct ReductionOp : Operator {
        datum_ptr X, Y;
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
            auto out = deref(Y);
            dlf::reduce<Reducer>(deref(X), out, axes, keepdims);
            Y->unget(out);
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
        datum_ptr X, Y;
        int axis;
        bool keepdims;

        ArgMaxOp(OperatorFactory* of, model::ArgMax* n)
            : X(of->alloc(n->input())),
              Y(of->alloc<int>(n->output())),
              axis(n->axis()), keepdims(n->keepdims()) {}

        void evaluate() override {
            auto out = deref<int>(Y);
            argmax(deref(X), out, axis, keepdims);
            Y->unget(out);
        }
    };

    void visit(model::ArgMax* n) override {
        result = std::make_unique<ArgMaxOp>(this, n);
    }

    struct ArgMinOp : Operator {
        datum_ptr X, Y;
        int axis;
        bool keepdims;

        ArgMinOp(OperatorFactory* of, model::ArgMin* n)
            : X(of->alloc(n->input())),
              Y(of->alloc<int>(n->output())),
              axis(n->axis()), keepdims(n->keepdims()) {}

        void evaluate() override {
            auto out = deref<int>(Y);
            argmin(deref(X), out, axis, keepdims);
            Y->unget(out);
        }
    };

    void visit(model::ArgMin* n) override {
        result = std::make_unique<ArgMinOp>(this, n);
    }

    struct CumSumOp : Operator {
        datum_ptr X, Y, axis;
        bool exclusive, reverse;

        CumSumOp(OperatorFactory* of, model::CumSum* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(of->alloc<int>(n->axis())),
              exclusive(n->exclusive()),
              reverse(n->reverse()) {}

        void evaluate() override {
            auto a = axis->template read<int>();
            assert(a.rank() == 0);
            cumsum(deref(X), deref(Y, X), *a, exclusive, reverse);
        }
    };

    void visit(model::CumSum* n) override {
        result = std::make_unique<CumSumOp>(this, n);
    }

    struct GemmOp : Operator {
        T alpha, beta;
        cblas::Transpose transA, transB;
        datum_ptr A, B, C, Y;

        GemmOp(OperatorFactory* of, model::Gemm* n)
            : alpha(n->alpha()), beta(n->beta()),
              transA(n->transA() ? cblas::Transpose::Trans : cblas::Transpose::NoTrans),
              transB(n->transB() ? cblas::Transpose::Trans : cblas::Transpose::NoTrans),
              A(of->alloc(n->A())),
              B(of->alloc(n->B())),
              C(of->alloc(n->C())),
              Y(of->alloc(n->Y())) {}

        void evaluate() override {
            auto out = deref(Y);
            gemm(transA, transB, alpha, deref(A), deref(B), beta, deref(C), out);
            Y->unget(out);
        }
    };

    void visit(model::Gemm* n) override {
        result = std::make_unique<GemmOp>(this, n);
    }

    struct MatMulOp : Operator {
        datum_ptr A, B, C;
        MatMulOp(OperatorFactory* of, model::MatMul* n)
            : A(of->alloc(n->A())), B(of->alloc(n->B())), C(of->alloc(n->C())) {}
        void evaluate() override {
            auto out = deref(C);
            matmul(deref(A), deref(B), out);
            C->unget(out);
        }
    };

    void visit(model::MatMul* n) override {
        result = std::make_unique<MatMulOp>(this, n);
    }

    struct DetOp : Operator {
        datum_ptr X, Y;
        DetOp(OperatorFactory* of, model::Det* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())) {}
        void evaluate() override {
            auto out = deref(Y);
            det(deref(X), out);
            Y->unget(out);
        }
    };

    void visit(model::Det* n) override {
        result = std::make_unique<DetOp>(this, n);
    }

    struct ConvOp : Operator {
        datum_ptr X, W, B, Y;
        dnn::Filter2D filter;

        ConvOp(OperatorFactory* of, model::Conv* n)
            : X(of->alloc(n->X())),
              W(of->alloc(n->W())),
              B(of->alloc(n->B())),
              Y(of->alloc(n->Y())),
              filter(dnn::Filter2D(X->shape(), W->shape(), n->group())
                .pads(n->pads())
                .strides(n->strides())
                .dilations(n->dilations())) {}

        void evaluate() override {
            filter.set_shape(X->shape(), W->shape(), filter.group());
            Y->template resize<T>(filter.output_shape());

            auto out = deref(Y);
            dnn::conv2d(deref(X), deref(W), out, filter);
            if (B != nullptr) {
                transformChannel(out, deref(B), out, 1, xfn::plus<T>());
            }
        }
    };

    void visit(model::Conv* n) override {
        result = std::make_unique<ConvOp>(this, n);
    }

    struct MaxPoolOp : Operator {
        datum_ptr X, Y;
        dnn::Filter2D filter;

        MaxPoolOp(OperatorFactory* of, model::MaxPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X->shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())
                .dilations(n->dilations())) {}

        void evaluate() override {
            filter.set_shape(X->shape());
            Y->template resize<T>(filter.output_shape());

            auto out = deref(Y);
            dnn::max_pooling(deref(X), out, filter);
        }
    };

    void visit(model::MaxPool* n) override {
        result = std::make_unique<MaxPoolOp>(this, n);
    }

    struct AveragePoolOp : Operator {
        datum_ptr X, Y;
        dnn::Filter2D filter;
        const bool count_include_pad;

        AveragePoolOp(OperatorFactory* of, model::AveragePool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X->shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())),
              count_include_pad(n->count_include_pad()) {}

        void evaluate() override {
            filter.set_shape(X->shape());
            Y->template resize<T>(filter.output_shape());

            auto out = deref(Y);
            dnn::average_pooling(deref(X), out, filter, count_include_pad);
        }
    };

    void visit(model::AveragePool* n) override {
        result = std::make_unique<AveragePoolOp>(this, n);
    }

    struct LpPoolOp : Operator {
        datum_ptr X, Y;
        dnn::Filter2D filter;
        int p;

        LpPoolOp(OperatorFactory* of, model::LpPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              filter(dnn::Filter2D(X->shape(), n->kernel_shape()[0], n->kernel_shape()[1])
                .pads(n->pads())
                .strides(n->strides())),
              p(n->get_i("p", 2)) {}

        void evaluate() override {
            filter.set_shape(X->shape());
            Y->template resize<T>(filter.output_shape());

            auto out = deref(Y);
            dnn::lp_pooling(deref(X), out, filter, p);
        }
    };

    void visit(model::LpPool* n) override {
        result = std::make_unique<LpPoolOp>(this, n);
    }

    struct GlobalMaxPoolOp : Operator {
        datum_ptr X, Y;
        GlobalMaxPoolOp(OperatorFactory* of, model::GlobalMaxPool* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())) {}
        void evaluate() override {
            auto out = deref(Y);
            dnn::global_max_pooling(deref(X), out);
            Y->unget(out);
        }
    };

    void visit(model::GlobalMaxPool* n) override {
        result = std::make_unique<GlobalMaxPoolOp>(this, n);
    }

    struct GlobalAveragePoolOp : Operator {
        datum_ptr X, Y;
        GlobalAveragePoolOp(OperatorFactory* of, model::GlobalAveragePool* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())) {}
        void evaluate() override {
            auto out = deref(Y);
            dnn::global_average_pooling(deref(X), out);
            Y->unget(out);
        }
    };

    void visit(model::GlobalAveragePool* n) override {
        result = std::make_unique<GlobalAveragePoolOp>(this, n);
    }

    struct GlobalLpPoolOp : Operator {
        datum_ptr X, Y; int p;
        GlobalLpPoolOp(OperatorFactory* of, model::GlobalLpPool* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              p(n->get_i("p", 2)) {}
        void evaluate() override {
            auto out = deref(Y);
            dnn::global_lp_pooling(deref(X), out, p);
            Y->unget(out);
        }
    };

    void visit(model::GlobalLpPool* n) override {
        result = std::make_unique<GlobalLpPoolOp>(this, n);
    }

    struct BatchNormalizationOp : Operator {
        datum_ptr X, Y, S, B, M, V;
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
            auto out = deref(Y, X);
            dnn::batch_norm(deref(X), out, deref(S), deref(B), deref(M), deref(V), epsilon);
        }
    };

    void visit(model::BatchNormalization* n) override {
        result = std::make_unique<BatchNormalizationOp>(this, n);
    }

    struct LRNOp : Operator {
        datum_ptr X, Y; int n; T alpha, beta, bias;

        LRNOp(OperatorFactory* of, model::LRN* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              n(n->size()),
              alpha(n->alpha()), beta(n->beta()), bias(n->bias()) {}

        void evaluate() override {
            auto out = deref(Y, X);
            dnn::lrn(deref(X), out, n, alpha, beta, bias);
        }
    };

    void visit(model::LRN* n) override {
        result = std::make_unique<LRNOp>(this, n);
    }

    struct NonMaxSuppressionOp : Operator {
        datum_ptr           boxes, scores;
        datum_ptr           selected_indices;
        bool                center_point_box;
        DatumValue<int64_t> max_output_boxes;
        DatumValue<T>       iou_threshold;
        DatumValue<T>       score_threshold;

        NonMaxSuppressionOp(OperatorFactory* of, model::NonMaxSuppression* n)
            : boxes(of->alloc(n->boxes())), scores(of->alloc(n->scores())),
              selected_indices(of->alloc<int32_t>(n->output())),
              center_point_box(n->center_point_box()),
              max_output_boxes(of, n->max_output_boxes_per_class()),
              iou_threshold(of, n->iou_threshold()),
              score_threshold(of, n->score_threshold())
        {}

        void evaluate() override {
            auto out = deref<int32_t>(selected_indices);
            dnn::nms(deref(boxes), deref(scores), out,
                     center_point_box,
                     *max_output_boxes,
                     *iou_threshold,
                     *score_threshold);
            selected_indices->unget(out);
        }
    };

    void visit(model::NonMaxSuppression* n) override {
        result = std::make_unique<NonMaxSuppressionOp>(this, n);
    }

    struct TopKOp : Operator {
        datum_ptr X, Y, I;
        DatumValue<int64_t> K;
        int axis;
        bool largest, sorted;

        TopKOp(OperatorFactory* of, model::TopK* n)
            : X(of->alloc(n->X())), Y(of->alloc(n->Y())),
              I(of->alloc<int32_t>(n->indices())),
              K(of, n->K()), axis(n->axis()),
              largest(n->largest()), sorted(n->sorted())
        {}

        void evaluate() override {
            auto values = deref(Y);
            auto indices = deref<int32_t>(I);
            top_k(deref(X), values, indices, *K, axis, largest, sorted);
            Y->unget(values);
            I->unget(indices);
        }
    };

    void visit(model::TopK* n) override {
        result = std::make_unique<TopKOp>(this, n);
    }

    struct ReshapeOp : Operator {
        datum_ptr X, Y;
        DatumValues<int, int64_t> shape;

        ReshapeOp(OperatorFactory* of, model::Reshape* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              shape(of, n->shape()) {}

        void evaluate() override {
            auto new_shape = X->shape().reshape(*shape);
            if (X == Y) {
                X->template resize<T>(std::move(new_shape));
            } else {
                Y->template resize<T>(std::move(new_shape));
                flat_copy(deref(X), deref(Y));
            }
        }
    };

    void visit(model::Reshape* n) override {
        result = std::make_unique<ReshapeOp>(this, n);
    }

    struct ShapeOp : Operator {
        datum_ptr X, Y;
        ShapeOp(OperatorFactory* of, model::Shape* n)
            : X(of->alloc(n->input())), Y(of->alloc<int64_t>(n->output())) {}
        void evaluate() override {
            auto dims = X->shape().extents();
            Y->set(Tensor<int64_t>({dims.size()}, dims.begin(), dims.end()));
        }
    };

    void visit(model::Shape* n) override {
        result = std::make_unique<ShapeOp>(this, n);
    }

    struct SizeOp : Operator {
        datum_ptr X, Y;
        SizeOp(OperatorFactory* of, model::Size* n)
            : X(of->alloc(n->input())), Y(of->alloc<int64_t>(n->output())) {}
        void evaluate() override {
            Y->set(Scalar<int64_t>(X->shape().size()));
        }
    };

    void visit(model::Size* n) override {
        result = std::make_unique<SizeOp>(this, n);
    }

    struct FlattenOp : Operator {
        datum_ptr X, Y;
        int axis;

        FlattenOp(OperatorFactory* of, model::Flatten* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              axis(n->axis()) {}

        void evaluate() override {
            auto shape = X->shape().flatten(axis);
            if (X == Y) {
                X->template resize<T>(std::move(shape));
            } else {
                Y->template resize<T>(std::move(shape));
                flat_copy(deref(X), deref(Y));
            }
        }
    };

    void visit(model::Flatten* n) override {
        result = std::make_unique<FlattenOp>(this, n);
    }

    struct SqueezeOp : Operator {
        datum_ptr X, Y;
        std::vector<int> axes;

        SqueezeOp(OperatorFactory* of, model::Squeeze* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output()))
        {
            auto temp = n->axes();
            axes.assign(temp.begin(), temp.end());
        }

        void evaluate() override {
            auto shape = X->shape().squeeze(axes);
            if (X == Y) {
                X->template resize<T>(std::move(shape));
            } else {
                Y->template resize<T>(std::move(shape));
                flat_copy(deref(X), deref(Y));
            }
        }
    };

    void visit(model::Squeeze* n) override {
        result = std::make_unique<SqueezeOp>(this, n);
    }

    struct UnsqueezeOp : Operator {
        datum_ptr X, Y;
        std::vector<int> axes;

        UnsqueezeOp(OperatorFactory* of, model::Unsqueeze* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output()))
        {
            auto temp = n->axes();
            axes.assign(temp.begin(), temp.end());
        }

        void evaluate() override {
            auto shape = X->shape().unsqueeze(axes);
            if (X == Y) {
                X->template resize<T>(std::move(shape));
            } else {
                Y->template resize<T>(std::move(shape));
                flat_copy(deref(X), deref(Y));
            }
        }
    };

    void visit(model::Unsqueeze* n) override {
        result = std::make_unique<UnsqueezeOp>(this, n);
    }

    struct ConcatOp : Operator {
        datum_list inputs;
        datum_ptr output;
        int axis;

        ConcatOp(OperatorFactory* of, model::Concat* n)
            : inputs(of->allocAll(n->inputs())),
              output(of->alloc(n->output())),
              axis(n->axis()) {}

        void evaluate() override {
            auto temp = derefAll(inputs);
            auto out = deref(output);
            concat(axis, temp.begin(), temp.end(), out);
            output->unget(out);
        }
    };

    void visit(model::Concat* n) override {
        result = std::make_unique<ConcatOp>(this, n);
    }

    struct SplitOp : Operator {
        datum_ptr input;
        datum_list outputs;
        int axis;

        SplitOp(OperatorFactory* of, model::Split* n)
            : input(of->alloc(n->input())),
              outputs(of->allocAll(n->outputs())),
              axis(n->axis()) {}

        void evaluate() override {
            auto temp = derefAll(outputs);
            split(deref(input), axis, temp.begin(), temp.end());

            auto it = temp.begin();
            for (int i = 0; i < outputs.size(); ++i, ++it) {
                outputs[i]->unget(*it);
            }
        }
    };

    void visit(model::Split* n) override {
        result = std::make_unique<SplitOp>(this, n);
    }

    struct GatherOp : Operator {
        datum_ptr X, Y, indices;
        int axis;
        GatherOp(OperatorFactory* of, model::Gather* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())),
              axis(n->axis()) {}
        void evaluate() override {
            auto out = deref(Y);
            gather(deref(X), out, deref<int>(indices), axis);
            Y->unget(out);
        }
    };

    void visit(model::Gather* n) override {
        result = std::make_unique<GatherOp>(this, n);
    }

    // deprecated, same as ScatterElements
    struct ScatterOp : Operator {
        datum_ptr X, Y, indices, updates;
        int axis;
        ScatterOp(OperatorFactory* of, model::Scatter* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())),
              axis(n->axis()) {}
        void evaluate() override {
            auto out = deref(Y, X);
            flat_copy(deref(X), out);
            scatter_elements(out, deref<int>(indices), deref(updates), axis);
        }
    };

    void visit(model::Scatter* n) override {
        result = std::make_unique<ScatterOp>(this, n);
    }

    struct GatherElementsOp : Operator {
        datum_ptr X, Y, indices;
        int axis;
        GatherElementsOp(OperatorFactory* of, model::GatherElements* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())),
              axis(n->axis()) {}
        void evaluate() override {
            auto out = deref(Y);
            gather_elements(deref(X), out, deref<int>(indices), axis);
            Y->unget(out);
        }
    };

    void visit(model::GatherElements* n) override {
        result = std::make_unique<GatherElementsOp>(this, n);
    }

    struct ScatterElementsOp : Operator {
        datum_ptr X, Y, indices, updates;
        int axis;
        ScatterElementsOp(OperatorFactory* of, model::ScatterElements* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())),
              axis(n->axis()) {}
        void evaluate() override {
            auto out = deref(Y, X);
            flat_copy(deref(X), out);
            scatter_elements(out, deref<int>(indices), deref(updates), axis);
        }
    };

    void visit(model::ScatterElements* n) override {
        result = std::make_unique<ScatterElementsOp>(this, n);
    }

    struct GatherNDOp : Operator {
        datum_ptr X, Y, indices;
        GatherNDOp(OperatorFactory* of, model::GatherND* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              indices(of->alloc<int>(n->indices())) {}
        void evaluate() override {
            auto out = deref(Y);
            gather_nd(deref(X), out, deref<int>(indices));
            Y->unget(out);
        }
    };

    void visit(model::GatherND* n) override {
        result = std::make_unique<GatherNDOp>(this, n);
    }

    struct ScatterNDOp : Operator {
        datum_ptr X, Y, indices, updates;
        ScatterNDOp(OperatorFactory* of, model::ScatterND* n)
            : X(of->alloc(n->input())),
              Y(of->allocInplace(n->input(), n->output())),
              indices(of->alloc<int>(n->indices())),
              updates(of->alloc(n->updates())) {}
        void evaluate() override {
            auto out = deref(Y, X);
            flat_copy(deref(X), out);
            scatter_nd(out, deref<int>(indices), deref(updates));
        }
    };

    void visit(model::ScatterND* n) override {
        result = std::make_unique<ScatterNDOp>(this, n);
    }

    struct SliceOp : Operator {
        datum_ptr X, Y;
        DatumValues<int, int64_t> starts, ends, axes, steps;

        SliceOp(OperatorFactory* of, model::Slice* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              starts(of, n->starts()),
              ends(of, n->ends()),
              axes(of, n->axes()),
              steps(of, n->steps()) {}

        void evaluate() override {
            auto out = deref(Y);
            reorder(deref(X).slice(*starts, *ends, *axes, *steps), out);
            Y->unget(out);
        }
    };

    void visit(model::Slice* n) override {
        result = std::make_unique<SliceOp>(this, n);
    }

    struct TransposeOp : Operator {
        datum_ptr X, Y;
        std::vector<size_t> perm;

        TransposeOp(OperatorFactory* of, model::Transpose* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output()))
        {
            if (n->has_perm()) {
                auto& x_perm = n->perm();
                perm.assign(x_perm.begin(), x_perm.end());
            } else {
                perm.resize(X->shape().rank());
                std::iota(perm.begin(), perm.end(), 0);
                std::reverse(perm.begin(), perm.end());
            }
        }

        void evaluate() override {
            auto out = deref(Y);
            reorder(deref(X).transpose(perm), out);
            Y->unget(out);
        }
    };

    void visit(model::Transpose* n) override {
        result = std::make_unique<TransposeOp>(this, n);
    }

    struct ExpandOp : Operator {
        datum_ptr X, Y;
        DatumShape shape;

        ExpandOp(OperatorFactory* of, model::Expand* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              shape(of, n->shape()) {}

        void evaluate() override {
            auto exp_shape = Shape::broadcast(X->shape(), *shape);
            Y->template resize<T>(exp_shape);
            reorder(deref(X).broadcast_to(exp_shape), deref(Y));
        }
    };

    void visit(model::Expand* n) override {
        result = std::make_unique<ExpandOp>(this, n);
    }

    struct PadOp : Operator {
        datum_ptr X, Y;
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
            auto out = deref(Y);
            pad(deref(X), out, pads, mode, value);
            Y->unget(out);
        }
    };

    void visit(model::Pad* n) override {
        result = std::make_unique<PadOp>(this, n);
    }

    struct TileOp : Operator {
        datum_ptr X, Y, reps;

        TileOp(OperatorFactory* of, model::Tile* n)
            : X(of->alloc(n->input())),
              Y(of->alloc(n->output())),
              reps(of->alloc<int>(n->repeats())) {}

        void evaluate() override {
            auto out = deref(Y);
            auto v = reps->template read<int>();
            assert(v.rank() == 1);
            tile(deref(X), out, std::vector<size_t>(v.begin(), v.end()));
            Y->unget(out);
        }
    };

    void visit(model::Tile* n) override {
        result = std::make_unique<TileOp>(this, n);
    }

    struct ResizeOp : Operator {
        datum_ptr X, Y;
        DatumValues<float> scales;

        ResizeOp(OperatorFactory* of, model::Resize* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              scales(of, n->scales()) {}

        void evaluate() override {
            auto out = deref(Y);
            im::resize(deref(X), out, *scales);
            Y->unget(out);
        }
    };

    void visit(model::Resize* n) override {
        result = std::make_unique<ResizeOp>(this, n);
    }

    struct SpaceToDepthOp : Operator {
        datum_ptr X, Y;
        int blocksize;

        SpaceToDepthOp(OperatorFactory* of, model::SpaceToDepth* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              blocksize(n->blocksize()) {}

        void evaluate() override {
            auto out = deref(Y);
            dnn::space_to_depth(deref(X), out, blocksize);
            Y->unget(out);
        }
    };

    void visit(model::SpaceToDepth* n) override {
        result = std::make_unique<SpaceToDepthOp>(this, n);
    }

    struct DepthToSpaceOp : Operator {
        datum_ptr X, Y;
        int blocksize;
        std::string mode;

        DepthToSpaceOp(OperatorFactory* of, model::DepthToSpace* n)
            : X(of->alloc(n->input())), Y(of->alloc(n->output())),
              blocksize(n->blocksize()),
              mode(n->mode()) {}

        void evaluate() override {
            auto out = deref(Y);
            dnn::depth_to_space(deref(X), out, blocksize, mode);
            Y->unget(out);
        }
    };

    void visit(model::DepthToSpace* n) override {
        result = std::make_unique<DepthToSpaceOp>(this, n);
    }

    struct WhereOp : Operator {
        datum_ptr C, X, Y, Z;

        WhereOp(OperatorFactory* of, model::Where* n)
            : C(of->alloc<bool>(n->condition())),
              X(of->alloc(n->X())),
              Y(of->alloc(n->Y())),
              Z(of->alloc(n->Z())) {}

        void evaluate() override {
            auto out = deref(Z);
            where(deref<bool>(C), deref(X), deref(Y), out);
            Z->unget(out);
        }
    };

    void visit(model::Where* n) override {
        result = std::make_unique<WhereOp>(this, n);
    }

    struct OneHotOp : Operator {
        datum_ptr indices, values, output;
        DatumValue<int64_t> depth;
        int axis;

        OneHotOp(OperatorFactory* of, model::OneHot* n)
            : indices(of->alloc(n->indices())),
              values(of->alloc(n->values())),
              output(of->alloc(n->output())),
              depth(of, n->depth()),
              axis(static_cast<int>(n->axis())) {}

        void evaluate() override {
            auto out = deref(output);
            one_hot(deref(indices), deref(values), out, *depth, axis);
            output->unget(out);
        }
    };

    void visit(model::OneHot* n) override {
        result = std::make_unique<OneHotOp>(this, n);
    }

    struct NonZeroOp : Operator {
        datum_ptr X, Y;
        NonZeroOp(OperatorFactory* of, model::NonZero* n)
            : X(of->alloc(n->input())), Y(of->alloc<int32_t>(n->output())) {}
        void evaluate() override {
            auto out = deref<int32_t>(Y);
            nonzero(deref(X), out, true);
            Y->unget(out);
        }
    };

    void visit(model::NonZero* n) override {
        result = std::make_unique<NonZeroOp>(this, n);
    }
};

template <typename Context, typename T>
Predictor<Context, T>::Predictor(model::Graph& graph,
    const std::unordered_map<std::string, size_t>& env)
{
    model::ShapeInference::newInstance(env)->infer(graph);
    model::Optimizer::newInstance()->optimize(graph);

    OperatorFactory<Context, T> factory(m_dataset);
    for (auto n : graph.nodes())
        m_operators.push_back(factory.createOperator(n));
    for (auto v : graph.inputs())
        m_inputs.push_back(factory.alloc(v));
    for (auto v : graph.outputs())
        m_outputs.push_back(factory.alloc(v));
}

template <typename Context, typename T>
void Predictor<Context, T>::predict() {
    for (auto& op : m_operators) {
        op->evaluate();
    }
}

}} // namespace dlf::predict
