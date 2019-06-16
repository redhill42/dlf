#include "model.h"
#include "model/operators.h"

namespace dlf { namespace model {

#define fail_shape_inference(...) throw ShapeInferenceError(cxx::concat(__VA_ARGS__))

static void propagateShape(Node* n, size_t index = 0) {
    Value* X = n->input(index);
    if (X->type() != DataType::UNDEFINED) {
        Value* Y = n->output(0);
        Y->set_type(X->type());
        Y->set_dims(X->dims());
    }
}

static void convPoolShapeInference(Node* n, Value* X, Value* W, bool use_dilation, bool require_kernel_shape) {
    auto sym = n->kind().str();

    // we need the first input shape for this inference.
    if (X->type() == DataType::UNDEFINED)
        return;

    // if kernel shape is an input (and not attribute)
    // we need the shape of the second input.
    if (!require_kernel_shape && (W == nullptr || W->type() == DataType::UNDEFINED))
        return;

    auto input_shape = X->dims();
    if (input_shape.size() < 2) {
        fail_shape_inference(sym, ": Input tensor must have at least 2 dimensions");
    }

    // first dim is the batch axis and the next is the number of channels.
    size_t n_axes = input_shape.size() - 2;

    // Only MaxPool and Conv support dilation. For simplicity of the code,
    // we just treat the reset of them as having all-1s dilation.
    std::vector<int64_t> dilations;
    if (use_dilation && n->hasAttribute(kdilations)) {
        dilations = n->get_is(kdilations);
        if (dilations.size() != n_axes) {
            fail_shape_inference(sym, ": Attribute dilations has incorrect size");
        }
    } else {
        dilations.assign(n_axes, 1);
    }

    std::vector<int64_t> strides;
    if (n->hasAttribute(kstrides)) {
        strides = n->get_is(kstrides);
        if (strides.size() != n_axes) {
            fail_shape_inference(sym, ": Attribute strides has incorrect size");
        }
    } else {
        strides.assign(n_axes, 1);
    }

    std::vector<int64_t> kernel_shape;
    if (n->hasAttribute(kkernel_shape)) {
        kernel_shape = n->get_is(kkernel_shape);
        if (kernel_shape.size() != n_axes) {
            fail_shape_inference(sym, ": Attribute kernel_shape has incorrect size");
        }
    } else if (require_kernel_shape) {
        fail_shape_inference(sym, ": Attribute kernel_shape must be specified");
    } else {
        assert(W != nullptr);
        auto& weight_shape = W->dims();
        if (weight_shape.size() != input_shape.size())
            fail_shape_inference(sym, ": Input tensors must have same shape");
        for (size_t i = 2; i < weight_shape.size(); i++)
            kernel_shape.push_back(weight_shape[i]);
        n->set_is(kkernel_shape, kernel_shape);
    }

    // accounting for dilation, how big is the kernel in the dimension
    for (size_t i = 0; i < kernel_shape.size(); i++) {
        kernel_shape[i] = (kernel_shape[i] - 1) * dilations[i] + 1;
    }

    std::vector<int64_t> pads;
    if (n->hasAttribute(kpads)) {
        pads = n->get_is(kpads);
        if (pads.size() != n_axes*2) {
            fail_shape_inference(sym, ": Attribute pads has incorrect size");
        }
    } else {
        auto auto_pad_mode = n->get_s(kauto_pad, "NOTSET");
        if (auto_pad_mode == "NOTSET")
            fail_shape_inference(sym, ": No explicit padding provided");
        pads.assign(n_axes*2, 0);
        if (auto_pad_mode != "VALID") {
            for (size_t i = 0; i < n_axes; i++) {
                auto residual = input_shape[i+2] % strides[i];
                if (residual == 0)
                    residual = strides[i];
                auto total_pad = kernel_shape[i] - residual;
                if (total_pad < 0)
                    total_pad = 0;
                auto half_pad = total_pad >> 1;
                if (auto_pad_mode == "SAME_UPPER") {
                    pads[i] = half_pad;
                    pads[i + n_axes] = total_pad - half_pad;
                } else {
                    pads[i] = total_pad - half_pad;
                    pads[i + n_axes] = half_pad;
                }
            }
        }
        n->set_is(kpads, pads);
    }

    Dims output_shape;

    if (require_kernel_shape) {
        // add the first two dimensions from the input.
        output_shape.push_back(input_shape[0]);
        output_shape.push_back(input_shape[1]);
    } else {
        output_shape.push_back(input_shape[0]);
        output_shape.push_back(W->dims()[0]);
    }

    for (size_t i = 0; i < n_axes; i++) {
        // the input size, including padding
        int64_t input_size = input_shape[i+2];
        input_size += pads[i];
        input_size += pads[i + n_axes];

        // how many times we can move the kernel from it's initial position,
        // based on the stride
        int64_t output_size;

        // default is floor mode, i.e. ceil_mode is set to 0
        if (n->get_i(kceil_mode, 0) == 0)
            output_size = (input_size - kernel_shape[i]) / strides[i];
        else
            output_size = (input_size - kernel_shape[i] - 1) / strides[i] + 1;

        // add in the initial position
        output_shape.push_back(output_size + 1);
    }

    n->output(0)->set_type(n->input(0)->type());
    n->output(0)->set_dims(output_shape);

    if (n->outputs().size() > 1) {
        // MaxPool with two outputs case.
        n->output(1)->set_type(DataType::INT64);
        n->output(1)->set_dims(output_shape);
    }
}

static void globalPoolShapeInference(Node* n) {
    Value* X = n->input(0);
    Value* Y = n->output(0);

    if (X->type() == DataType::UNDEFINED)
        return;

    auto dims = X->dims();
    if (dims.size() < 2)
        return;
    std::fill(dims.begin()+2, dims.end(), 1);

    Y->set_type(X->type());
    Y->set_dims(dims);
}

class ShapeInferenceImpl final : public ShapeInference, DefaultVisitor {
public:
    void infer(Node* n) override {
        n->accept(*this);
    }

    void visit(Constant* n) override {
        if (!n->hasAttribute(kvalue) || n->attributeKind(kvalue) != AttributeKind::TENSOR)
            fail_shape_inference("Constant: Missing value attribute or invalid tensor value");
        n->output()->set_type(n->value().type());
        n->output()->set_dims(n->value().dims());
    }

    void visit(Conv* n) override {
        convPoolShapeInference(n, n->X(), n->W(), true, false);
    }

    void visit(ConvInteger* n) override {
        convPoolShapeInference(n, n->X(), n->W(), true, false);
        n->Y()->set_type(DataType::INT32);
    }

    void visit(AveragePool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, false, true);
    }

    void visit(MaxPool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, true, true);
    }

    void visit(LpPool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, false, true);
    }

    void visit(GlobalAveragePool* n) override {
        globalPoolShapeInference(n);
    }

    void visit(GlobalMaxPool* n) override {
        globalPoolShapeInference(n);
    }

    void visit(GlobalLpPool* n) override {
        globalPoolShapeInference(n);
    }

    void visit(Dropout* n) override {
        if (n->input()->type() == DataType::UNDEFINED)
            return;

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(n->input()->dims());
        if (n->mask() != nullptr) {
            n->mask()->set_type(DataType::BOOL);
            n->mask()->set_dims(n->input()->dims());
        }
    }

    void visit(Flatten* n) override {
        if (n->input()->type() == DataType::UNDEFINED)
            return;

        auto dims = n->input()->dims();
        size_t rank = dims.size();
        size_t axis = n->get_i(kaxis, 1);
        if (axis > rank)
            fail_shape_inference("Flatten: Invalid value (", axis, ") for attribute 'axis'");
        size_t a = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
        size_t b = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());

        n->output()->set_type(n->input()->type());
        n->output()->set_dims({a, b});
    }

    void visit(Reshape* n) override {
        if (n->data()->type() == DataType::UNDEFINED)
            return;
        if (n->shape() == nullptr || n->shape()->type() == DataType::UNDEFINED)
            return;
        if (!n->shape()->has_initializer())
            return;
        if (n->shape()->type() != DataType::INT64)
            fail_shape_inference("Reshape: Invalid shape");

        auto input_shape = n->data()->dims();
        auto total_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());
        auto shape = n->shape()->initializer().int64_data();
        auto new_size = shape.empty() ? size_t(0) : size_t(1);
        int  pending = -1;

        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == -1) {
                if (pending != -1)
                    fail_shape_inference("Reshape: Invalid shape");
                pending = i;
            } else {
                new_size *= shape[i];
            }
        }

        if (pending != -1) {
            shape[pending] = total_size / new_size;
            new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        }

        if (total_size != new_size)
            fail_shape_inference("Reshape: Invalid shape");

        Dims new_shape;
        for (auto d : shape) {
            new_shape.push_back(static_cast<size_t>(d));
        }

        n->reshaped()->set_type(n->data()->type());
        n->reshaped()->set_dims(new_shape);
    }

    void visit(Gemm* n) override {
        if (n->A()->type() == DataType::UNDEFINED)
            return;
        if (n->B()->type() == DataType::UNDEFINED)
            return;

        if (n->A()->dims().size() != 2 || n->B()->dims().size() != 2)
            fail_shape_inference("GEMM: Invalid input shape");

        size_t M = n->A()->dim(0);
        size_t K = n->A()->dim(1);
        size_t P = n->B()->dim(0);
        size_t N = n->B()->dim(1);

        if (n->get_i(ktransA, 0))
            std::swap(M, K);
        if (n->get_i(ktransB, 0))
            std::swap(P, N);
        if (K != P)
            fail_shape_inference("GEMM: Invalid input shape");

        n->Y()->set_type(n->A()->type());
        n->Y()->set_dims({M, N});
        n->C()->set_dims({M, N});
    }

    void visitNode(Node* n) override {
        propagateShape(n);
    }
};

ShapeInference& ShapeInference::Instance() {
    static ShapeInferenceImpl instance;
    return instance;
}

}} // namespace dlf:: model