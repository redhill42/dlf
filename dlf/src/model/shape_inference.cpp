#include "model.h"

namespace dlf { namespace model {

#define fail_shape_inference(...) throw ShapeInferenceError(cxx::string_concat(__VA_ARGS__))

inline bool hasInput(Value* input) {
    return input != nullptr && input->type() != DataType::UNDEFINED;
}

template <typename... Args>
inline bool hasInput(Value* input, Args... args) {
    return hasInput(input) && hasInput(args...);
}

void propagateShape(Node* n, size_t index = 0) {
    Value* X = n->input(index);
    if (hasInput(X)) {
        Value* Y = n->output(0);
        Y->set_type(X->type());
        Y->set_dims(X->dims());
    }
}

static Dims broadcastShape(const std::vector<Dims>& shapes) {
    // get the result shape size
    size_t result_shape_size = 0;
    for (auto& dim : shapes) {
        if (dim.size() > result_shape_size) {
            result_shape_size = dim.size();
        }
    }

    Dims result_shape;
    result_shape.reserve(result_shape_size);

    for (size_t i = 0; i < result_shape_size; i++) {
        size_t current_dim = 1;
        for (size_t j = 0; j < shapes.size(); j++) {
            if (i < result_shape_size - shapes[j].size()) {
                // shape j will be filled with 1 at dimension i
                continue;
            }

            auto dim_i_j = shapes[j][i - result_shape_size + shapes[j].size()];
            if (dim_i_j == 1)
                continue;
            if (current_dim != dim_i_j && current_dim != 1)
                fail_shape_inference("Incompatible dimensions");
            current_dim = dim_i_j;
        }
        result_shape.push_back(current_dim);
    }

    return result_shape;
}

static inline Dims broadcastShape(const Dims& shapeL, const Dims& shapeR) {
    return broadcastShape({shapeL, shapeR});
}

static std::vector<int64_t> decodeShape(Node* n, const TensorData& data, bool positive_only = true) {
    if (data.type() != DataType::INT64 || data.dims().size() != 1)
        fail_shape_inference(n->kind().str(), ": Invalid shape");

    std::vector<int64_t> shape;
    if (data.has_raw_data()) {
        auto raw = reinterpret_cast<const int64_t*>(data.raw_data().data());
        auto len = data.dims()[0];
        shape.assign(raw, raw+len);
    } else {
        shape = data.int64_data();
    }

    if (positive_only) {
        if (std::any_of(shape.begin(), shape.end(), [](auto x){ return x <= 0; })) {
            fail_shape_inference(n->kind().str(), ": Invalid shape");
        }
    }

    return shape;
}


class ShapeInferenceImpl final : public ShapeInference, DefaultVisitor {
public:
    void infer(Node* n) override {
        n->accept(*this);
    }

    void infer(Graph& g) override {
        for (auto n : g.nodes()) {
            infer(n);
        }
    }

    //-----------------------------------------------------------------------

    void visit(If* n) override {
        // there are no inputs so we just need to run the subgraph inferencing for
        // then/else subgraphs and apply those to the outputs.
        infer(*n->then_branch());
        infer(*n->else_branch());

        auto then_outputs = n->then_branch()->outputs();
        auto else_outputs = n->else_branch()->outputs();

        if (then_outputs.size() != else_outputs.size()) {
            fail_shape_inference("If: then_branch and else_branch produce different number of outputs.");
        }

        if (then_outputs.size() != n->outputs().size()) {
            fail_shape_inference(
                "If: The node has ", n->outputs().size(), " outputs ",
                "but subgraphs produce ", then_outputs.size(), " outputs");
        }

        for (size_t i = 0; i < then_outputs.size(); i++) {
            if (then_outputs[i]->type() == DataType::UNDEFINED || else_outputs[i]->type() == DataType::UNDEFINED)
                continue;
            if (then_outputs[i]->type() != else_outputs[i]->type())
                fail_shape_inference("If: Mismatched type for output", i);
            if (then_outputs[i]->dims() != else_outputs[i]->dims())
                fail_shape_inference("If: Mismatched shape for output", i);
            n->output(i)->set_type(then_outputs[i]->type());
            n->output(i)->set_dims(then_outputs[i]->dims());
        }
    }

    void visit(Loop* n) override {
        fail_shape_inference("Loop: Unsupported operator"); // FIXME
    }

    void visit(Scan* n) override {
        fail_shape_inference("Scan: Unsupported operator"); // FIXME
    }

    void visit(Where* n) override {
        if (!hasInput(n->condition(), n->X(), n->Y()))
            return;

        std::vector<Dims> shapes = {
            n->condition()->dims(),
            n->X()->dims(),
            n->Y()->dims()
        };

        n->output()->set_type(n->X()->type());
        n->output()->set_dims(broadcastShape(shapes));
    }

    //-----------------------------------------------------------------------

    void visit(Constant* n) override {
        if (!n->hasAttribute(kvalue) || n->attributeKind(kvalue) != AttributeKind::TENSOR)
            fail_shape_inference("Constant: Missing value attribute or invalid tensor value");
        n->output()->set_type(n->value().type());
        n->output()->set_dims(n->value().dims());
    }

    void visit(ConstantOfShape* n) override {
        if (!hasInput(n->input()) || !n->input()->has_initializer())
            return;

        auto shape = decodeShape(n, n->input()->initializer());
        n->output()->set_dims({shape.begin(), shape.end()});

        if (n->has_value()) {
            n->output()->set_type(n->value().type());
        } else {
            n->output()->set_type(DataType::FLOAT);
        }
    }

    void visit(EyeLike* n) override {
        if (!hasInput(n->input()))
            return;
        if (n->input()->dims().size() != 2)
            fail_shape_inference("EyeLike: Input tensor must be 2-dimensional");

        n->output()->set_type(n->has_dtype() ? n->dtype() : n->input()->type());
        n->output()->set_dims(n->input()->dims());
    }

    void visit(RandomNormal* n) override {
        n->output()->set_type(n->dtype());
        n->output()->set_dims(n->shape());
    }

    void visit(RandomNormalLike* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(n->has_dtype() ? n->dtype() : n->input()->type());
            n->output()->set_dims(n->input()->dims());
        }
    }

    void visit(RandomUniform* n) override {
        n->output()->set_type(n->dtype());
        n->output()->set_dims(n->shape());
    }

    void visit(RandomUniformLike* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(n->has_dtype() ? n->dtype() : n->input()->type());
            n->output()->set_dims(n->input()->dims());
        }
    }

    void visit(Multinomial* n) override {
        if (!hasInput(n->input()))
            return;

        DataType dtype;
        if (n->has_dtype()) {
            dtype = n->dtype();
            if (dtype != DataType::INT32 && dtype != DataType::INT64) {
                fail_shape_inference("Multinomial: Output type must be int32 or int64");
            }
        } else {
            dtype = DataType::INT32;
        }

        auto& input_shape = n->input()->dims();
        if (input_shape.size() != 2) {
            fail_shape_inference("Multinomial: Input tensor must have rank 2");
        }

        size_t batch_size = input_shape[0];
        size_t sample_size = static_cast<size_t>(n->get_i(ksample_size, 1));

        n->output()->set_type(dtype);
        n->output()->set_dims({batch_size, sample_size});
    }

    //-----------------------------------------------------------------------

    static void unaryLogicalShapeInference(Node* n) {
        if (hasInput(n->input())) {
            n->output()->set_type(DataType::BOOL);
            n->output()->set_dims(n->input()->dims());
        }
    }

    static void binaryLogicalShapeInference(Node* n) {
        if (hasInput(n->input(0), n->input(1))) {
            n->output()->set_type(DataType::BOOL);
            n->output()->set_dims(broadcastShape(n->input(0)->dims(), n->input(1)->dims()));
        }
    }

    void visit(And* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Or* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Xor* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Greater* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Less* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Equal* n) override {
        binaryLogicalShapeInference(n);
    }

    void visit(Not* n) override {
        unaryLogicalShapeInference(n);
    }

    void visit(BitShift* n) override {
        if (hasInput(n->input(0), n->input(1))) {
            n->output()->set_type(n->input(0)->type());
            n->output()->set_dims(broadcastShape(n->input(0)->dims(), n->input(1)->dims()));
        }
    }

    //-----------------------------------------------------------------------

    static void binaryMathShapeInference(Node* n) {
        if (hasInput(n->input(0), n->input(1))) {
            n->output()->set_type(n->input(0)->type());
            n->output()->set_dims(broadcastShape(n->input(0)->dims(), n->input(1)->dims()));
        }
    }

    void visit(Add* n) override {
        binaryMathShapeInference(n);
    }

    void visit(Sub* n) override {
        binaryMathShapeInference(n);
    }

    void visit(Mul* n) override {
        binaryMathShapeInference(n);
    }

    void visit(Div* n) override {
        binaryMathShapeInference(n);
    }

    void visit(Mod* n) override {
        binaryMathShapeInference(n);
    }

    void visit(Pow* n) override {
        binaryMathShapeInference(n);
    }

    void visit(IsNaN* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(DataType::BOOL);
            n->output()->set_dims(n->input()->dims());
        }
    }

    void visit(IsInf* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(DataType::BOOL);
            n->output()->set_dims(n->input()->dims());
        }
    }

    static void multiOpShapeInference(Node* n) {
        std::vector<Dims> shapes;
        for (size_t i = 0; i < n->inputs().size(); i++) {
            if (!hasInput(n->input(i)))
                return;
            shapes.push_back(n->input(i)->dims());
        }

        n->output()->set_type(n->input(0)->type());
        n->output()->set_dims(broadcastShape(shapes));
    }

    void visit(Max* n) override {
        multiOpShapeInference(n);
    }

    void visit(Min* n) override {
        multiOpShapeInference(n);
    }

    void visit(Sum* n) override {
        multiOpShapeInference(n);
    }

    void visit(Mean* n) override {
        multiOpShapeInference(n);
    }

    void visit(Gemm* n) override {
        if (!hasInput(n->A(), n->B()))
            return;

        if (n->A()->dims().size() != 2 || n->B()->dims().size() != 2)
            fail_shape_inference("GEMM: Invalid input shape");

        size_t M = n->A()->dim(0);
        size_t K = n->A()->dim(1);
        size_t P = n->B()->dim(0);
        size_t N = n->B()->dim(1);

        if (n->transA())
            std::swap(M, K);
        if (n->transB())
            std::swap(P, N);
        if (K != P)
            fail_shape_inference("GEMM: Invalid input shape");

        n->Y()->set_type(n->A()->type());
        n->Y()->set_dims({M, N});
    }

    static void matmulShapeInference(Node* n, Value* lhs, Value* rhs) {
        if (!hasInput(lhs) || !hasInput(rhs))
            return;

        auto shapeL = lhs->dims();
        auto shapeR = rhs->dims();

        if (shapeL.size() == 0 || shapeR.size() == 0)
            fail_shape_inference(n->kind().str(), ": Input tensors of wrong rank (0).");

        // First promote each shape to at least rank-2. This logic is
        // specific to matmul, not generic broadcasting.
        if (shapeL.size() == 1)
            shapeL.insert(shapeL.begin(), 1);
        if (shapeR.size() == 1)
            shapeR.push_back(1);

        // Check for compatible matrix multiply dimensions
        if (shapeL[shapeL.size() - 1] != shapeR[shapeR.size() - 2]) {
            fail_shape_inference(n->kind().str(), ": Incompatible dimensions for matrix multiplication");
        }

        // Now call out to generic multidimensional broadcasting for
        // the broadcastable prefixes.
        Dims prefixShapeL, prefixShapeR;
        for (int i = 0; i < shapeL.size() - 2; i++)
            prefixShapeL.push_back(shapeL[i]);
        for (int i = 0; i < shapeR.size() - 2; i++)
            prefixShapeR.push_back(shapeR[i]);
        Dims output_shape = broadcastShape(prefixShapeL, prefixShapeR);

        // Back to matmul-specific. Add the trailing dimensions back in.
        if (lhs->dims().size() != 1)
            output_shape.push_back(shapeL[shapeL.size() - 2]);
        if (rhs->dims().size() != 1)
            output_shape.push_back(shapeR[shapeR.size() - 1]);

        n->output()->set_type(lhs->type());
        n->output()->set_dims(output_shape);
    }

    void visit(MatMul* n) override {
        matmulShapeInference(n, n->A(), n->B());
    }

    void visit(QLinearMatMul* n) override {
        matmulShapeInference(n, n->A(), n->B());
    }

    void visit(MatMulInteger* n) override {
        matmulShapeInference(n, n->A(), n->B());
    }

    void visit(TopK* n) override {
        if (!hasInput(n->input()) || !hasInput(n->K()))
            return;
        if (!n->K()->has_initializer())
            return;
        if (n->K()->type() != DataType::INT64 || n->K()->dims().size() != 1 || n->K()->dim(0) != 1)
            fail_shape_inference("TopK: K input must be a one-dimensional tensor of size 1");

        auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();
        auto axis = n->axis();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            fail_shape_inference("TopK: Invalid value for attribute axis");

        auto k = n->K()->initializer().decode<int64_t>()(0);
        if (k > input_shape[axis])
            fail_shape_inference("TopK: Axis has less than the requested k elements");

        auto output_shape = input_shape;
        output_shape[axis] = k;

        n->output()->set_type(n->input()->type());
        n->indices()->set_type(DataType::INT64);
        n->output()->set_dims(output_shape);
        n->indices()->set_dims(output_shape);
    }

    //-----------------------------------------------------------------------

    static bool getConvPoolShapeInfo(
        Node* n, Value* X, Value* W, bool use_dilation, bool require_kernel_shape, bool use_auto_pad,
        std::vector<int64_t>& kernel_shape, std::vector<int64_t>& strides, std::vector<int64_t>& pads)
    {
        auto sym = n->kind().str();

        // we need the first input shape for this inference.
        if (!hasInput(X))
            return false;

        // if kernel shape is an input (and not attribute)
        // we need the shape of the second input.
        if (!require_kernel_shape && !hasInput(W))
            return false;

        const auto& input_shape = X->dims();
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
            n->set_is(kdilations, dilations);
        }

        if (n->hasAttribute(kstrides)) {
            strides = n->get_is(kstrides);
            if (strides.size() != n_axes) {
                fail_shape_inference(sym, ": Attribute strides has incorrect size");
            }
        } else {
            strides.assign(n_axes, 1);
            n->set_is(kstrides, strides);
        }

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

        if (n->hasAttribute(kpads)) {
            pads = n->get_is(kpads);
            if (pads.size() != n_axes*2) {
                fail_shape_inference(sym, ": Attribute pads has incorrect size");
            }
        } else {
            auto auto_pad_mode = n->get_s(kauto_pad, "VALID");
            if (use_auto_pad && auto_pad_mode == "NOTSET")
                fail_shape_inference(sym, ": No explicit padding provided");
            pads.assign(n_axes*2, 0);
            if (auto_pad_mode == "SAME_UPPER" || auto_pad_mode == "SAME_LOWER") {
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

        return true;
    }

    static void convPoolShapeInference(Node* n, Value* X, Value* W, bool use_dilation, bool require_kernel_shape) {
        auto sym = n->kind().str();

        std::vector<int64_t> kernel_shape, strides, pads;
        if (!getConvPoolShapeInfo(n, X, W, use_dilation, require_kernel_shape, true, kernel_shape, strides, pads))
            return;

        Dims input_shape = X->dims();
        auto n_axes = input_shape.size() - 2;
        Dims output_shape;

        if (require_kernel_shape) {
            // add the first two dimensions from the input.
            output_shape.push_back(input_shape[0]);
            output_shape.push_back(input_shape[1]);
        } else {
            output_shape.push_back(input_shape[0]);
            output_shape.push_back(W->dim(0));
        }

        for (size_t i = 0; i < n_axes; i++) {
            // the input size, including padding
            int64_t input_size = input_shape[i+2] + pads[i] + pads[i+n_axes];

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

    static void roiPoolShapeInference(Node* n) {
        if (!hasInput(n->input(0), n->input(1)))
            return;

        auto& input_shape = n->input(0)->dims();
        auto& roi_shape = n->input(1)->dims();

        if (input_shape.size() < 2)
            fail_shape_inference("RoiPool: Input tensor must have at least 2 dimensions");
        if (roi_shape.size() != 2)
            fail_shape_inference("RoiPool: RoIs tensor must have 2 dimensions");

        // first dim is the batch axis and the next is the number of channels.
        size_t n_axes = input_shape.size() - 2;

        std::vector<int64_t> pooled_shape;
        if (n->hasAttribute(kpooled_shape)) {
            pooled_shape = n->get_is(kpooled_shape);
            if (pooled_shape.size() != n_axes) {
                fail_shape_inference("RoiPool: Attribute pooled_shape has incorrect length");
            }
        } else {
            fail_shape_inference("RoiPool: Attribute pooled_shape must be specified");
        }

        Dims output_shape {
            roi_shape[0], input_shape[1],
            static_cast<size_t>(pooled_shape[0]),
            static_cast<size_t>(pooled_shape[1])
        };

        n->output()->set_type(n->input(0)->type());
        n->output()->set_dims(std::move(output_shape));

    }

    static void globalPoolShapeInference(Node* n) {
        if (!hasInput(n->input()))
            return;

        auto dims = n->input()->dims();
        if (dims.size() < 2)
            return;
        std::fill(dims.begin()+2, dims.end(), 1);

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(dims);
    }

    void visit(AveragePool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, false, true);
    }

    void visit(MaxPool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, true, true);
    }

    void visit(MaxUnpool* n) override {
        if (!hasInput(n->X(), n->I()))
            return;

        std::vector<int64_t> kernel_shape, strides, pads;
        if (!getConvPoolShapeInfo(n, n->X(), nullptr, false, true, false, kernel_shape, strides, pads))
            return;

        const auto& input_shape = n->X()->dims();
        size_t n_axes = input_shape.size() - 2;
        Dims output_shape;

        if (n->output_shape() != nullptr) {
            // If the third input, output_shape, is specified, then use that instead
            // of inferring shape from inputs.
            if (!n->output_shape()->has_initializer())
                // 'output_shape' is specified as input. Actual shape will be
                // determined at runtime
                return;

            auto shape_data = decodeShape(n, n->output_shape()->initializer());
            if (shape_data.size() != input_shape.size())
                fail_shape_inference("MaxUnpool: output_shape must have same rank as the shape of input tensor X");
            output_shape.assign(shape_data.begin(), shape_data.end());
        } else {
            output_shape.push_back(input_shape[0]); // the first dim is the batch size
            output_shape.push_back(n->I()->dim(1)); // channels should be the second dim of second input
            for (size_t i = 0; i < n_axes; i++) {
                auto new_dim = strides[i] * (input_shape[i+2] - 1);
                new_dim += kernel_shape[i];
                new_dim -= pads[i] + pads[i + n_axes];
                output_shape.push_back(static_cast<size_t>(new_dim));
            }
        }

        n->output()->set_type(n->X()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(MaxRoiPool* n) override {
        roiPoolShapeInference(n);
    }

    void visit(LpPool* n) override {
        convPoolShapeInference(n, n->input(), nullptr, false, true);
    }

    void visit(Conv* n) override {
        convPoolShapeInference(n, n->X(), n->W(), true, false);
    }

    void visit(ConvInteger* n) override {
        convPoolShapeInference(n, n->X(), n->W(), true, false);
        n->output()->set_type(DataType::INT32);
    }

    void visit(QLinearConv* n) override {
        convPoolShapeInference(n, n->X(), n->W(), true, false);
    }

    void visit(ConvTranspose* n) override {
        std::vector<int64_t> kernel_shape, strides, pads;
        if (!getConvPoolShapeInfo(n, n->X(), n->W(), true, false, true, kernel_shape, strides, pads)) {
            return;
        }

        const auto& input_shape = n->X()->dims();
        auto n_axes = input_shape.size() - 2;
        auto group = n->get_i(kgroup, 1);

        std::vector<int64_t> output_padding;
        if (n->has_output_padding()) {
            output_padding = n->output_padding();
            if (output_padding.size() != n_axes) {
                fail_shape_inference("ConvTranspose: Attribute output_padding has incorrect value");
            }
        } else {
            output_padding.assign(n_axes, 0);
        }

        Dims output_shape;

        output_shape.push_back(input_shape[0]);
        output_shape.push_back(n->W()->dim(1) * group); // channels should be the second dim of W

        if (n->has_output_shape()) {
            const auto& shape_data = n->output_shape();
            if (shape_data.size() != n_axes)
                fail_shape_inference("ConvTranspose: Attribute output_shape has incorrect value");
            for (size_t i = 0; i < n_axes; i++) {
                if (shape_data[i] < input_shape[i+2])
                    fail_shape_inference(
                        "ConvTranspose: output shape value cannot be smaller than the input shape value");
                output_shape.push_back(static_cast<size_t>(shape_data[i]));
            }
        } else {
            for (int i = 0; i < n_axes; i++) {
                size_t new_dim = strides[i] * (input_shape[i+2] - 1);
                new_dim += output_padding[i];
                new_dim += kernel_shape[i];
                new_dim -= pads[i] + pads[i + n_axes];
                output_shape.push_back(static_cast<size_t>(new_dim));
            }
        }

        n->output()->set_type(n->X()->type());
        n->output()->set_dims(std::move(output_shape));
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
        if (!hasInput(n->input()))
            return;

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(n->input()->dims());
        if (n->mask() != nullptr) {
            n->mask()->set_type(DataType::BOOL);
            n->mask()->set_dims(n->input()->dims());
        }
    }

    void visit(Flatten* n) override {
        if (!hasInput(n->input()))
            return;

        auto dims = n->input()->dims();
        auto rank = dims.size();
        auto axis = n->axis();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis > rank)
            fail_shape_inference("Flatten: Invalid value (", axis, ") for attribute 'axis'");
        size_t a = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
        size_t b = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());

        n->output()->set_type(n->input()->type());
        n->output()->set_dims({a, b});
    }

    void visit(TfIdfVectorizer* n) override {
        if (!hasInput(n->input()))
            return;

        if (!n->has_ngram_indexes())
            fail_shape_inference("TfIdfVectorizer: Missing 'ngram_indexes' attribute");
        const auto& ngram_indexes = n->ngram_indexes();
        if (std::any_of(ngram_indexes.begin(), ngram_indexes.end(), [](auto i){return i<0;}))
            fail_shape_inference("TfIdfVectorizer: ngram_indexes must have no negative values");
        auto greatest_hit = *std::max_element(ngram_indexes.begin(), ngram_indexes.end()) + 1;

        auto shape = n->input()->dims();
        if (shape.size() == 1 || shape.size() == 2)
            shape.back() = greatest_hit;
        else
            fail_shape_inference("TfIdfVectorizer: Input tensor must have rank 1 or 2");

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(shape);
    }

    void visit(StringNormalizer* n) override {
        // Cannot inference output shape
    }

    //-----------------------------------------------------------------------

    static void RNNShapeInference(Node* n) {
        if (!hasInput(n->input(0), n->input(1), n->input(2)))
            return;

        size_t seq_length, batch_size, input_size, num_directions, hidden_size;

        auto& input_shape = n->input(0)->dims();
        if (input_shape.size() != 3)
            fail_shape_inference("RNN: The input tensor must have rank 3");
        seq_length = input_shape[0];
        batch_size = input_shape[1];
        input_size = input_shape[2];

        auto direction = n->get_s(kdirection, "forward");
        if (direction == "forward" || direction == "reverse")
            num_directions = 1;
        else if (direction == "bidirectional")
            num_directions = 2;
        else
            fail_shape_inference("RNN: Attribute 'direction' has incorrect value");

        hidden_size = n->get_i(khidden_size, -1);
        if (hidden_size <= 0)
            fail_shape_inference("RNN: Attribute 'hidden_size' has incorrect value");

        auto& weight_shape = n->input(1)->dims();
        if (weight_shape.size() != 3)
            fail_shape_inference("RNN: The weight tensor must have rank 3");
        if (weight_shape[0] != num_directions || weight_shape[1] != hidden_size || weight_shape[2] != input_size)
            fail_shape_inference("RNN: The weight tensor has incorrect shape");

        auto& recur_shape = n->input(2)->dims();
        if (recur_shape.size() != 3)
            fail_shape_inference("RNN: The recurrence weight tensor must have rank 3");
        if (recur_shape[0] != num_directions || recur_shape[1] != hidden_size || recur_shape[2] != hidden_size)
            fail_shape_inference("RNN: The recurrence weight tensor has incorrect shape");

        if (n->outputs().size() > 0) { // Y
            n->output(0)->set_type(n->input()->type());
            n->output(0)->set_dims({seq_length, num_directions, batch_size, hidden_size});
        }
        if (n->outputs().size() > 1) { // Y_h
            n->output(1)->set_type(n->input()->type());
            n->output(1)->set_dims({num_directions, batch_size, hidden_size});
        }
        if (n->outputs().size() > 2) { // Y_c
            n->output(2)->set_type(n->input()->type());
            n->output(2)->set_dims({num_directions, batch_size, hidden_size});
        }
    }

    void visit(RNN* n) override {
        RNNShapeInference(n);
    }

    void visit(GRU* n) override {
        RNNShapeInference(n);
    }

    void visit(LSTM* n) override {
        RNNShapeInference(n);
    }

    //-----------------------------------------------------------------------

    static void reduceShapeInference(Node* n) {
        if (!hasInput(n->input()))
            return;

        auto keep_dims = n->get_i(kkeepdims, 1);
        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();

        auto axes = n->get_is(kaxes, {});
        for (auto& a : axes) {
            if (a < 0) a += rank;
            if (a < 0 || a >= rank)
                fail_shape_inference(n->kind().str(), ": The 'axes' attribute has incorrect value");
        }

        Dims output_shape;
        for (size_t i = 0; i < rank; i++) {
            // axes empty means reduce all dim
            if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
                output_shape.push_back(input_shape[i]);
            } else if (keep_dims) {
                output_shape.push_back(1);
            }
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    static void argReduceShapeInference(Node* n) {
        if (!hasInput(n->input()))
            return;

        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();
        auto keep_dims = n->get_i(kkeepdims, 1);
        auto axis = n->get_i(kaxis, 0);
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            fail_shape_inference(n->kind().str(), ": The 'axis' attribute has incorrect value");

        Dims output_shape;
        for (size_t i = 0; i < rank; i++) {
            if (i != axis) {
                output_shape.push_back(input_shape[i]);
            } else if (keep_dims) {
                output_shape.push_back(1);
            }
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(ReduceMax* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceMin* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceSum* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceSumSquare* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceMean* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceProd* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceLogSum* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceLogSumExp* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceL1* n) override {
        reduceShapeInference(n);
    }

    void visit(ReduceL2* n) override {
        reduceShapeInference(n);
    }

    void visit(ArgMax* n) override {
        argReduceShapeInference(n);
    }

    void visit(ArgMin* n) override {
        argReduceShapeInference(n);
    }

    //-----------------------------------------------------------------------

    void visit(Cast* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(n->to());
            n->output()->set_dims(n->input()->dims());
        }
    }

    void visit(Reshape* n) override {
        if (!hasInput(n->data(), n->shape()))
            return;
        if (!n->shape()->has_initializer())
            return;
        if (n->shape()->type() != DataType::INT64)
            fail_shape_inference("Reshape: Invalid shape");

        auto input_shape = n->data()->dims();
        auto total_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());
        auto shape = decodeShape(n, n->shape()->initializer(), false);
        auto new_size = shape.empty() ? size_t(0) : size_t(1);
        int  pending = -1;

        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] < 0) {
                if (shape[i] != -1 || pending != -1)
                    fail_shape_inference("Reshape: Invalid shape");
                pending = i;
            } else {
                if (shape[i] == 0) {
                    if (i >= input_shape.size())
                        fail_shape_inference("Reshape: Invalid shape");
                    shape[i] = input_shape[i];
                }
                new_size *= shape[i];
            }
        }

        if (pending != -1) {
            shape[pending] = total_size / new_size;
            new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        }

        if (total_size != new_size)
            fail_shape_inference("Reshape: Incompatible shape");

        Dims new_shape;
        for (auto d : shape) {
            new_shape.push_back(static_cast<size_t>(d));
        }

        n->reshaped()->set_type(n->data()->type());
        n->reshaped()->set_dims(new_shape);
    }

    void visit(Shape* n) override {
        if (hasInput(n->input())) {
            n->output()->set_type(DataType::INT64);
            n->output()->set_dims({n->input()->dims().size()});
        }
    }

    void visit(Size* n) override {
        n->output()->set_type(DataType::INT64);
        n->output()->set_dims({});
    }

    void visit(Concat* n) override {
        if (n->inputs().size() == 0)
            return;
        for (auto v : n->inputs())
            if (!hasInput(v)) return;
        if (!n->has_axis())
            fail_shape_inference("Concat: Missing required attribute 'axis'");

        auto rank = n->input(0)->dims().size();
        auto axis = n->axis();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            fail_shape_inference("Concat: Invalid axis attribute");

        Dims output_shape(rank);
        for (size_t i = 0; i < n->inputs().size(); i++) {
            const auto& shape = n->input(i)->dims();
            if (shape.size() != rank)
                fail_shape_inference("Concat: All inputs to concat must have same rank");
            if (n->input(i)->type() != n->input(0)->type())
                fail_shape_inference("Concat: All inputs to concat must have same type");
            for (size_t j = 0; j < rank; j++) {
                if (j == axis) {
                    output_shape[j] += shape[j];
                } else if (i == 0) {
                    output_shape[j] = shape[j];
                } else if (output_shape[j] != shape[j]) {
                    fail_shape_inference("Concat: Incompatible input tensor shape");
                }
            }
        }

        n->output()->set_type(n->input(0)->type());
        n->output()->set_dims(std::move(output_shape));
    }

    void visit(Split* n) override {
        if (!hasInput(n->input()))
            return;

        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();
        auto axis = n->axis();
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            fail_shape_inference("Split: Invalid axis attribute value");

        int split_dim = input_shape[axis];
        int n_split = n->outputs().size();
        std::vector<int64_t> split;

        if (n->has_split()) {
            split = n->split();
            if (split.size() != n_split)
                fail_shape_inference("Split: Invalid split attribute value");
            if (std::accumulate(split.begin(), split.end(), 0, std::plus<>()) != split_dim)
                fail_shape_inference("Split: Invalid split attribute value");
        } else {
            int chunk_size = split_dim / n_split;
            int left_over  = split_dim - chunk_size * n_split;
            for (int i = 0; i < n_split; i++) {
                split.push_back(i < left_over ? chunk_size+1 : chunk_size);
            }
        }

        for (int i = 0; i < n_split; i++) {
            auto shape = input_shape;
            shape[axis] = split[i];
            n->output(i)->set_type(n->input()->type());
            n->output(i)->set_dims(shape);
        }
    }

    void visit(Slice* n) override {
        if (n->inputs().size() < 3 || n->inputs().size() > 5)
            fail_shape_inference("Slice: Invalid number of input tensors");

        if (!hasInput(n->data()))
            return;

        // Shape inference if starts and ends are available and axes/steps are
        // either not set or set and has initializer.
        if (!n->starts()->has_initializer() || !n->ends()->has_initializer() ||
            (n->axes() != nullptr && !n->axes()->has_initializer()) ||
            (n->steps() != nullptr && !n->steps()->has_initializer()))
            return;

        auto starts = n->starts()->initializer().decode<int64_t>();
        auto ends   = n->ends()->initializer().decode<int64_t>();

        if (starts.rank() != 1 || ends.rank() != 1 || starts.size() != ends.size()) {
            fail_shape_inference("Slice: Incorrect or missing input value for starts and ends");
        }

        const auto& input_shape = n->input()->dims();
        auto input_rank = input_shape.size();

        Tensor<int64_t> axes({starts.size()});
        if (n->axes() == nullptr) {
            std::iota(axes.begin(), axes.end(), 0);
        } else {
            axes = n->axes()->initializer().decode<int64_t>();
            if (axes.rank() != 1 || axes.size() != starts.size()) {
                fail_shape_inference("Slice: Input axes has incorrect length");
            }
        }

        Tensor<int64_t> steps({starts.size()});
        if (n->steps() == nullptr) {
            std::fill(steps.begin(), steps.end(), 1);
        } else {
            steps = n->steps()->initializer().decode<int64_t>();
            if (steps.rank() != 1 || steps.size() != axes.size()) {
                fail_shape_inference("Slice: Input steps has incorrect length");
            }
        }

        Dims output_shape = input_shape;
        std::unordered_set<int64_t> unique_axes;

        for (size_t i = 0; i < axes.size(); ++i) {
            auto axis = axes(i);
            if (axis < 0) axis += input_rank;
            if (axis < 0 || axis >= input_rank)
                fail_shape_inference("Slice: Input axes has invalid data");
            if (unique_axes.find(axis) != unique_axes.end())
                fail_shape_inference("Slice: 'axes' has duplicates");
            unique_axes.insert(axis);

            int input_dim = static_cast<int>(input_shape[axis]);
            int step = static_cast<int>(steps(i));
            if (step == 0)
                fail_shape_inference("Slice: 'step' cannot be 0");

            int start = static_cast<int>(starts(i));
            if (start < 0)
                start += input_dim;
            if (step < 0)
                start = cxx::clamp(start, 0, input_dim - 1);
            else
                start = cxx::clamp(start, 0, input_dim);

            int end = static_cast<int>(ends(i));
            if (end < 0)
                end += input_dim;
            if (step < 0)
                end = cxx::clamp(end, -1, input_dim);
            else
                end = cxx::clamp(end, 0, input_dim);

            // find output dim value for this axis
            auto temp = (end - start - (step<0 ? -1 : 1)) / step + 1;
            if (temp < 0) temp = 0;
            output_shape[axis] = temp;
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    static bool validatePerm(size_t rank, const std::vector<int64_t>& perm) {
        if (perm.size() != rank)
            return false;

        std::unordered_set<int64_t> unique_index;
        for (auto index : perm) {
            if (!(0 <= index && index < rank))
                return false;
            if (unique_index.find(index) != unique_index.end())
                return false;
            unique_index.insert(index);
        }
        return true;
    }

    void visit(Transpose* n) override {
        if (!hasInput(n->input()))
            return;

        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();

        std::vector<int64_t> perm;
        if (n->has_perm()) {
            perm = n->perm();
            if (!validatePerm(rank, perm)) {
                fail_shape_inference("Transpose: Invalid perm attribute value");
            }
        } else {
            perm.resize(rank);
            std::iota(perm.begin(), perm.end(), 0);
            std::reverse(perm.begin(), perm.end());
        }

        Dims output_shape(rank);
        for (size_t i = 0; i < rank; i++) {
            output_shape[i] = input_shape[perm[i]];
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Gather* n) override {
        if (!hasInput(n->data(), n->indices()))
            return;

        const auto& data_shape = n->data()->dims();
        const auto& indices_shape = n->indices()->dims();
        int r = static_cast<int>(data_shape.size());
        int q = static_cast<int>(indices_shape.size());
        int axis = static_cast<int>(n->axis());

        if (r == 0)
            fail_shape_inference("Gather: data tensor must have rank >= 1");
        if (axis < 0) axis += r;
        if (axis < 0 || axis >= r)
            fail_shape_inference("Gather: axis must be in [-r, r-1]");

        int output_rank = q + r - 1;
        Dims output_shape;

        for (int i = 0; i < output_rank; i++) {
            size_t dim;
            if (i < axis)
                dim = data_shape[i];
            else if (i >= axis && i < axis + q)
                dim = indices_shape[i - axis];
            else
                dim = data_shape[i - q + 1];
            output_shape.push_back(dim);
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Squeeze* n) override {
        if (!hasInput(n->input()))
            return;

        const auto& input_shape = n->input()->dims();
        auto input_rank = input_shape.size();

        std::unordered_set<int64_t> axes;
        if (n->has_axes()) {
            for (auto a : n->axes()) {
                if (a < 0) a += input_rank;
                if (a < 0 || a >= input_rank)
                    fail_shape_inference("Squeeze: Invalid axis in 'axes' attribute");
                axes.insert(a); // duplicate is ok
            }
        }

        Dims output_shape;
        for (int i = 0; i < input_shape.size(); i++) {
            if (axes.find(i) != axes.end()) {
                if (input_shape[i] != 1)
                    fail_shape_inference("Squeeze: cannot select an axis to squeeze out which has size not equal to one");
                continue;
            } else if (axes.empty() && input_shape[i] == 1) {
                continue;
            } else {
                output_shape.push_back(input_shape[i]);
            }
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Unsqueeze* n) override {
        if (!hasInput(n->input()))
            return;
        if (!n->has_axes())
            fail_shape_inference("Unsqueeze: Missing required attribute 'axes'");

        const auto& input_shape = n->input()->dims();
        auto output_rank = input_shape.size() + n->axes().size();

        std::unordered_set<int64_t> axes;
        for (auto a : n->axes()) {
            if (a < 0) a += output_rank;
            if (a < 0 || a >= output_rank)
                fail_shape_inference("Unsqueeze: Invalid axis in 'axes' attribute");
            if (axes.find(a) != axes.end())
                fail_shape_inference("Unsqueeze: Duplicate axis value in 'axes' attribute");
            axes.insert(a);
        }

        Dims output_shape;
        for (size_t i = 0, j = 0; i < output_rank; i++) {
            if (axes.find(i) != axes.end()) {
                output_shape.push_back(1);
            } else {
                output_shape.push_back(input_shape[j++]);
            }
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Pad* n) override {
        if (!hasInput(n->input()))
            return;
        if (!n->has_pads())
            fail_shape_inference("Pad: Missing required attribute 'pads'");

        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();
        const auto& pads = n->pads();

        if (pads.size() != rank * 2) {
            fail_shape_inference("Pad: the 'pads' attribute has incorrect length");
        }

        Dims output_shape;
        for (size_t i = 0; i < rank; i++) {
            auto newdim = static_cast<int64_t>(input_shape[i]) + pads[i] + pads[i + rank];
            if (newdim < 0) newdim = 0;
            output_shape.push_back(static_cast<size_t>(newdim));
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(SpaceToDepth* n) override {
        if (!hasInput(n->input()))
            return;
        if (!n->has_blocksize())
            fail_shape_inference("SpaceToDepth: Missing 'blocksize' attribute");

        auto blocksize = n->blocksize();
        if (blocksize <= 0)
            fail_shape_inference("SpaceToDepth: Attribute 'blocksize' has incorrect value");

        auto input_shape = n->input()->dims();
        if (input_shape.size() != 4)
            fail_shape_inference("SpaceToDepth: Input tensor must be 4-dimensional");

        Dims output_shape = {
            input_shape[0],
            input_shape[1] * (blocksize * blocksize),
            input_shape[2] / blocksize,
            input_shape[3] / blocksize
        };

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(DepthToSpace* n) override {
        if (!hasInput(n->input()))
            return;
        if (!n->has_blocksize())
            fail_shape_inference("DepthToSpace: Missing 'blocksize' attribute");

        auto blocksize = n->blocksize();
        if (blocksize <= 0)
            fail_shape_inference("DepthToSpace: Attribute 'blocksize' has incorrect value");

        auto input_shape = n->input()->dims();
        if (input_shape.size() != 4)
            fail_shape_inference("DepthToSpace: Input tensor must be 4-dimensional");

        Dims output_shape = {
            input_shape[0],
            input_shape[1] / (blocksize * blocksize),
            input_shape[2] * blocksize,
            input_shape[3] * blocksize
        };

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Tile* n) override {
        if (!hasInput(n->input(), n->repeats()))
            return;
        if (!n->repeats()->has_initializer())
            return;
        if (n->repeats()->type() != DataType::INT64 || n->repeats()->dims().size() != 1)
            fail_shape_inference("Tile: The repeats tensor has incorrect value");

        const auto& input_shape = n->input()->dims();
        auto input_rank = input_shape.size();
        auto repeats = n->repeats()->initializer().decode<int64_t>();
        if (repeats.rank() != 1 || repeats.size() != input_rank)
            fail_shape_inference("Tile: 'Repeats' input has incorrect number of values");

        Dims output_shape;
        for (size_t i = 0; i < input_rank; i++) {
            output_shape.push_back(static_cast<size_t>(input_shape[i] * repeats(i)));
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Resize* n) override {
        if (!hasInput(n->input(), n->scales()))
            return;
        if (!n->scales()->has_initializer())
            return;

        const auto& input_shape = n->input()->dims();
        auto rank = input_shape.size();
        auto scales = n->scales()->initializer().decode<float>();

        if (scales.rank() != 1 || scales.size() != rank) {
            fail_shape_inference("Resize: Number of elements of input 'scales' must be same as rank of input");
        }

        Dims output_shape;
        for (size_t i = 0; i < rank; i++) {
            if (scales(i) <= 0)
                fail_shape_inference("Resize: Scale value must be greater than 0");
            auto dim = static_cast<size_t>(std::floor(input_shape[i] * scales(i)));
            output_shape.push_back(dim);
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Expand* n) override {
        if (!hasInput(n->input()) || !hasInput(n->shape()))
            return;
        if (!n->shape()->has_initializer())
            return;
        if (n->shape()->dims().size() != 1 || n->shape()->type() != DataType::INT64)
            fail_shape_inference("Expand: 'shape' input must be 1D tensor of type INT64");

        auto& input_shape = n->input()->dims();
        auto shape_data = decodeShape(n, n->shape()->initializer());
        Dims shape(shape_data.begin(), shape_data.end());
        auto output_shape = broadcastShape(input_shape, shape);

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(Compress* n) override {
        if (!hasInput(n->input(), n->condition()))
            return;
        if (!n->condition()->has_initializer())
            return;
        if (n->condition()->dims().size() != 1 || n->condition()->type() != DataType::BOOL)
            fail_shape_inference("Compress: 'condition' input must be 1D tensor of type BOOL");

        auto cond = n->condition()->initializer().decode<bool>();
        auto num_selected = std::count(cond.begin(), cond.end(), true);
        auto& input_shape = n->input()->dims();
        Dims output_shape;

        if (n->has_axis()) {
            int axis = static_cast<int>(n->axis());
            if (axis < 0) axis += input_shape.size();
            if (axis < 0 || axis >= input_shape.size())
                fail_shape_inference("Compress: The 'axis' attribute has incorrect value");
            output_shape = input_shape;
            if (num_selected < output_shape[axis])
                output_shape[axis] = num_selected;
        } else {
            auto total_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<>());
            if (num_selected > total_size)
                num_selected = total_size;
            output_shape.push_back(num_selected);
        }

        n->output()->set_type(n->input()->type());
        n->output()->set_dims(output_shape);
    }

    void visit(OneHot* n) override {
        fail_shape_inference("OneHot: Unsupported operation"); // FIXME
    }

    void visit(NonZero* n) override {
        // cannot inference output shape
    }

    //-----------------------------------------------------------------------

    void visitNode(Node* n) override {
        propagateShape(n);
    }
};

ShapeInference& ShapeInference::Instance() {
    static ShapeInferenceImpl instance;
    return instance;
}

}} // namespace dlf:: model
