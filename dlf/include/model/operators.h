#pragma once

namespace dlf { namespace model {

#define BEGIN_OPERATOR(name) \
class name : public Node { \
public: \
    static constexpr NodeKind Kind = k##name; \
    name(Graph* graph) : Node(graph, Kind) {} \
    void accept(Visitor& v) override { v.visit(this); }
#define END_OPERATOR() };

#define DEFINE_OPERATOR(name) BEGIN_OPERATOR(name) END_OPERATOR()

#define DEFINE_BINARY_OPERATOR(name) \
BEGIN_OPERATOR(name) \
    Value* A() { return input(0); } \
    Value* B() { return input(1); } \
    Value* C() { return output(); } \
END_OPERATOR()

#define DEFINE_FLOAT_ATTRIBUTE(name, def) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    float name() const \
        { return get_f(k##name, def); } \
    auto name(float value) \
        { set_f(k##name, value); return this; }

#define DEFINE_INT_ATTRIBUTE(name, def) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    int64_t name() const \
        { return get_i(k##name, def); } \
    auto name(int64_t value) \
        { set_i(k##name, value); return this; }

#define DEFINE_BOOL_ATTRIBUTE(name, def) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    bool name() const \
        { return !!get_i(k##name, static_cast<int64_t>(def)); } \
    auto name(bool value) \
        { set_i(k##name, static_cast<int64_t>(value)); return this; }

#define DEFINE_STRING_ATTRIBUTE(name, def) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    std::string name() const \
        { return get_s(k##name, def); } \
    auto name(std::string value) \
         { set_s(k##name, std::move(value)); return this; }

#define DEFINE_GRAPH_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    std::shared_ptr<Graph> name() const \
        { return get_g(k##name); } \
    auto name(std::shared_ptr<Graph> value) \
        { set_g(k##name, value); return this; }

#define DEFINE_TENSOR_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    const TensorData& name() const \
        { return get_t(k##name); } \
    auto name(TensorData value) \
        { set_t(k##name, std::move(value)); return this; }

#define DEFINE_INTS_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    const std::vector<int64_t>& name() const \
        { return get_is(k##name); } \
    auto name(std::vector<int64_t> value) \
        { set_is(k##name, std::move(value)); return this; }

#define DEFINE_FLOATS_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    const std::vector<float>& name() const \
        { return get_fs(k##name); } \
    auto name(std::vector<float> value) \
        { set_fs(k##name, std::move(value)); return this; }

#define DEFINE_STRINGS_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    const std::vector<std::string>& name() const \
        { return get_ss(k##name); } \
    auto name(std::vector<std::string> value) \
        { set_ss(k##name, std::move(value)); return this; }

#define DEFINE_DTYPE_ATTRIBUTE(name) \
    bool has_##name() const noexcept \
        { return hasAttribute(k##name); } \
    DataType name() const  \
        { return has_##name() ? static_cast<DataType>(get_i(k##name)) : DataType::FLOAT; } \
    auto name(DataType dt) \
        { set_i(k##name, static_cast<int64_t>(dt)); return this; }

#define DEFINE_SHAPE_ATTRIBUTE(name) \
    bool has_##name() const noexcept { return hasAttribute(k##name); } \
    std::vector<size_t> name() const { \
        const auto& shape__ = get_is(k##name); \
        return std::vector<size_t>(shape__.begin(), shape__.end()); \
    } \
    auto name(const std::vector<size_t>& dims__) { \
        set_is(k##name, std::vector<int64_t>(dims__.begin(), dims__.end())); \
        return this; \
    }

//==-------------------------------------------------------------------------
// Control flow operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(If)
    DEFINE_GRAPH_ATTRIBUTE(then_branch)
    DEFINE_GRAPH_ATTRIBUTE(else_branch)
END_OPERATOR()

BEGIN_OPERATOR(Loop)
    DEFINE_GRAPH_ATTRIBUTE(body)

    Value* M()    { return input(0); }
    Value* cond() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Scan)
    DEFINE_GRAPH_ATTRIBUTE(body)
    DEFINE_INT_ATTRIBUTE(num_scan_inputs, 0)
    DEFINE_INTS_ATTRIBUTE(scan_input_axes)
    DEFINE_INTS_ATTRIBUTE(scan_input_directions)
    DEFINE_INTS_ATTRIBUTE(scan_output_axes)
    DEFINE_INTS_ATTRIBUTE(scan_output_directions)
END_OPERATOR()

BEGIN_OPERATOR(Where)
    Value* condition() { return input(0); }
    Value* X() { return input(1); }
    Value* Y() { return input(2); }
    Value* Z() { return output(); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// Generator operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(Constant)
    DEFINE_TENSOR_ATTRIBUTE(value)
END_OPERATOR()

BEGIN_OPERATOR(ConstantOfShape)
    DEFINE_TENSOR_ATTRIBUTE(value)
END_OPERATOR()

BEGIN_OPERATOR(EyeLike)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(RandomNormal)
    DEFINE_FLOAT_ATTRIBUTE(mean, 0.f)
    DEFINE_FLOAT_ATTRIBUTE(scale, 1.f)
    DEFINE_FLOAT_ATTRIBUTE(seed, 0.f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_SHAPE_ATTRIBUTE(shape)
END_OPERATOR()

BEGIN_OPERATOR(RandomNormalLike)
    DEFINE_FLOAT_ATTRIBUTE(mean, 0.f)
    DEFINE_FLOAT_ATTRIBUTE(scale, 1.f)
    DEFINE_FLOAT_ATTRIBUTE(seed, 0.f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(RandomUniform)
    DEFINE_FLOAT_ATTRIBUTE(high, 1.f)
    DEFINE_FLOAT_ATTRIBUTE(low,  0.f)
    DEFINE_FLOAT_ATTRIBUTE(seed, 0.f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_SHAPE_ATTRIBUTE(shape)
END_OPERATOR()

BEGIN_OPERATOR(RandomUniformLike)
    DEFINE_FLOAT_ATTRIBUTE(high, 1.f)
    DEFINE_FLOAT_ATTRIBUTE(low, 0.f)
    DEFINE_FLOAT_ATTRIBUTE(seed, 0.f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(Multinomial)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_INT_ATTRIBUTE(sample_size, 1)
    DEFINE_FLOAT_ATTRIBUTE(seed, 0.f)
END_OPERATOR()

//==-------------------------------------------------------------------------
// Logical operators
//==-------------------------------------------------------------------------

DEFINE_BINARY_OPERATOR(And)
DEFINE_BINARY_OPERATOR(Or)
DEFINE_BINARY_OPERATOR(Xor)
DEFINE_BINARY_OPERATOR(Greater)
DEFINE_BINARY_OPERATOR(Less)
DEFINE_BINARY_OPERATOR(Equal)
DEFINE_OPERATOR(Not)

BEGIN_OPERATOR(BitShift)
    DEFINE_STRING_ATTRIBUTE(direction, "")
END_OPERATOR()

//==-------------------------------------------------------------------------
// Math operators
//==-------------------------------------------------------------------------

DEFINE_BINARY_OPERATOR(Add)
DEFINE_BINARY_OPERATOR(Sub)
DEFINE_BINARY_OPERATOR(Mul)
DEFINE_BINARY_OPERATOR(Div)
DEFINE_BINARY_OPERATOR(Pow)

BEGIN_OPERATOR(Mod)
    DEFINE_BOOL_ATTRIBUTE(fmod, false)

    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return output(); }
END_OPERATOR()

DEFINE_OPERATOR(Sign)
DEFINE_OPERATOR(Neg)
DEFINE_OPERATOR(Abs)
DEFINE_OPERATOR(Reciprocal)
DEFINE_OPERATOR(Floor)
DEFINE_OPERATOR(Ceil)
DEFINE_OPERATOR(Round)
DEFINE_OPERATOR(Sqrt)
DEFINE_OPERATOR(Exp)
DEFINE_OPERATOR(Log)
DEFINE_OPERATOR(Sin)
DEFINE_OPERATOR(Cos)
DEFINE_OPERATOR(Tan)
DEFINE_OPERATOR(Asin)
DEFINE_OPERATOR(Acos)
DEFINE_OPERATOR(Atan)
DEFINE_OPERATOR(Sinh)
DEFINE_OPERATOR(Cosh)
DEFINE_OPERATOR(Tanh)
DEFINE_OPERATOR(Asinh)
DEFINE_OPERATOR(Acosh)
DEFINE_OPERATOR(Atanh)
DEFINE_OPERATOR(Erf)

DEFINE_OPERATOR(IsNaN)

BEGIN_OPERATOR(IsInf)
    DEFINE_BOOL_ATTRIBUTE(detect_positive, true)
    DEFINE_BOOL_ATTRIBUTE(detect_negative, true)
END_OPERATOR()

DEFINE_OPERATOR(Max)
DEFINE_OPERATOR(Min)
DEFINE_OPERATOR(Sum)
DEFINE_OPERATOR(Mean)

BEGIN_OPERATOR(Clip)
    DEFINE_FLOAT_ATTRIBUTE(min, std::numeric_limits<float>::lowest())
    DEFINE_FLOAT_ATTRIBUTE(max, std::numeric_limits<float>::max())
END_OPERATOR()

BEGIN_OPERATOR(Shrink)
    DEFINE_FLOAT_ATTRIBUTE(bias, 0.f)
    DEFINE_FLOAT_ATTRIBUTE(lambd, 0.5f)
END_OPERATOR()

DEFINE_OPERATOR(Sigmoid)

DEFINE_OPERATOR(Relu)

BEGIN_OPERATOR(PRelu)
    Value* slope() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(LeakyRelu)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 0.01f)
END_OPERATOR()

BEGIN_OPERATOR(ThresholdedRelu)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 1.0f)
END_OPERATOR()

BEGIN_OPERATOR(Selu)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 1.67326319217681884765625f)
    DEFINE_FLOAT_ATTRIBUTE(gamma, 1.05070102214813232421875f)
END_OPERATOR()

BEGIN_OPERATOR(Elu)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 1.0f)
END_OPERATOR()

BEGIN_OPERATOR(HardSigmoid)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 0.2f)
    DEFINE_FLOAT_ATTRIBUTE(beta, 0.5f)
END_OPERATOR()

BEGIN_OPERATOR(Softmax)
    DEFINE_FLOAT_ATTRIBUTE(axis, 1)
END_OPERATOR()

BEGIN_OPERATOR(LogSoftmax)
    DEFINE_FLOAT_ATTRIBUTE(axis, 1)
END_OPERATOR()

BEGIN_OPERATOR(Hardmax)
    DEFINE_FLOAT_ATTRIBUTE(axis, 1)
END_OPERATOR()

DEFINE_OPERATOR(Softsign)
DEFINE_OPERATOR(Softplus)

BEGIN_OPERATOR(Gemm)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 1.f)
    DEFINE_FLOAT_ATTRIBUTE(beta, 1.f)
    DEFINE_BOOL_ATTRIBUTE(transA, false)
    DEFINE_BOOL_ATTRIBUTE(transB, false)

    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

DEFINE_BINARY_OPERATOR(MatMul)

BEGIN_OPERATOR(MatMulInteger)
    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* A_zero_point() { return input(2); }
    Value* B_zero_point() { return input(3); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(QLinearMatMul)
    Value* A()            { return input(0); }
    Value* A_scale()      { return input(1); }
    Value* A_zero_point() { return input(2); }
    Value* B()            { return input(3); }
    Value* B_scale()      { return input(4); }
    Value* B_zero_point() { return input(5); }
    Value* Y_scale()      { return input(6); }
    Value* Y_zero_point() { return input(7); }
    Value* Y()            { return output(); }
END_OPERATOR()

DEFINE_OPERATOR(Det)

BEGIN_OPERATOR(TopK)
    DEFINE_INT_ATTRIBUTE(axis, -1)
    DEFINE_BOOL_ATTRIBUTE(largest, true)
    DEFINE_BOOL_ATTRIBUTE(sorted, true)

    Value* X() { return input(0); }
    Value* K() { return input(1); }
    Value* Y() { return output(0); }
    Value* indices() { return output(1); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// CNN operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(AveragePool)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_BOOL_ATTRIBUTE(ceil_mode, false)
    DEFINE_BOOL_ATTRIBUTE(count_include_pad, false)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)
END_OPERATOR()

BEGIN_OPERATOR(MaxPool)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_BOOL_ATTRIBUTE(ceil_mode, false)
    DEFINE_INTS_ATTRIBUTE(dilations)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INT_ATTRIBUTE(storage_order, 0)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* indices() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(MaxUnpool)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* X() { return input(0); }
    Value* I() { return input(1); }
    Value* output_shape() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(MaxRoiPool)
    DEFINE_INTS_ATTRIBUTE(pooled_shape)
    DEFINE_FLOAT_ATTRIBUTE(spatial_scale, 1.f)

    Value* X()    { return input(0); }
    Value* rois() { return input(1); }
    Value* Y()    { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(LpPool)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)
END_OPERATOR()

BEGIN_OPERATOR(Conv)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_INTS_ATTRIBUTE(dilations)
    DEFINE_INT_ATTRIBUTE(group, 1)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(ConvInteger)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_INTS_ATTRIBUTE(dilations)
    DEFINE_INT_ATTRIBUTE(group, 1)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* X_zero_point() { return input(2); }
    Value* W_zero_point() { return input(3); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(QLinearConv)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_INTS_ATTRIBUTE(dilations)
    DEFINE_INT_ATTRIBUTE(group, 1)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* X()            { return input(0); }
    Value* X_scale()      { return input(1); }
    Value* X_zero_point() { return input(2); }
    Value* W()            { return input(3); }
    Value* W_scale()      { return input(4); }
    Value* W_zero_point() { return input(5); }
    Value* Y_scale()      { return input(6); }
    Value* Y_zero_point() { return input(7); }
    Value* B()            { return input(8); }
    Value* Y()            { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(ConvTranspose)
    DEFINE_STRING_ATTRIBUTE(auto_pad, "NOTSET")
    DEFINE_INTS_ATTRIBUTE(dilations)
    DEFINE_INT_ATTRIBUTE(group, 1)
    DEFINE_INTS_ATTRIBUTE(kernel_shape)
    DEFINE_INTS_ATTRIBUTE(output_padding)
    DEFINE_INTS_ATTRIBUTE(output_shape)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_INTS_ATTRIBUTE(strides)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

DEFINE_OPERATOR(GlobalAveragePool)
DEFINE_OPERATOR(GlobalMaxPool)
DEFINE_OPERATOR(GlobalLpPool)

BEGIN_OPERATOR(BatchNormalization)
    DEFINE_FLOAT_ATTRIBUTE(epsilon, 1e-5f)
    DEFINE_FLOAT_ATTRIBUTE(momentum, 0.9f)

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* mean()  { return input(3); }
    Value* var()   { return input(4); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(InstanceNormalization)
    DEFINE_FLOAT_ATTRIBUTE(epsilon, 1e-5f)

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(LpNormalization)
    DEFINE_INT_ATTRIBUTE(axis, -1)
END_OPERATOR()

BEGIN_OPERATOR(Dropout)
    DEFINE_FLOAT_ATTRIBUTE(ratio, 0.5f)
    Value* mask() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(LRN)
    DEFINE_FLOAT_ATTRIBUTE(alpha, 0.0001f)
    DEFINE_FLOAT_ATTRIBUTE(beta, 0.75f)
    DEFINE_FLOAT_ATTRIBUTE(bias, 1.0f)
    DEFINE_INT_ATTRIBUTE(size, 0)
END_OPERATOR()

BEGIN_OPERATOR(TfIdfVectorizer)
    DEFINE_INT_ATTRIBUTE(max_gram_length, 0)
    DEFINE_INT_ATTRIBUTE(max_skip_count, 0)
    DEFINE_INT_ATTRIBUTE(min_gram_length, 0)
    DEFINE_STRING_ATTRIBUTE(mode, "")
    DEFINE_INTS_ATTRIBUTE(ngram_counts)
    DEFINE_INTS_ATTRIBUTE(ngram_indexes)
    DEFINE_INTS_ATTRIBUTE(pool_int64s)
    DEFINE_STRINGS_ATTRIBUTE(pool_strings)
    DEFINE_FLOATS_ATTRIBUTE(weights)
END_OPERATOR()

BEGIN_OPERATOR(StringNormalizer)
    DEFINE_STRING_ATTRIBUTE(case_change_action, "NONE")
    DEFINE_BOOL_ATTRIBUTE(is_case_sensitive, false)
    DEFINE_STRING_ATTRIBUTE(locale, "")
    DEFINE_STRINGS_ATTRIBUTE(stopwords)
END_OPERATOR()

//==-------------------------------------------------------------------------
// RNN operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(RNN)
    DEFINE_FLOATS_ATTRIBUTE(activation_alpha)
    DEFINE_FLOATS_ATTRIBUTE(activation_beta)
    DEFINE_STRINGS_ATTRIBUTE(activations)
    DEFINE_FLOAT_ATTRIBUTE(clip, 0.0f)
    DEFINE_STRING_ATTRIBUTE(direction, "forward")
    DEFINE_INT_ATTRIBUTE(hidden_size, 0)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* R() { return input(2); }
    Value* B() { return input(3); }
    Value* sequence_lens() { return input(4); }
    Value* initial_h() { return input(5); }

    Value* Y() { return output(0); }
    Value* Y_h() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(GRU)
    DEFINE_FLOATS_ATTRIBUTE(activation_alpha)
    DEFINE_FLOATS_ATTRIBUTE(activation_beta)
    DEFINE_STRINGS_ATTRIBUTE(activations)
    DEFINE_FLOAT_ATTRIBUTE(clip, 0.0f)
    DEFINE_STRING_ATTRIBUTE(direction, "forward")
    DEFINE_INT_ATTRIBUTE(hidden_size, 0)
    DEFINE_INT_ATTRIBUTE(linear_before_reset, 0)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* R() { return input(2); }
    Value* B() { return input(3); }
    Value* sequence_lens() { return input(4); }
    Value* initial_h() { return input(5); }

    Value* Y() { return output(0); }
    Value* Y_h() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(LSTM)
    DEFINE_FLOATS_ATTRIBUTE(activation_alpha)
    DEFINE_FLOATS_ATTRIBUTE(activation_beta)
    DEFINE_STRINGS_ATTRIBUTE(activations)
    DEFINE_FLOAT_ATTRIBUTE(clip, 0.0f)
    DEFINE_STRING_ATTRIBUTE(direction, "forward")
    DEFINE_INT_ATTRIBUTE(hidden_size, 0)
    DEFINE_INT_ATTRIBUTE(input_forget, 0)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* R() { return input(2); }
    Value* B() { return input(3); }
    Value* sequence_lens() { return input(4); }
    Value* initial_h() { return input(5); }
    Value* initial_c() { return input(6); }
    Value* P() { return input(7); }

    Value* Y()   { return output(0); }
    Value* Y_h() { return output(1); }
    Value* Y_c() { return output(2); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// Object detection operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(RoiAlign)
    DEFINE_STRING_ATTRIBUTE(mode, "avg")
    DEFINE_INT_ATTRIBUTE(output_height, 1)
    DEFINE_INT_ATTRIBUTE(output_width, 1)
    DEFINE_INT_ATTRIBUTE(sampling_ratio, 0)
    DEFINE_FLOAT_ATTRIBUTE(spatial_scale, 1.0f)

    Value* rois() { return input(1); }
    Value* batch_indices() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(NonMaxSuppression)
    DEFINE_BOOL_ATTRIBUTE(center_point_box, false)

    Value* boxes() { return input(0); }
    Value* scores() { return input(1); }
    Value* max_output_boxes_per_class() { return input(2); }
    Value* iou_threshold() { return input(3); }
    Value* score_threshold() { return input(4); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// Reduction operators
//==-------------------------------------------------------------------------

#define DEFINE_REDUCTION_OPERATOR(name) \
BEGIN_OPERATOR(name) \
    DEFINE_INTS_ATTRIBUTE(axes) \
    DEFINE_BOOL_ATTRIBUTE(keepdims, true) \
END_OPERATOR()

DEFINE_REDUCTION_OPERATOR(ReduceMax)
DEFINE_REDUCTION_OPERATOR(ReduceMin)
DEFINE_REDUCTION_OPERATOR(ReduceSum)
DEFINE_REDUCTION_OPERATOR(ReduceSumSquare)
DEFINE_REDUCTION_OPERATOR(ReduceMean)
DEFINE_REDUCTION_OPERATOR(ReduceProd)
DEFINE_REDUCTION_OPERATOR(ReduceLogSum)
DEFINE_REDUCTION_OPERATOR(ReduceLogSumExp)
DEFINE_REDUCTION_OPERATOR(ReduceL1)
DEFINE_REDUCTION_OPERATOR(ReduceL2)

BEGIN_OPERATOR(ArgMax)
    DEFINE_INT_ATTRIBUTE(axis, 0)
    DEFINE_BOOL_ATTRIBUTE(keepdims, true)
END_OPERATOR()

BEGIN_OPERATOR(ArgMin)
    DEFINE_INT_ATTRIBUTE(axis, 0)
    DEFINE_BOOL_ATTRIBUTE(keepdims, true)
END_OPERATOR()

#undef DEFINE_REDUCTION_OPERATOR

BEGIN_OPERATOR(CumSum)
    DEFINE_BOOL_ATTRIBUTE(exclusive, false)
    DEFINE_BOOL_ATTRIBUTE(reverse, false)
    Value* axis() { return input(1); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// Tensor operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(Cast)
    DEFINE_DTYPE_ATTRIBUTE(to)
END_OPERATOR()

BEGIN_OPERATOR(Reshape)
    Value* data()     { return input(0); }
    Value* shape()    { return input(1); }
    Value* reshaped() { return output(0); }
END_OPERATOR()

DEFINE_OPERATOR(Shape)
DEFINE_OPERATOR(Size)

BEGIN_OPERATOR(Concat)
    DEFINE_INT_ATTRIBUTE(axis, 0)
END_OPERATOR()

BEGIN_OPERATOR(Split)
    DEFINE_INT_ATTRIBUTE(axis, 0)
    DEFINE_INTS_ATTRIBUTE(split)
END_OPERATOR()

BEGIN_OPERATOR(Slice)
    Value* data()   { return input(0); }
    Value* starts() { return input(1); }
    Value* ends()   { return input(2); }
    Value* axes()   { return input(3); }
    Value* steps()  { return input(4); }
END_OPERATOR()

BEGIN_OPERATOR(Transpose)
    DEFINE_INTS_ATTRIBUTE(perm)
END_OPERATOR()

BEGIN_OPERATOR(Gather)
    DEFINE_INT_ATTRIBUTE(axis, 0)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(GatherElements)
    DEFINE_INT_ATTRIBUTE(axis, 0)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(GatherND)
    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Scatter)
    DEFINE_INT_ATTRIBUTE(axis, 0)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
    Value* updates() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(ScatterElements)
    DEFINE_INT_ATTRIBUTE(axis, 0)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
    Value* updates() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(ScatterND)
    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
    Value* updates() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(Flatten)
    DEFINE_INT_ATTRIBUTE(axis, 1)
END_OPERATOR()

BEGIN_OPERATOR(Squeeze)
    DEFINE_INTS_ATTRIBUTE(axes)
END_OPERATOR()

BEGIN_OPERATOR(Unsqueeze)
    DEFINE_INTS_ATTRIBUTE(axes)
END_OPERATOR()

BEGIN_OPERATOR(Pad)
    DEFINE_INTS_ATTRIBUTE(pads)
    DEFINE_STRING_ATTRIBUTE(mode, "constant")
    DEFINE_FLOAT_ATTRIBUTE(value, 0.0f)
END_OPERATOR()

BEGIN_OPERATOR(SpaceToDepth)
    DEFINE_INT_ATTRIBUTE(blocksize, 0)
END_OPERATOR()

BEGIN_OPERATOR(DepthToSpace)
    DEFINE_INT_ATTRIBUTE(blocksize, 0)
    DEFINE_STRING_ATTRIBUTE(mode, "DCR")
END_OPERATOR()

BEGIN_OPERATOR(Tile)
    Value* repeats() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Resize)
    DEFINE_STRING_ATTRIBUTE(mode, "nearest")
    Value* scales() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Expand)
    Value* shape() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Compress)
    DEFINE_INT_ATTRIBUTE(axis, 0)
    Value* condition() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(OneHot)
    DEFINE_INT_ATTRIBUTE(axis, -1)
    Value* indices() { return input(0); }
    Value* depth()   { return input(1); }
    Value* values()  { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(NonZero)
END_OPERATOR()

BEGIN_OPERATOR(ReverseSequence)
    DEFINE_INT_ATTRIBUTE(batch_axis, 1)
    DEFINE_INT_ATTRIBUTE(time_axis, 0)
    Value* sequence_lens() { return input(1); }
END_OPERATOR()

DEFINE_OPERATOR(Identity)

#undef BEGIN_OPERATOR
#undef END_OPERATOR
#undef DEFINE_OPERATOR
#undef DEFINE_BINARY_OPERATOR
#undef DEFINE_FLOAT_ATTRIBUTE
#undef DEFINE_INT_ATTRIBUTE
#undef DEFINE_BOOL_ATTRIBUTE
#undef DEFINE_STRING_ATTRIBUTE
#undef DEFINE_GRAPH_ATTRIBUTE
#undef DEFINE_TENSOR_ATTRIBUTE
#undef DEFINE_INTS_ATTRIBUTE
#undef DEFINE_FLOATS_ATTRIBUTE
#undef DEFINE_STRING_ATTRIBUTE
#undef DEFINE_SHAPE_ATTRIBUTE
#undef DEFINE_DTYPE_ATTRIBUTE

//==-------------------------------------------------------------------------

class DefaultVisitor : public Visitor {
public:
#define DEFAULT_VISITOR(op) \
    void visit(op* n) override { visitNode(n); }
    FORALL_OPERATORS(DEFAULT_VISITOR)
#undef DEFAULT_VISITOR
    virtual void visit(Node* n) override { visitNode(n); }
    virtual void visitNode(Node*) {}
};

//==-------------------------------------------------------------------------
}} // namespace dlf::model
