#pragma once

#include "model.h"

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

#define DEFINE_ATTRIBUTE(name, kind, method) \
    bool has_##name() const noexcept { \
        return hasAttribute(k##name); \
    } \
    const AttributeType<AttributeKind::kind>& name() const { \
        return get_##method(k##name); \
    } \
    void set_##name(AttributeType<AttributeKind::kind> value) { \
        set_##method(k##name, std::move(value)); \
    }

#define DEFINE_DTYPE_ATTRIBUTE(name) \
    bool has_##name() const noexcept { return hasAttribute(k##name); } \
    DataType name() const { \
        return has_##name() ? static_cast<DataType>(get_i(k##name)) : DataType::FLOAT; \
    } \
    void set_##name(DataType dt) { \
        set_i(k##name, static_cast<int64_t>(dt)); \
    }

#define DEFINE_SHAPE_ATTRIBUTE(name) \
    bool has_##name() const noexcept { return hasAttribute(k##name); } \
    Dims name() const { \
        const auto& shape__ = get_is(k##name); \
        return Dims(shape__.begin(), shape__.end()); \
    } \
    void set_##name(const Dims& dims__) { \
        set_is(k##name, std::vector<int64_t>(dims__.begin(), dims__.end())); \
    }

//==-------------------------------------------------------------------------
// Control flow operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(If)
    DEFINE_ATTRIBUTE(then_branch, GRAPH, g)
    DEFINE_ATTRIBUTE(else_branch, GRAPH, g)
END_OPERATOR()

BEGIN_OPERATOR(Loop)
    DEFINE_ATTRIBUTE(body, GRAPH, g)

    Value* M()    { return input(0); }
    Value* cond() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Scan)
    DEFINE_ATTRIBUTE(body, GRAPH, g)
    DEFINE_ATTRIBUTE(num_scan_inputs, INT, i)
    DEFINE_ATTRIBUTE(scan_input_axes, INTS, is)
    DEFINE_ATTRIBUTE(scan_input_directions, INTS, is)
    DEFINE_ATTRIBUTE(scan_output_axes, INTS, is)
    DEFINE_ATTRIBUTE(scan_output_directions, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Where)
    Value* condition() { return input(0); }
    Value* X() { return input(1); }
    Value* Y() { return input(2); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// Generator operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(Constant)
    DEFINE_ATTRIBUTE(value, TENSOR, t)
END_OPERATOR()

BEGIN_OPERATOR(ConstantOfShape)
    DEFINE_ATTRIBUTE(value, TENSOR, t)
END_OPERATOR()

BEGIN_OPERATOR(EyeLike)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(RandomNormal)
    DEFINE_ATTRIBUTE(mean, FLOAT, f)
    DEFINE_ATTRIBUTE(scale, FLOAT, f)
    DEFINE_ATTRIBUTE(seed, FLOAT, f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_SHAPE_ATTRIBUTE(shape)
END_OPERATOR()

BEGIN_OPERATOR(RandomNormalLike)
    DEFINE_ATTRIBUTE(mean, FLOAT, f)
    DEFINE_ATTRIBUTE(scale, FLOAT, f)
    DEFINE_ATTRIBUTE(seed, FLOAT, f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(RandomUniform)
    DEFINE_ATTRIBUTE(high, FLOAT, f)
    DEFINE_ATTRIBUTE(low, FLOAT, f)
    DEFINE_ATTRIBUTE(seed, FLOAT, f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_SHAPE_ATTRIBUTE(shape)
END_OPERATOR()

BEGIN_OPERATOR(RandomUniformLike)
    DEFINE_ATTRIBUTE(high, FLOAT, f)
    DEFINE_ATTRIBUTE(low, FLOAT, f)
    DEFINE_ATTRIBUTE(seed, FLOAT, f)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
END_OPERATOR()

BEGIN_OPERATOR(Multinomial)
    DEFINE_DTYPE_ATTRIBUTE(dtype)
    DEFINE_ATTRIBUTE(sample_size, INT, i)
    DEFINE_ATTRIBUTE(seed, FLOAT, f)
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
    DEFINE_ATTRIBUTE(direction, STRING, s)
END_OPERATOR()

//==-------------------------------------------------------------------------
// Math operators
//==-------------------------------------------------------------------------

DEFINE_BINARY_OPERATOR(Add)
DEFINE_BINARY_OPERATOR(Sub)
DEFINE_BINARY_OPERATOR(Mul)
DEFINE_BINARY_OPERATOR(Div)
DEFINE_BINARY_OPERATOR(Mod)
DEFINE_BINARY_OPERATOR(Pow)

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
    DEFINE_ATTRIBUTE(detect_positive, INT, i)
    DEFINE_ATTRIBUTE(detect_negative, INT, i)
END_OPERATOR()

DEFINE_OPERATOR(Max)
DEFINE_OPERATOR(Min)
DEFINE_OPERATOR(Sum)
DEFINE_OPERATOR(Mean)

BEGIN_OPERATOR(Clip)
    DEFINE_ATTRIBUTE(min, FLOAT, f)
    DEFINE_ATTRIBUTE(max, FLOAT, f)
END_OPERATOR()

DEFINE_OPERATOR(Sigmoid)

DEFINE_OPERATOR(Relu)

BEGIN_OPERATOR(PRelu)
    Value* slope() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(LeakyRelu)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(ThresholdedRelu)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(Selu)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
    DEFINE_ATTRIBUTE(gamma, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(Elu)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(HardSigmoid)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
    DEFINE_ATTRIBUTE(beta, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(Softmax)
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(LogSoftmax)
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(Hardmax)
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

DEFINE_OPERATOR(Softsign)
DEFINE_OPERATOR(Softplus)

BEGIN_OPERATOR(Gemm)
    DEFINE_ATTRIBUTE(alpha,  FLOAT, f)
    DEFINE_ATTRIBUTE(beta,   FLOAT, f)
    DEFINE_ATTRIBUTE(transA, INT, i)
    DEFINE_ATTRIBUTE(transB, INT, i)

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

BEGIN_OPERATOR(TopK)
    DEFINE_ATTRIBUTE(axis, INT, i)

    Value* X() { return input(0); }
    Value* K() { return input(1); }
    Value* indices() { return output(1); }
END_OPERATOR()

//==-------------------------------------------------------------------------
// CNN operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(AveragePool)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(ceil_mode, INT, i)
    DEFINE_ATTRIBUTE(count_include_pad, INT, i)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(MaxPool)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(ceil_mode, INT, i)
    DEFINE_ATTRIBUTE(dilations, INTS, is)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(storage_order, INT, i)
    DEFINE_ATTRIBUTE(strides, INTS, is)

    Value* indices() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(MaxUnpool)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)

    Value* X() { return input(0); }
    Value* I() { return input(1); }
    Value* output_shape() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(MaxRoiPool)
    DEFINE_ATTRIBUTE(pooled_shape, INTS, is)
    DEFINE_ATTRIBUTE(spatial_scale, FLOAT, f)

    Value* X()    { return input(0); }
    Value* rois() { return input(1); }
    Value* Y()    { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(LpPool)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Conv)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(dilations, INTS, is)
    DEFINE_ATTRIBUTE(group, INT, i)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(ConvInteger)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(dilations, INTS, is)
    DEFINE_ATTRIBUTE(group, INT, i)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* X_zero_point() { return input(2); }
    Value* W_zero_point() { return input(3); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(QLinearConv)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(dilations, INTS, is)
    DEFINE_ATTRIBUTE(group, INT, i)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)

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
    DEFINE_ATTRIBUTE(auto_pad, STRING, s)
    DEFINE_ATTRIBUTE(dilations, INTS, is)
    DEFINE_ATTRIBUTE(group, INT, i)
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is)
    DEFINE_ATTRIBUTE(output_padding, INTS, is)
    DEFINE_ATTRIBUTE(output_shape, INTS, is)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(strides, INTS, is)

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

DEFINE_OPERATOR(GlobalAveragePool)
DEFINE_OPERATOR(GlobalMaxPool)
DEFINE_OPERATOR(GlobalLpPool)

BEGIN_OPERATOR(BatchNormalization)
    DEFINE_ATTRIBUTE(epsilon,  FLOAT, f)
    DEFINE_ATTRIBUTE(momentum, FLOAT, f)

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* mean()  { return input(3); }
    Value* var()   { return input(4); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(InstanceNormalization)
    DEFINE_ATTRIBUTE(epsilon,  FLOAT, f)

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(LpNormalization)
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(Dropout)
    DEFINE_ATTRIBUTE(ratio, FLOAT, f)
    Value* mask() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(Shrink)
    DEFINE_ATTRIBUTE(bias, FLOAT, f)
    DEFINE_ATTRIBUTE(lambd, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(Flatten)
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(LRN)
    DEFINE_ATTRIBUTE(alpha, FLOAT, f)
    DEFINE_ATTRIBUTE(beta, FLOAT, f)
    DEFINE_ATTRIBUTE(bias, FLOAT, f)
    DEFINE_ATTRIBUTE(size, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(TfIdfVectorizer)
    DEFINE_ATTRIBUTE(max_gram_length, INT, i)
    DEFINE_ATTRIBUTE(max_skip_count, INT, i)
    DEFINE_ATTRIBUTE(min_gram_length, INT, i)
    DEFINE_ATTRIBUTE(mode, STRING, s)
    DEFINE_ATTRIBUTE(ngram_counts, INTS, is)
    DEFINE_ATTRIBUTE(ngram_indexes, INTS, is)
    DEFINE_ATTRIBUTE(pool_int64s, INTS, is)
    DEFINE_ATTRIBUTE(pool_strings, STRINGS, ss)
    DEFINE_ATTRIBUTE(weights, FLOATS, fs)
END_OPERATOR()

BEGIN_OPERATOR(StringNormalizer)
    DEFINE_ATTRIBUTE(case_change_action, STRING, s)
    DEFINE_ATTRIBUTE(is_case_sensitive, INT, i)
    DEFINE_ATTRIBUTE(locale, STRING, s)
    DEFINE_ATTRIBUTE(stopwords, STRINGS, ss)
END_OPERATOR()

//==-------------------------------------------------------------------------
// RNN operators
//==-------------------------------------------------------------------------

BEGIN_OPERATOR(RNN)
    DEFINE_ATTRIBUTE(activation_alpha, FLOATS, fs)
    DEFINE_ATTRIBUTE(activation_beta, FLOATS, fs)
    DEFINE_ATTRIBUTE(activations, STRINGS, ss)
    DEFINE_ATTRIBUTE(clip, FLOAT, f)
    DEFINE_ATTRIBUTE(direction, STRING, s)
    DEFINE_ATTRIBUTE(hidden_size, INT, i)

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
    DEFINE_ATTRIBUTE(activation_alpha, FLOATS, fs)
    DEFINE_ATTRIBUTE(activation_beta, FLOATS, fs)
    DEFINE_ATTRIBUTE(activations, STRINGS, ss)
    DEFINE_ATTRIBUTE(clip, FLOAT, f)
    DEFINE_ATTRIBUTE(direction, STRING, s)
    DEFINE_ATTRIBUTE(hidden_size, INT, i)
    DEFINE_ATTRIBUTE(linear_before_reset, INT, i)

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
    DEFINE_ATTRIBUTE(activation_alpha, FLOATS, fs)
    DEFINE_ATTRIBUTE(activation_beta, FLOATS, fs)
    DEFINE_ATTRIBUTE(activations, STRINGS, ss)
    DEFINE_ATTRIBUTE(clip, FLOAT, f)
    DEFINE_ATTRIBUTE(direction, STRING, s)
    DEFINE_ATTRIBUTE(hidden_size, INT, i)
    DEFINE_ATTRIBUTE(input_forget, INT, i)

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
    DEFINE_ATTRIBUTE(mode, STRING, s)
    DEFINE_ATTRIBUTE(output_height, INT, i)
    DEFINE_ATTRIBUTE(output_width, INT, i)
    DEFINE_ATTRIBUTE(sampling_ratio, INT, i)
    DEFINE_ATTRIBUTE(spatial_scale, FLOAT, f)

    Value* rois() { return input(1); }
    Value* batch_indices() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(NonMaxSuppression)
    DEFINE_ATTRIBUTE(center_point_box, INT, i)

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
    DEFINE_ATTRIBUTE(axes, INTS, is) \
    DEFINE_ATTRIBUTE(keepdims, INT, i) \
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
    DEFINE_ATTRIBUTE(axis, INT, i)
    DEFINE_ATTRIBUTE(keepdims, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(ArgMin)
    DEFINE_ATTRIBUTE(axis, INT, i)
    DEFINE_ATTRIBUTE(keepdims, INT, i)
END_OPERATOR()

#undef DEFINE_REDUCTION_OPERATOR

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
    DEFINE_ATTRIBUTE(axis, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(Split)
    DEFINE_ATTRIBUTE(axis, INT, i)
    DEFINE_ATTRIBUTE(split, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Slice)
    Value* data()   { return input(0); }
    Value* starts() { return input(1); }
    Value* ends()   { return input(2); }
    Value* axes()   { return input(3); }
    Value* steps()  { return input(4); }
END_OPERATOR()

BEGIN_OPERATOR(Transpose)
    DEFINE_ATTRIBUTE(perm, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Scatter)
    DEFINE_ATTRIBUTE(axis, INT, i)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
    Value* updates() { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(Gather)
    DEFINE_ATTRIBUTE(axis, INT, i)

    Value* data()    { return input(0); }
    Value* indices() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Squeeze)
    DEFINE_ATTRIBUTE(axes, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Unsqueeze)
    DEFINE_ATTRIBUTE(axes, INTS, is)
END_OPERATOR()

BEGIN_OPERATOR(Pad)
    DEFINE_ATTRIBUTE(pads, INTS, is)
    DEFINE_ATTRIBUTE(mode, STRING, s)
    DEFINE_ATTRIBUTE(value, FLOAT, f)
END_OPERATOR()

BEGIN_OPERATOR(SpaceToDepth)
    DEFINE_ATTRIBUTE(blocksize, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(DepthToSpace)
    DEFINE_ATTRIBUTE(blocksize, INT, i)
END_OPERATOR()

BEGIN_OPERATOR(Tile)
    Value* repeats() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Resize)
    DEFINE_ATTRIBUTE(mode, STRING, s)
    Value* scales() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Expand)
    Value* shape() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(Compress)
    DEFINE_ATTRIBUTE(axis, INT, i)
    Value* condition() { return input(1); }
END_OPERATOR()

BEGIN_OPERATOR(OneHot)
    DEFINE_ATTRIBUTE(axis, INT, i)
    Value* indices() { return input(0); }
    Value* depth()   { return input(1); }
    Value* values()  { return input(2); }
END_OPERATOR()

BEGIN_OPERATOR(NonZero)
END_OPERATOR()

BEGIN_OPERATOR(ReverseSequence)
    DEFINE_ATTRIBUTE(batch_axis, INT, i)
    DEFINE_ATTRIBUTE(time_axis, INT, i)
    Value* sequence_lens() { return input(1); }
END_OPERATOR()

DEFINE_OPERATOR(Identity)

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
