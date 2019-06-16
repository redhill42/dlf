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

#define DEFINE_ATTRIBUTE(name, kind, method) \
    bool has_##name() const { \
        return hasAttribute(k##name); \
    } \
    const AttributeType<AttributeKind::kind>& name() const { \
        return get_##method(k##name); \
    } \

//==-------------------------------------------------------------------------

BEGIN_OPERATOR(Add)
    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(AveragePool)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s);
    DEFINE_ATTRIBUTE(ceil_mode, INT, i);
    DEFINE_ATTRIBUTE(count_include_pad, INT, i);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(strides, INTS, is);
END_OPERATOR()

BEGIN_OPERATOR(BatchNormalization)
    DEFINE_ATTRIBUTE(epsilon,  FLOAT, f);
    DEFINE_ATTRIBUTE(momentum, FLOAT, f);

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* mean()  { return input(3); }
    Value* var()   { return input(4); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(Constant)
    DEFINE_ATTRIBUTE(value, TENSOR, t);
END_OPERATOR()

BEGIN_OPERATOR(Conv)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s);
    DEFINE_ATTRIBUTE(dilations, INTS, is);
    DEFINE_ATTRIBUTE(group, INT, i);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(strides, INTS, is);

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(ConvInteger)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s);
    DEFINE_ATTRIBUTE(dilations, INTS, is);
    DEFINE_ATTRIBUTE(group, INT, i);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(strides, INTS, is);

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* X_zero_point() { return input(2); }
    Value* W_zero_point() { return input(3); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(Dropout)
    DEFINE_ATTRIBUTE(ratio, FLOAT, f);
    Value* mask() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(Flatten)
    DEFINE_ATTRIBUTE(axis, INT, i);
END_OPERATOR()

BEGIN_OPERATOR(Gemm)
    DEFINE_ATTRIBUTE(alpha,  FLOAT, f);
    DEFINE_ATTRIBUTE(beta,   FLOAT, f);
    DEFINE_ATTRIBUTE(transA, INT, i);
    DEFINE_ATTRIBUTE(transB, INT, i);

    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return input(2); }
    Value* Y() { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(GlobalAveragePool)
END_OPERATOR()

BEGIN_OPERATOR(GlobalLpPool)
END_OPERATOR()

BEGIN_OPERATOR(GlobalMaxPool)
END_OPERATOR()

BEGIN_OPERATOR(InstanceNormalization)
    DEFINE_ATTRIBUTE(epsilon,  FLOAT, f);

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* Y()     { return output(); }
END_OPERATOR()

BEGIN_OPERATOR(LpNormalization)
    DEFINE_ATTRIBUTE(axis, INT, i);
END_OPERATOR()

BEGIN_OPERATOR(LpPool)
    DEFINE_ATTRIBUTE(auto_pad, STRING, s);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(strides, INTS, is);
END_OPERATOR()

BEGIN_OPERATOR(MaxPool)
    DEFINE_ATTRIBUTE(ceil_mode, INT, i);
    DEFINE_ATTRIBUTE(dilations, INTS, is);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(storage_order, INT, i);
    DEFINE_ATTRIBUTE(strides, INTS, is);

    Value* Indices() { return output(1); }
END_OPERATOR()

BEGIN_OPERATOR(Relu)
END_OPERATOR()

BEGIN_OPERATOR(Reshape)
    Value* data()     { return input(0); }
    Value* shape()    { return input(1); }
    Value* reshaped() { return output(0); }
END_OPERATOR()

BEGIN_OPERATOR(Shrink)
    DEFINE_ATTRIBUTE(bias, FLOAT, f);
    DEFINE_ATTRIBUTE(lambd, FLOAT, f);
END_OPERATOR()

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
