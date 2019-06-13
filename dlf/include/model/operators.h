#pragma once

#include "model.h"

namespace dlf { namespace model {

#define DEFINE_ATTRIBUTE(name, kind, method) \
    bool has_##name() const { \
        return hasAttribute(k##name); \
    } \
    const AttributeType<AttributeKind::kind>& name() const { \
        return get_##method(k##name); \
    } \

//==-------------------------------------------------------------------------

class Add : public Node {
public:
    static constexpr NodeKind Kind = kAdd;
    Add(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return output(); }
};

class BatchNormalization : public Node {
public:
    static constexpr NodeKind Kind = kBatchNormalization;
    BatchNormalization(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    DEFINE_ATTRIBUTE(epsilon,  FLOAT, f);
    DEFINE_ATTRIBUTE(momentum, FLOAT, f);

    Value* X()     { return input(0); }
    Value* scale() { return input(1); }
    Value* B()     { return input(2); }
    Value* mean()  { return input(3); }
    Value* var()   { return input(4); }
    Value* Y()     { return output(); }
};

class Conv : public Node {
public:
    static constexpr NodeKind Kind = kConv;
    Conv(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    DEFINE_ATTRIBUTE(auto_pad, STRING, s);
    DEFINE_ATTRIBUTE(dilations, INTS, is);
    DEFINE_ATTRIBUTE(group, INT, i);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(strides, INTS, is);

    Value* X() { return input(0); }
    Value* W() { return input(1); }
    Value* B() { return inputs().size() < 3 ? nullptr : input(2); }
    Value* Y() { return output(); }
};

class Flatten : public Node {
public:
    static constexpr NodeKind Kind = kFlatten;
    Flatten(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    DEFINE_ATTRIBUTE(axis, INT, i);

    Value* X() { return input(); }
    Value* Y() { return output(); }
};

class Gemm : public Node {
public:
    static constexpr NodeKind Kind = kGemm;
    Gemm(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    DEFINE_ATTRIBUTE(alpha,  FLOAT, f);
    DEFINE_ATTRIBUTE(beta,   FLOAT, f);
    DEFINE_ATTRIBUTE(transA, INT, i);
    DEFINE_ATTRIBUTE(transB, INT, i);

    Value* A() { return input(0); }
    Value* B() { return input(1); }
    Value* C() { return input(2); }
    Value* Y() { return output(); }
};

class GlobalAveragePool : public Node {
public:
    static constexpr NodeKind Kind = kGlobalAveragePool;
    GlobalAveragePool(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    Value* X() { return input(); }
    Value* Y() { return output(); }
};

class MaxPool : public Node {
public:
    static constexpr NodeKind Kind = kMaxPool;
    MaxPool(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    DEFINE_ATTRIBUTE(ceil_mode, INT, i);
    DEFINE_ATTRIBUTE(dilations, INTS, is);
    DEFINE_ATTRIBUTE(kernel_shape, INTS, is);
    DEFINE_ATTRIBUTE(pads, INTS, is);
    DEFINE_ATTRIBUTE(storage_order, INT, i);
    DEFINE_ATTRIBUTE(strides, INTS, is);

    Value* X() { return input(); }
    Value* Y() { return output(0); }
    Value* Indices() { return outputs().size()==1 ? nullptr : output(1); }
};

class Relu : public Node {
public:
    static constexpr NodeKind Kind = kRelu;
    Relu(Graph* graph) : Node(graph, Kind) {}
    void accept(Visitor& v) override { v.visit(this); }

    Value* X() { return input(); }
    Value* Y() { return output(); }
};

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
