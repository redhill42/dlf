#include "model.h"
#include "model/operators.h"

namespace dlf { namespace model {

static void copyShape(const Value* from, Value* to) {
    if (to->type() == DataType::UNDEFINED) {
        to->set_type(from->type());
        to->set_dims(from->dims());
    }
}

static void inferConvShape(Node* n, std::vector<size_t>& dims) {
    int axes = dims.size() - 2;
    for (int i = 0; i < axes; i++) {
        size_t width  = dims[i+2];
        size_t kernel = n->get_is(kkernel_shape)[i];
        size_t pads   = 0;
        size_t stride = 1;

        if (n->hasAttribute(kpads))
            pads = n->get_is(kpads)[i] + n->get_is(kpads)[i+axes];
        if (n->hasAttribute(kstrides))
            stride = n->get_is(kstrides)[i];
        dims[i+2] = (width - kernel + pads) / stride + 1;
    }
}

class ShapeInferenceImpl final : public ShapeInference, DefaultVisitor {
public:
    void infer(Node* n) override {
        n->accept(*this);
    }

    void visit(Conv* n) override {
        if (n->Y()->type() == DataType::UNDEFINED) {
            auto dims = n->X()->dims();
            dims[1] = n->W()->dims()[0];
            inferConvShape(n, dims);

            n->Y()->set_type(n->X()->type());
            n->Y()->set_dims(dims);
        }
    }

    void visit(MaxPool* n) override {
        if (n->Y()->type() == DataType::UNDEFINED) {
            auto dims = n->X()->dims();
            inferConvShape(n, dims);

            n->Y()->set_type(n->X()->type());
            n->Y()->set_dims(dims);
        }
    }

    void visit(GlobalAveragePool* n) override {
        if (n->Y()->type() == DataType::UNDEFINED) {
            auto dims = n->X()->dims();
            std::fill(dims.begin()+2, dims.end(), 1);

            n->Y()->set_type(n->X()->type());
            n->Y()->set_dims(dims);
        }
    }

    void visit(Gemm* n) override {
        size_t M = n->A()->dims()[0];
        size_t K = n->A()->dims()[1];
        size_t P = n->B()->dims()[0];
        size_t N = n->B()->dims()[1];

        if (n->get_i(ktransA, 0))
            std::swap(M, K);
        if (n->get_i(ktransB, 0))
            std::swap(P, N);
        assert(K == P);

        n->Y()->set_type(n->A()->type());
        n->Y()->set_dims({M, N});
        n->C()->set_dims({M, N});
    }

    void visit(Add* n) override {
        assert(n->A()->dims() == n->B()->dims());
        copyShape(n->A(), n->C());
    }

    void visit(Flatten* n) override {
        if (n->Y()->type() == DataType::UNDEFINED) {
            auto dims = n->X()->dims();
            size_t axis = n->get_i(kaxis, 1);
            size_t a = std::accumulate(dims.begin(), dims.begin()+axis, 1, std::multiplies<>());
            size_t b = std::accumulate(dims.begin()+axis, dims.end(), 1, std::multiplies<>());

            n->Y()->set_type(n->X()->type());
            n->Y()->set_dims({a, b});
        }
    }

    void visitNode(Node* n) override {
        copyShape(n->input(0), n->output(0));
    }
};

ShapeInference& ShapeInference::Instance() {
    static ShapeInferenceImpl instance;
    return instance;
}

}} // namespace dlf:: model