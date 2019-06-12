#pragma once

#include "model.h"
#include "onnx.pb.h"

namespace dlf { namespace model {

class Serializer {
public:
    template <typename Derived>
    void load(const onnx::NodeProto& proto, Attributes<Derived>& attr);

    void load(const onnx::TensorProto& proto, TensorData& tensor);
};

template <typename Derived>
void Serializer::load(const onnx::NodeProto& proto, Attributes<Derived>& attr) {
    auto size = proto.attribute_size();
    for (int i = 0; i < size; i++) {
        auto ap = proto.attribute(i);
        auto name = Symbol(ap.name());

        switch (ap.type()) {
        case onnx::AttributeProto::FLOAT:
            attr.set_f(name, ap.f());
            break;

        case onnx::AttributeProto::INT:
            attr.set_i(name, ap.i());
            break;

        case onnx::AttributeProto::STRING:
            attr.set_s(name, ap.s());
            break;

        case onnx::AttributeProto::FLOATS: {
            std::vector<float> fs(ap.floats().begin(), ap.floats().end());
            attr.set_fs(name, std::move(fs));
            break;
          }

        case onnx::AttributeProto::INTS: {
            std::vector<int64_t> is(ap.ints().begin(), ap.ints().end());
            attr.set_is(name, std::move(is));
            break;
          }

        case onnx::AttributeProto::STRINGS: {
            std::vector<std::string> ss(ap.strings().begin(), ap.strings().end());
            attr.set_ss(name, std::move(ss));
            break;
          }

        default:
            // ignore other types now
            break;
        }
    }
}

}} // namespace dlf::model
