#include "model/serialize.h"

namespace dlf { namespace model {

template <typename Src, typename Dst>
static inline void copy(const Src& src, Dst& dst) {
    dst.resize(src.size());
    std::copy(src.begin(), src.end(), dst.begin());
}

void Serializer::load(const onnx::TensorProto& proto, TensorData& tensor) {
    tensor.set_dims({proto.dims().begin(), proto.dims().end()});
    tensor.set_type(static_cast<DataType>(proto.data_type()));

    if (proto.has_name()) {
        tensor.set_name(proto.name());
    }

    if (proto.has_raw_data()) {
        tensor.set_raw_data(proto.raw_data());
    } else {
        copy(proto.float_data(),  tensor.float_data());
        copy(proto.double_data(), tensor.double_data());
        copy(proto.int32_data(),  tensor.int32_data());
        copy(proto.int64_data(),  tensor.int64_data());
        copy(proto.uint64_data(), tensor.uint64_data());
        copy(proto.string_data(), tensor.string_data());
    }
}

}} // namespace dlf::model
