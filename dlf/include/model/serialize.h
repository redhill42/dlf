#pragma once

#include <stdexcept>
#include "model.h"

namespace dlf { namespace model {

class ConvertError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    ConvertError(const std::string& message) : std::runtime_error(message) {}
};

#define fail_convert(...) throw ConvertError(cxx::string_concat(__VA_ARGS__))

struct ONNX {};

template <typename Format = ONNX> std::unique_ptr<Graph> import_model(std::istream& input);
template <typename Format = ONNX> void export_model(std::ostream& output, const Graph& graph);

}} // namespace dlf::model
