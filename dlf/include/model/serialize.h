#pragma once

#include <stdexcept>
#include "model.h"

namespace dlf { namespace model {

class ConvertError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
    ConvertError(const std::string& message) : std::runtime_error(message) {}
};

#define fail_convert(...) throw ConvertError(cxx::concat(__VA_ARGS__))

enum class ModelFormat {
    ONNX
};

template <ModelFormat> std::unique_ptr<Graph> importModel(std::istream& input);

}} // namespace dlf::model
