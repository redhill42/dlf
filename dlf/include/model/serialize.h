#pragma once

#include <stdexcept>
#include "model.h"

namespace dlf { namespace model {

class ConvertError : public std::runtime_error {
    std::string expanded_message;

public:
    using std::runtime_error::runtime_error;

    ConvertError(const std::string& message) : std::runtime_error(message) {}

    const char* what() const noexcept override {
        if (!expanded_message.empty())
            return expanded_message.c_str();
        return std::runtime_error::what();
    }

    void appendContext(const std::string& context) {
        expanded_message = cxx::concat(
            std::runtime_error::what(), "\n\n==> Context: ", context);
    }
};

#define fail_convert(...) throw ConvertError(cxx::concat(__VA_ARGS__))

enum class ModelFormat {
    ONNX
};

template <ModelFormat> std::unique_ptr<Graph> importModel(std::istream& input);

}} // namespace dlf::model
