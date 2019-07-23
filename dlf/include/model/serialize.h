#pragma once

#include <stdexcept>
#include <fstream>
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

template <typename Format = ONNX>
std::unique_ptr<Graph> import_model(const std::string& path) {
    std::fstream is(path, std::ios::in | std::ios::binary);
    if (is.fail())
        fail_convert(path, ": ", strerror(errno));
    return import_model<Format>(is);
}

template <typename Format = ONNX>
void export_model(const std::string& path, const Graph& graph) {
    std::fstream os(path, std::ios::out | std::ios::binary);
    if (os.fail())
        fail_convert(path, ": ", strerror(errno));
    export_model<Format>(os, graph);
}

}} // namespace dlf::model
