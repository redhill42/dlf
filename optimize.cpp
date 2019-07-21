#include <cstdio>
#include <fstream>
#include "model.h"

int main(int argc, char** argv) {
    using namespace dlf::model;

    if (argc != 3) {
        printf("Usage: optimize input.onnx output.onnx\n");
        return 1;
    }

    std::fstream is(argv[1], std::ios::in | std::ios::binary);
    if (is.fail()) {
        std::cerr << argv[1] << ": " << strerror(errno)  << std::endl;
        return 1;
    }

    std::fstream os(argv[2], std::ios::out | std::ios::binary);
    if (os.fail()) {
        std::cerr << argv[2] << ": " << strerror(errno) << std::endl;
        return 1;
    }

    auto g = import_model(is);
    ShapeInference::Instance().infer(*g);
    Optimizer().optimize(*g);
    export_model(os, *g);

    return 0;
}
