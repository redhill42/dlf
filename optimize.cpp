#include <cstdio>
#include <fstream>
#include "model.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: optimize input.onnx output.onnx\n");
        return 1;
    }

    std::fstream is(argv[1], std::ios::in | std::ios::binary);
    auto g = dlf::model::importModel<dlf::model::ONNX>(is);
    dlf::model::ShapeInference::Instance().infer(*g);
    dlf::model::Optimizer optimizer;
    optimizer.optimize(*g);

    std::fstream os(argv[2], std::ios::out | std::ios::binary);
    dlf::model::exportModel<dlf::model::ONNX>(os, *g);

    return 0;
}
