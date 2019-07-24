#include <cstdio>
#include <fstream>
#include "model.h"

int main(int argc, char** argv) {
    using namespace dlf::model;

    if (argc != 3) {
        printf("Usage: optimize input.onnx output.onnx\n");
        return 1;
    }

    try {
        auto g = import_model(argv[1]);
        ShapeInference::newInstance()->infer(*g);
        Optimizer::newInstance()->optimize(*g);
        export_model(argv[2], *g);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
