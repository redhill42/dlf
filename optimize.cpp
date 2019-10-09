#include <cstdio>
#include <fstream>
#include <unistd.h>
#include "model.h"

int main(int argc, char** argv) {
    using namespace dlf::model;

    std::unordered_map<std::string, size_t> env;
    int c;

    while ((c = getopt(argc, argv, "D:")) != -1) {
        switch (c) {
        case 'D': {
            std::string arg = optarg;
            auto sep = arg.find('=');
            if (sep == std::string::npos) {
                std::cerr << "Invalid definition: " << arg << std::endl;
                return 1;
            }

            char* endptr;
            size_t arg_val = std::strtol(arg.substr(sep+1).c_str(), &endptr, 10);
            if (*endptr != '\0') {
                std::cerr << "Invalid definition: " << arg << std::endl;
                return 1;
            }

            env[arg.substr(0, sep)] = arg_val;
            break;
        }

        case '?':
            std::cerr << "Unknown option `-" << optopt << "'.\n";
            return 1;

        default:
            abort();
        }
    }

    if (argc - optind != 2) {
        printf("Usage: optimize input.onnx output.onnx\n");
        return 1;
    }

    try {
        auto g = import_model(argv[optind]);
        ShapeInference::newInstance(env)->infer(*g);
        Optimizer::newInstance()->optimize(*g);
        export_model(argv[optind+1], *g);
    } catch (std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}
