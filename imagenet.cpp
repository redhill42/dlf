#include <cstdio>
#include <algorithm>
#include <chrono>

#include <ctype.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tensor.h>
#include <model.h>
#include <predict.h>

#include "synset.txt"

using namespace dlf;
using Real = float;

static std::string ImageLabel(size_t i) {
    std::string label = synset[i];
    label = label.substr(label.find(' ')+1);
    label = label.substr(0, label.find(','));
    return label;
}

struct Score {
    size_t index;
    Real   score;
};

struct ImageClass {
    cv::Mat image;          // The original image
    size_t  original;       // The original image class
    Score   inferred[5];    // The top-5 inferred image class and score

    ImageClass(cv::Mat&& image, size_t original)
        : image(std::move(image)), original(original) {}

    void setResult(const Tensor<Real>& scores, const Tensor<int>& indices) {
        for (int i = 0; i < 5; i++) {
            inferred[i].score = scores(i);
            inferred[i].index = indices(i);
        }
    }

    bool hit() const {
        return original == inferred[0].index;
    }

    bool hitTop5() const {
        for (int i = 0; i < 5; i++)
            if (original == inferred[i].index)
                return true;
        return false;
    }

    std::string label() const {
        return ImageLabel(inferred[0].index);
    }
};

template <typename Context>
predict::Predictor<Context, Real> create_predictor(
    const char* path, const std::unordered_map<std::string, size_t>& env,
    const std::vector<size_t>& input_shape)
{
    auto g = model::import_model(path);
    g->input(0)->set_dims(input_shape);
    return predict::Predictor<Context, Real>(std::move(g), env);
}

#define KEEP_RATIO 1
cv::Mat prepare(const std::string& path, int size = 224) {
    cv::Mat src_img = cv::imread(path);
    cv::Mat dst_img;

#if KEEP_RATIO
    auto ratio = static_cast<double>(size) / std::min(src_img.rows, src_img.cols);
    auto dx = std::max(size, static_cast<int>(std::round(src_img.cols*ratio)));
    auto dy = std::max(size, static_cast<int>(std::round(src_img.rows*ratio)));
    auto x = (dx - size) / 2;
    auto y = (dy - size) / 2;

    cv::resize(src_img, dst_img, cv::Size(dx, dy));
    return dst_img(cv::Rect(x, y, size, size));
#else
    cv::resize(src_img, dst_img, cv::Size(256, 256));
    return dst_img(cv::Rect(16, 16, size, size));
#endif
}

auto preprocess(const cv::Mat& img) {
    size_t rows = img.rows, cols = img.cols;

    cv::Mat tmp_img;
    cv::cvtColor(img, tmp_img, cv::COLOR_BGR2RGB);

    auto pixels = Tensor<uint8_t>::wrap({1, rows, cols, 3}, tmp_img.data) / Real{255};
    auto mean = Tensor<Real>({3}, {0.485, 0.456, 0.406});
    auto stdev = Tensor<Real>({3}, {0.229, 0.224, 0.225});
    return ((pixels - mean) / stdev).transpose(0, 3, 1, 2);
}

std::pair<Tensor<Real>, Tensor<int>> postprocess(const Tensor<Real>& scores) {
    return top_k(dnn::softmax(scores), 5);
}

template <class RandomGenerator>
std::vector<ImageClass> load_images(std::string dir, RandomGenerator& g, size_t n) {
    std::vector<ImageClass> images;

    std::vector<size_t> indexes(cxx::size(synset));
    std::iota(indexes.begin(), indexes.end(), 0);
    std::shuffle(indexes.begin(), indexes.end(), g);

    for (size_t i = 0; i < n; i++) {
        std::string label = synset[indexes[i]];
        label = label.substr(0, label.find(' '));

        std::string subdir = dir + '/' + label;
        DIR* d = opendir(subdir.c_str());
        if (!d) throw std::runtime_error(cxx::string_concat("cannot open directory ", subdir));

        std::vector<std::string> files;
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string d_name = ent->d_name;
            if (d_name.find(".JPEG") != std::string::npos)
                files.push_back(subdir + '/' + d_name);
        }
        closedir(d);

        std::shuffle(files.begin(), files.end(), g);
        images.emplace_back(prepare(files[0]), indexes[i]);
    }

    return images;
}

template <typename Context>
void predict_images(predict::Predictor<Context, Real>& predictor, std::vector<ImageClass>& images) {
    Tensor<Real> batch({images.size(), 3, size_t(images[0].image.rows), size_t(images[0].image.cols)});
    for (int i = 0; i < images.size(); i++) {
        reorder(preprocess(images[i].image), unsqueeze(batch[i], 0));
    }

    predictor.set(0, batch);
    predictor.predict();

    auto result = postprocess(predictor.get(0));
    for (int i = 0; i < images.size(); i++) {
        images[i].setResult(result.first[i], result.second[i]);
    }
}

cv::Mat create_image_grid(
    const std::vector<ImageClass>& images,
    size_t size, size_t w, size_t h)
{
    cv::Mat canvas = cv::Mat::zeros(cv::Size(20 + (20+size)*w, 20 + (60+size)*h), CV_8UC3);

    for (int i = 0, x = 20, y = 20; i < images.size(); i++, x += (20+size)) {
        auto& img = images[i].image;

        if (i % w == 0 && x != 20) {
            x = 20;
            y += 60 + size;
        }

        // Set the image ROI to display the current image
        cv::Rect ROI(x, y, img.cols, img.rows);
        img.copyTo(canvas(ROI));

        // Show label
        auto& result = images[i];
        cv::Scalar color;
        if (result.hit()) {
            color = cv::Scalar(255, 255, 255);
        } else if (result.hitTop5()) {
            color = cv::Scalar(0, 255, 255);
        } else {
            color = cv::Scalar(0, 0, 255);
        }

        auto font = cv::FONT_HERSHEY_COMPLEX;
        cv::putText(canvas, result.label(), cv::Point(x, y+size+22), font, 0.5, color, 1
        #if CV_MAJOR_VERSION >= 3
            , cv::LINE_AA
        #endif
        );

        if (!result.hit()) {
            cv::putText(canvas, ImageLabel(result.original),
                        cv::Point(x, y+size+47), font, 0.5,
                        cv::Scalar(255, 255, 255), 1
            #if CV_MAJOR_VERSION >= 3
                      , cv::LINE_AA
            #endif
            );
        }
    }

    return canvas;
}

static bool is_dir(const char* path) {
    struct stat s;
    if (stat(path, &s) != 0) {
        perror(path);
        exit(1);
    }
    return S_ISDIR(s.st_mode);
}

template <typename Context>
void run_interactive(const char* model_path, const char* data_path,
                     const std::unordered_map<std::string, size_t>& env)
{
    std::string title = model_path;
    auto pos = title.rfind('/');
    if (pos != std::string::npos)
        title = title.substr(pos+1);
    title = "Image Classification - " + title;

    auto predictor = create_predictor<Context>(model_path, env, {15, 3, 224, 224});
    std::mt19937 rng(std::random_device{}());
    cv::namedWindow(title);

    do {
        auto images = load_images(data_path, rng, 15);
        predict_images(predictor, images);
        auto canvas = create_image_grid(images, 224, 5, 3);
        cv::imshow(title, canvas);
    } while (cv::waitKey(0) != 'q');
}

template <typename Context>
void run_benchmark(const char* model_path, const char* data_path,
                   const std::unordered_map<std::string, size_t>& env,
                   size_t batch_size, size_t round)
{
    // prepare batch
    auto predictor = create_predictor<Context>(model_path, env, {batch_size, 3, 224, 224});
    auto image = prepare(data_path);
    auto input = Tensor<Real>({batch_size, 3, 224, 224});
    for (size_t i = 0; i < batch_size; i++) {
        reorder(preprocess(image), unsqueeze(input[i], 0));
    }

    // show prediction result
    predictor.set(0, input);
    predictor.predict();
    auto result = postprocess(predictor.get(0));
    for (int i = 0; i < 5; i++) {
        auto score = result.first(0, i);
        auto index = result.second(0, i);
        std::cout << ImageLabel(index) << ": " << score*100 << "%\n";
    }

    // warm up
    for (int i = 0; i < 10; i++) {
        predictor.set(0, input);
        predictor.predict();
        predictor.get(0);
    }

    // cooking
    auto start = std::chrono::high_resolution_clock().now();
    for (size_t i = 0; i < round; i++) {
        predictor.set(0, input);
        predictor.predict();
        predictor.get(0);
    }
    auto end = std::chrono::high_resolution_clock().now();

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << elapsed.count() << " seconds\n";
    std::cout << int(round * batch_size / elapsed.count()) << " images/second\n";
}

template <typename Context>
void run_program(const char* model_path, const char* data_path,
                 const std::unordered_map<std::string, size_t>& env,
                 size_t batch_size, size_t round)
{
    if (is_dir(data_path)) {
        run_interactive<Context>(model_path, data_path, env);
    } else {
        run_benchmark<Context>(model_path, data_path, env, batch_size, round);
    }
}

int main(int argc, char** argv) {
    std::unordered_map<std::string, size_t> env;
    bool    gpu = true;
    int     batch_size = 1;
    int     round = 1000;
    char*   endptr = nullptr;
    int     c;

    while ((c = getopt(argc, argv, "CD:b:r:")) != -1) {
        switch (c) {
        case 'D': {
            std::string def = optarg;
            auto sep = def.find('=');
            if (sep == std::string::npos) {
                std::cerr << "Invalid definition: " << def << std::endl;
                return 1;
            }

            size_t def_val = std::strtol(def.substr(sep+1).c_str(), &endptr, 10);
            if (*endptr != '\0') {
                std::cerr << "Invalid definition: " << def << std::endl;
                return 1;
            }

            env[def.substr(0, sep)] = def_val;
            break;
        }

        case 'C':
            gpu = false;
            break;
        case 'b':
            batch_size = std::strtol(optarg, &endptr, 10);
            if (*endptr != '\0' || batch_size <= 0) {
                std::cerr << "Invalid batch size `" << optarg << "'.\n";
                return 1;
            }
            break;
        case 'r':
            round = std::strtol(optarg, &endptr, 10);
            if (*endptr != '\0' || round <= 0) {
                std::cerr << "Invalid iteration count `" << optarg << "'.\n";
                return 1;
            }
            break;
        case '?':
            std::cerr << "Unknown option `-" << optopt << "'.\n";
            return 1;
        default:
            abort();
        }
    }

    if (argc - optind != 2) {
        printf("Usage: %s model file\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[optind];
    const char* data_path = argv[optind+1];

    if (gpu) {
        run_program<predict::GPU>(model_path, data_path, env, batch_size, round);
    } else {
        run_program<predict::CPU>(model_path, data_path, env, batch_size, round);
    }

    return 0;
}
