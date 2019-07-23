#include <cstdio>
#include <random>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <model.h>
#include <predict.h>

using namespace dlf;

constexpr unsigned IMAGE_COUNT  = 10'000;
constexpr unsigned IMAGE_WIDTH  = 28;
constexpr unsigned IMAGE_HEIGHT = 28;
constexpr unsigned IMAGE_SIZE   = IMAGE_WIDTH * IMAGE_HEIGHT;
constexpr unsigned FILE_OFFSET  = 16;

constexpr unsigned IMAGE_COLS = 60;
constexpr unsigned IMAGE_ROWS = 20;

cv::Mat load_image(FILE* file, int num) {
    auto image = cv::Mat(IMAGE_WIDTH, IMAGE_HEIGHT, CV_8UC1);
    unsigned offset = FILE_OFFSET + num * IMAGE_SIZE;
    fseek(file, offset, SEEK_SET);
    auto n = fread(image.data, 1, IMAGE_SIZE, file);
    assert(n == IMAGE_SIZE);
    return image;
}

cv::Mat create_image_grid(const std::vector<cv::Mat>& images,
                          const std::vector<int>& digits)
{
    cv::Mat canvas = cv::Mat::zeros(
        cv::Size(40 + IMAGE_WIDTH*IMAGE_COLS, 40 + (IMAGE_HEIGHT+35)*IMAGE_ROWS),
        CV_8UC1);

    for (int i = 0; i < IMAGE_ROWS; i++) {
        for (int j = 0; j < IMAGE_COLS; j++) {
            auto& img = images[i * IMAGE_COLS + j];
            auto digit = digits[i * IMAGE_COLS + j];
            auto x = 20 + j*IMAGE_WIDTH;
            auto y = 20 + i*(IMAGE_HEIGHT+35);
            img.copyTo(canvas(cv::Rect(x, y, IMAGE_WIDTH, IMAGE_HEIGHT)));

            auto font = cv::FONT_HERSHEY_COMPLEX;
            cv::putText(canvas, std::to_string(digit), cv::Point(x+5, y+IMAGE_HEIGHT+22),
                        font, 0.75, {255, 255, 255}, 1
                #if CV_MAJOR_VERSION >= 3
                      , cv::LINE_AA
                #endif
            );
        }
    }

    cv::Mat result;
    cv::resize(canvas, result, cv::Size(0, 0), 1.5, 1.5);
    return result;
}

template <typename Context = predict::CPU, typename T = float>
predict::Predictor<Context, T> create_predictor(const char* path) {
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error(cxx::string_concat("failed to open ", path));
    }

    auto g = model::import_model(fs);
    return predict::Predictor<Context, T>(std::move(g));
}

Tensor<float> preprocess(const cv::Mat& image) {
    return Tensor<uint8_t>::wrap({1, 1, IMAGE_HEIGHT, IMAGE_WIDTH}, image.data) / 255.f;
}

int postprocess(Tensor<float>&& scores) {
    scores = dnn::softmax(squeeze(std::move(scores)), 0);
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
}

template <typename Context, typename T>
int predict_image(predict::Predictor<Context, T>& pred, const cv::Mat& image) {
    pred.set(0, preprocess(image));
    pred.predict();
    return postprocess(pred.get(0));
}

int main(int argc, char** argv) {
    FILE* file = fopen("t10k-images-idx3-ubyte", "r");
    auto predictor = create_predictor("mnist.onnx");
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> rng(0, IMAGE_COUNT-1);

    cv::namedWindow("Images");
    do {
        std::vector<cv::Mat> images;
        std::vector<int> digits(images.size());
        for (int i = 0; i < IMAGE_ROWS * IMAGE_COLS; i++) {
            images.push_back(load_image(file, rng(gen)));
            digits.push_back(predict_image(predictor, images.back()));
        }

        auto canvas = create_image_grid(images, digits);
        cv::imshow("Images", canvas);
    } while (cv::waitKey(0) != 'q');

    return 0;
}