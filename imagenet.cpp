#include <cstdio>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tensor.h>
#include <model.h>
#include <predict.h>

#include "synset.txt"

template <typename Context = dlf::predict::GPU, typename T = float>
dlf::predict::Predictor<Context, T> create_predictor(const char* path) {
    std::fstream fs(path, std::ios::in | std::ios::binary);
    if (!fs.is_open()) {
        throw std::runtime_error(cxx::string_concat("failed to open ", path));
    }

    auto g = dlf::model::importModel<dlf::model::ONNX>(fs);
    return dlf::predict::Predictor<Context, T>(std::move(g));
}

cv::Mat prepare(const char* path) {
    cv::Mat img;
    cv::resize(cv::imread(path), img, cv::Size(256, 256));
    return img(cv::Rect(16, 16, 224, 224));
}

dlf::Tensor<float> preprocess(const cv::Mat& img) {
    size_t rows = img.rows, cols = img.cols;

    auto tmp1 = dlf::Tensor<uint8_t>({1, rows, cols, 3});
    auto tmp2 = cv::Mat(rows, cols, CV_8UC3, tmp1.data());
    img.copyTo(tmp2);
    cv::cvtColor(tmp2, tmp2, cv::COLOR_BGR2RGB);

    auto dst = dlf::transpose(tmp1.cast<float>(), {0, 3, 1, 2});
    auto mean = dlf::Tensor<float>({1, 3, 1, 1}, {0.485, 0.456, 0.406});
    auto stdev = dlf::Tensor<float>({1, 3, 1, 1}, {0.229, 0.224, 0.225});
    return (dst / 255 - mean) / stdev;
}

std::string postprocess(dlf::Tensor<float>&& scores) {
    scores.reshape({1000});
    dlf::softmax(scores, scores, 0);

    std::vector<size_t> indexes(scores.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::sort(indexes.begin(), indexes.end(), [&](auto i, auto j) {
        return scores(i) > scores(j);
    });

    std::string label = synset[indexes[0]];
    label = label.substr(label.find(' ')+1);
    label = label.substr(0, label.find(','));
    return label;
}

template <class RandomGenerator>
std::vector<cv::Mat> load_images(std::string dir, RandomGenerator& g, size_t n) {
    std::vector<cv::Mat> images;

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
        images.push_back(prepare(files[0].c_str()));
    }

    return images;
}

template <typename Context, typename T>
std::vector<std::string> predict_images(
    dlf::predict::Predictor<Context, T>& predictor,
    const std::vector<cv::Mat>& images)
{
    std::vector<std::string> result;
    for (auto& image : images) {
        predictor.set(0, preprocess(image));
        predictor.predict();
        result.push_back(postprocess(predictor.get(0)));
    }
    return result;
}

cv::Mat create_image_grid(const std::vector<cv::Mat>& images,
                          const std::vector<std::string>& labels,
                          size_t size, size_t w, size_t h)
{
    cv::Mat canvas = cv::Mat::zeros(cv::Size(20 + (20 + size) * w, 20 + (60 + size) * h), CV_8UC3);

    for (int i = 0, x = 20, y = 20; i < images.size(); i++, x += (20 + size)) {
        auto& img = images[i];

        if (i % w == 0 && x != 20) {
            x = 20;
            y += 60 + size;
        }

        // Set the image ROI to display the current image
        cv::Rect ROI(x, y, img.cols, img.rows);
        img.copyTo(canvas(ROI));

        // Show label
        auto font = cv::FONT_HERSHEY_COMPLEX;
        auto color = cv::Scalar(255, 255, 255);
        cv::putText(canvas, labels[i], cv::Point(x, y+size+30), font, 0.65, color, 1
        #if CV_MAJOR_VERSION >= 3
            , cv::LINE_AA
        #endif
        );
    }

    return canvas;
}

static bool is_dir(const char* path) {
    struct stat s;
    if (stat(path, &s) != 0) {
        perror("imagenet");
        exit(1);
    }
    return S_ISDIR(s.st_mode);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("Usage: %s model file\n", argv[0]);
        return 1;
    }

    if (is_dir(argv[2])) {
        auto predictor = create_predictor(argv[1]);
        std::mt19937 rng(std::random_device{}());
        cv::namedWindow("Images");
        do {
            auto images = load_images(argv[2], rng, 15);
            auto labels = predict_images(predictor, images);
            auto canvas = create_image_grid(images, labels, 224, 5, 3);
            cv::imshow("Images", canvas);
        } while (cv::waitKey(0) != 'q');
    } else {
        auto predictor = create_predictor(argv[1]);
        auto image = prepare(argv[2]);
        predictor.set(0, preprocess(image));
        predictor.predict();
        auto label = postprocess(predictor.get(0));
        std::cout << label << std::endl;
    }

    return 0;
}
