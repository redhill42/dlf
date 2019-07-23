#include <cstdio>
#include <algorithm>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tensor.h>
#include <model.h>
#include <predict.h>

#include "synset.txt"

using namespace dlf;
using Context = predict::CPU;
using Real = float;
using Predictor = predict::Predictor<Context, Real>;

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

    void setResult(const std::vector<Score> scores) {
        std::copy(scores.begin(), scores.begin()+5, inferred);
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

Predictor create_predictor(const char* path) {
    auto g = model::import_model(path);
    return predict::Predictor<Context, Real>(std::move(g));
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

Tensor<Real> preprocess(const cv::Mat& img) {
    size_t rows = img.rows, cols = img.cols;

    cv::Mat tmp_img;
    cv::cvtColor(img, tmp_img, cv::COLOR_BGR2RGB);

    auto tmp = Tensor<uint8_t>::wrap({1, rows, cols, 3}, tmp_img.data);
    auto mean = Tensor<Real>({3}, {0.485, 0.456, 0.406});
    auto stdev = Tensor<Real>({3}, {0.229, 0.224, 0.225});
    return transpose((tmp / Real{255} - mean) / stdev, {0, 3, 1, 2});
}

std::vector<Score> postprocess(Tensor<Real>&& scores) {
    scores = dnn::softmax(squeeze(std::move(scores)), 0);

    std::vector<Score> result(scores.size());
    for (int i = 0; i < 1000; i++) {
        result[i].index = i;
        result[i].score = scores(i);
    }

    std::sort(result.begin(), result.end(), [](auto x, auto y) {
        return x.score > y.score;
    });

    return result;
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

void predict_images(Predictor& predictor, std::vector<ImageClass>& images) {
    for (auto& img : images) {
        predictor.set(0, preprocess(img.image));
        predictor.predict();
        img.setResult(postprocess(predictor.get(0)));
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
        std::string title = argv[1];
        auto pos = title.rfind('/');
        if (pos != std::string::npos)
            title = title.substr(pos+1);
        title = "Image Classification - " + title;

        auto predictor = create_predictor(argv[1]);
        std::mt19937 rng(std::random_device{}());
        cv::namedWindow(title);

        do {
            auto images = load_images(argv[2], rng, 15);
            predict_images(predictor, images);
            auto canvas = create_image_grid(images, 224, 5, 3);
            cv::imshow(title, canvas);
        } while (cv::waitKey(0) != 'q');
    } else {
        auto predictor = create_predictor(argv[1]);
        auto image = prepare(argv[2]);
        predictor.set(0, preprocess(image));
        predictor.predict();
        auto result = postprocess(predictor.get(0));

        for (int i = 0; i < 5; i++) {
            auto x = result[i];
            std::cout << ImageLabel(x.index) << ": "
                      << static_cast<int>(x.score*100) << "%\n";
        }
    }

    return 0;
}
