#include "model.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace dlf::model;

TEST(TensorData, SetData) {
    std::vector<int16_t> data(24);
    std::generate(data.begin(), data.end(), [i=1]() mutable { return i++; });
    TensorData dt({2, 3, 4}, data.begin(), data.end());

    auto& v = dt.int32_data();
    EXPECT_EQ(v.size(), 24);
    for (int i = 0; i < 24; i++) {
        EXPECT_EQ(v[i], i + 1);
    }
}

TEST(TensorData, SetIncompatibleData) {
    std::vector<int16_t> data(24);
    std::fill(data.begin(), data.end(), 1);
    TensorData dt({2, 3, 4}, DataType::INT32);
    EXPECT_ANY_THROW(dt.set_data(data.begin(), data.end()));
}

TEST(TensorData, Encode) {
    auto t = dlf::Tensor<int32_t>::range({2, 3, 4}, 1);
    auto dt = TensorData(t);

    EXPECT_EQ(dt.dims(), std::vector<size_t>({2, 3, 4}));
    EXPECT_EQ(dt.int32_data().size(), 24);
}

TEST(TensorData, Decode) {
    std::vector<int32_t> data(24);
    std::generate(data.begin(), data.end(), [i=1]() mutable { return i++; });
    TensorData dt({2, 3, 4}, data.begin(), data.end());
    EXPECT_EQ(dt.decode<int16_t>(), dlf::Tensor<int16_t>::range({2, 3, 4}, 1));
}

TEST(TensorData, Complex) {
    TensorData dt({2}, DataType::COMPLEX64);
    dt.float_data().push_back(1);
    dt.float_data().push_back(2);
    dt.float_data().push_back(3);
    dt.float_data().push_back(4);

    auto t = dt.decode<std::complex<float>>();
    EXPECT_EQ(t.shape(), dlf::Shape({2}));
    EXPECT_EQ(t(0), std::complex<float>(1, 2));
    EXPECT_EQ(t(1), std::complex<float>(3, 4));
}
