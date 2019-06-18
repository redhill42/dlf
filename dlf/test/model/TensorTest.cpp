#include "model.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace dlf::model;

TEST(TensorData, SetData) {
    std::vector<int16_t> data(24);
    std::iota(data.begin(), data.end(), 1);
    TensorData dt({2, 3, 4}, data);
    EXPECT_EQ(dt.type(), DataType::INT16);

    auto& v = dt.int32_data();
    EXPECT_EQ(v.size(), 24);
    for (int i = 0; i < 24; i++) {
        EXPECT_EQ(v[i], i + 1);
    }
}

TEST(TensorData, SetStringData) {
    std::vector<std::string> data = {"venus", "jupiter", "mercury", "mars", "saturn"};
    TensorData dt({data.size()}, data);
    EXPECT_EQ(dt.type(), DataType::STRING);
    EXPECT_EQ(dt.string_data(), data);
}

TEST(TensorData, SetComplexData) {
    const std::complex<float> data[] {{1, 2}, {3, 4}};
    TensorData dt({2}, data);
    EXPECT_EQ(dt.type(), DataType::COMPLEX64);
    EXPECT_EQ(dt.float_data(), std::vector<float>({1, 2, 3, 4}));
}

TEST(TensorData, SetIncompatibleData) {
    TensorData dt({2, 3, 4}, DataType::INT32);
    std::vector<int16_t> data(24);
    EXPECT_ANY_THROW(dt.set_data(data.begin(), data.end()));
}

TEST(TensorData, Encode) {
    auto dt = TensorData(dlf::Tensor<int32_t>::range({2, 3, 4}, 1));
    EXPECT_EQ(dt.dims(), Dims({2, 3, 4}));
    EXPECT_EQ(dt.int32_data().size(), 24);
}

TEST(TensorData, Decode) {
    std::vector<int32_t> data(24);
    std::iota(data.begin(), data.end(), 1);
    TensorData dt({2, 3, 4}, data);
    EXPECT_EQ(dt.decode<int16_t>(), dlf::Tensor<int16_t>::range({2, 3, 4}, 1));
}

TEST(TensorData, DecodeString) {
    TensorData dt({2}, DataType::STRING);
    dt.string_data().push_back("alice");
    dt.string_data().push_back("bob");
    EXPECT_EQ(dt.decode<std::string>(), dlf::Tensor<std::string>({2}, {"alice", "bob"}));
}

TEST(TensorData, DecodeComplex) {
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
