#pragma once

#include <stdint.h>
#include <string>

#define FORALL_BUILTIN_SYMBOLS(_) \
  _(Abs)                          \
  _(Acos)                         \
  _(Acosh)                        \
  _(Add)                          \
  _(And)                          \
  _(ArgMax)                       \
  _(ArgMin)                       \
  _(Asin)                         \
  _(Asinh)                        \
  _(Atan)                         \
  _(Atanh)                        \
  _(AveragePool)                  \
  _(BatchNormalization)           \
  _(BitShift)                     \
  _(Captured)                     \
  _(Cast)                         \
  _(Ceil)                         \
  _(Clip)                         \
  _(Concat)                       \
  _(Compress)                     \
  _(Constant)                     \
  _(ConstantOfShape)              \
  _(Conv)                         \
  _(ConvInteger)                  \
  _(ConvTranspose)                \
  _(Cos)                          \
  _(Cosh)                         \
  _(DepthToSpace)                 \
  _(DequantizeLinear)             \
  _(Div)                          \
  _(Dropout)                      \
  _(Elu)                          \
  _(Equal)                        \
  _(Erf)                          \
  _(Exp)                          \
  _(Expand)                       \
  _(EyeLike)                      \
  _(Flatten)                      \
  _(Floor)                        \
  _(GRU)                          \
  _(Gather)                       \
  _(Gemm)                         \
  _(GlobalAveragePool)            \
  _(GlobalLpPool)                 \
  _(GlobalMaxPool)                \
  _(Greater)                      \
  _(HardSigmoid)                  \
  _(Hardmax)                      \
  _(Identity)                     \
  _(If)                           \
  _(InstanceNormalization)        \
  _(IsInf)                        \
  _(IsNaN)                        \
  _(LRN)                          \
  _(LSTM)                         \
  _(LeakyRelu)                    \
  _(Less)                         \
  _(Log)                          \
  _(LogSoftmax)                   \
  _(Loop)                         \
  _(LpNormalization)              \
  _(LpPool)                       \
  _(MatMul)                       \
  _(MatMulInteger)                \
  _(Max)                          \
  _(MaxPool)                      \
  _(MaxRoiPool)                   \
  _(MaxUnpool)                    \
  _(Mean)                         \
  _(Min)                          \
  _(Mod)                          \
  _(Mul)                          \
  _(Multinomial)                  \
  _(Neg)                          \
  _(NonMaxSuppression)            \
  _(NonZero)                      \
  _(Not)                          \
  _(OneHot)                       \
  _(Or)                           \
  _(PRelu)                        \
  _(Pad)                          \
  _(Param)                        \
  _(Pow)                          \
  _(QLinearConv)                  \
  _(QLinearMatMul)                \
  _(QuantizeLinear)               \
  _(RNN)                          \
  _(RandomNormal)                 \
  _(RandomNormalLike)             \
  _(RandomUniform)                \
  _(RandomUniformLike)            \
  _(Reciprocal)                   \
  _(ReduceL1)                     \
  _(ReduceL2)                     \
  _(ReduceLogSum)                 \
  _(ReduceLogSumExp)              \
  _(ReduceMax)                    \
  _(ReduceMean)                   \
  _(ReduceMin)                    \
  _(ReduceProd)                   \
  _(ReduceSum)                    \
  _(ReduceSumSquare)              \
  _(Relu)                         \
  _(Reshape)                      \
  _(Resize)                       \
  _(Return)                       \
  _(ReverseSequence)              \
  _(RoiAlign)                     \
  _(Round)                        \
  _(Scan)                         \
  _(Scatter)                      \
  _(Selu)                         \
  _(Shape)                        \
  _(Shrink)                       \
  _(Sigmoid)                      \
  _(Sign)                         \
  _(Sin)                          \
  _(Sinh)                         \
  _(Size)                         \
  _(Slice)                        \
  _(Softmax)                      \
  _(Softplus)                     \
  _(Softsign)                     \
  _(SpaceToDepth)                 \
  _(Select)                       \
  _(Split)                        \
  _(Sqrt)                         \
  _(Squeeze)                      \
  _(StringNormalizer)             \
  _(Sub)                          \
  _(Sum)                          \
  _(Tan)                          \
  _(Tanh)                         \
  _(TfldfVectorizer)              \
  _(ThresholdedRelu)              \
  _(Tile)                         \
  _(TopK)                         \
  _(Transpose)                    \
  _(Undefined)                    \
  _(Unsequeeze)                   \
  _(Upsample)                     \
  _(Where)                        \
  _(Xor)                          \
                                  \
  _(abs)                          \
  _(acos)                         \
  _(add)                          \
  _(alpha)                        \
  _(asin)                         \
  _(atan)                         \
  _(atan2)                        \
  _(auto_pad)                     \
  _(axes)                         \
  _(axis)                         \
  _(beta)                         \
  _(bias)                         \
  _(body)                         \
  _(broadcast)                    \
  _(cat)                          \
  _(ceil)                         \
  _(ceil_mode)                    \
  _(chunk)                        \
  _(clamp)                        \
  _(consumed_inputs)              \
  _(cos)                          \
  _(cosh)                         \
  _(count_include_pad)            \
  _(device)                       \
  _(dilation)                     \
  _(dilations)                    \
  _(dim)                          \
  _(div)                          \
  _(else_branch)                  \
  _(epsilon)                      \
  _(eq)                           \
  _(equal)                        \
  _(expand)                       \
  _(expm1)                        \
  _(exponent)                     \
  _(floor)                        \
  _(fmod)                         \
  _(frac)                         \
  _(ge)                           \
  _(group)                        \
  _(gt)                           \
  _(inplace)                      \
  _(is_test)                      \
  _(keepdims)                     \
  _(kernel)                       \
  _(kernels)                      \
  _(kernel_shape)                 \
  _(lambd)                        \
  _(le)                           \
  _(lerp)                         \
  _(lgamma)                       \
  _(log1p)                        \
  _(lt)                           \
  _(max)                          \
  _(min)                          \
  _(mode)                         \
  _(momentum)                     \
  _(mul)                          \
  _(ne)                           \
  _(neg)                          \
  _(ones)                         \
  _(order)                        \
  _(other)                        \
  _(pad)                          \
  _(pads)                         \
  _(perm)                         \
  _(pow)                          \
  _(ratio)                        \
  _(reciprocal)                   \
  _(remainder)                    \
  _(round)                        \
  _(rsqrt)                        \
  _(scale)                        \
  _(scales)                       \
  _(shape)                        \
  _(sigmoid)                      \
  _(sin)                          \
  _(sinh)                         \
  _(size)                         \
  _(split)                        \
  _(storage_order)                \
  _(stride)                       \
  _(strides)                      \
  _(sub)                          \
  _(tan)                          \
  _(tanh)                         \
  _(then_branch)                  \
  _(to)                           \
  _(transA)                       \
  _(transB)                       \
  _(trunc)                        \
  _(value)                        \
  _(zeros)

namespace dlf { namespace model {

enum BuiltinSymbol {
#define DEFINE_SYMBOL(s) k##s,
    FORALL_BUILTIN_SYMBOLS(DEFINE_SYMBOL)
#undef DEFINE_SYMBOL
    kLastSymbol, // where we start counting for new symbols
};

class Symbol {
    uint32_t m_value;

public:
    Symbol() {}
    /*implicit*/ constexpr Symbol(BuiltinSymbol value) : m_value(value) {}
    /*implicit*/ Symbol(const std::string& s);
    /*implicit*/ Symbol(const char* s) : Symbol(std::string(s)) {}

    constexpr operator uint32_t() const noexcept { return m_value; }
    constexpr uint32_t val() const noexcept { return m_value; }
    const char* str() const noexcept;
};

inline constexpr bool operator==(Symbol lhs, Symbol rhs) noexcept {
    return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}

// necessary to prevent ambiguous overload resolutions
inline constexpr bool operator==(BuiltinSymbol lhs, Symbol rhs) noexcept {
    return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}

inline constexpr bool operator==(Symbol lhs, BuiltinSymbol rhs) noexcept {
    return static_cast<uint32_t>(lhs) == static_cast<uint32_t>(rhs);
}

inline Symbol operator""_sym(const char* s, size_t) noexcept {
    return Symbol(s);
}

}} // namespace dlf::model

// make symbol behave like an integer in hash tables
namespace std {
template <>
struct hash<dlf::model::Symbol> {
    std::size_t operator()(dlf::model::Symbol s) const noexcept {
        return std::hash<uint32_t>()(static_cast<uint32_t>(s));
    }
};

} // namespace std
