#pragma once

#include <stdint.h>
#include <string>

#define FORALL_OPERATORS(_) \
  _(Abs)                    \
  _(Acos)                   \
  _(Acosh)                  \
  _(Add)                    \
  _(And)                    \
  _(ArgMax)                 \
  _(ArgMin)                 \
  _(Asin)                   \
  _(Asinh)                  \
  _(Atan)                   \
  _(Atanh)                  \
  _(AveragePool)            \
  _(BatchNormalization)     \
  _(BitShift)               \
  _(Cast)                   \
  _(Ceil)                   \
  _(Clip)                   \
  _(Compress)               \
  _(Concat)                 \
  _(Constant)               \
  _(ConstantOfShape)        \
  _(Conv)                   \
  _(ConvInteger)            \
  _(ConvTranspose)          \
  _(Cos)                    \
  _(Cosh)                   \
  _(CumSum)                 \
  _(DepthToSpace)           \
  _(Div)                    \
  _(Dropout)                \
  _(Elu)                    \
  _(Equal)                  \
  _(Erf)                    \
  _(Exp)                    \
  _(Expand)                 \
  _(EyeLike)                \
  _(Flatten)                \
  _(Floor)                  \
  _(GRU)                    \
  _(Gather)                 \
  _(GatherElements)         \
  _(GatherND)               \
  _(Gemm)                   \
  _(GlobalAveragePool)      \
  _(GlobalLpPool)           \
  _(GlobalMaxPool)          \
  _(Greater)                \
  _(HardSigmoid)            \
  _(Hardmax)                \
  _(Identity)               \
  _(If)                     \
  _(InstanceNormalization)  \
  _(IsInf)                  \
  _(IsNaN)                  \
  _(LRN)                    \
  _(LSTM)                   \
  _(LeakyRelu)              \
  _(Less)                   \
  _(Log)                    \
  _(LogSoftmax)             \
  _(Loop)                   \
  _(LpNormalization)        \
  _(LpPool)                 \
  _(MatMul)                 \
  _(MatMulInteger)          \
  _(Max)                    \
  _(MaxPool)                \
  _(MaxRoiPool)             \
  _(MaxUnpool)              \
  _(Mean)                   \
  _(Min)                    \
  _(Mod)                    \
  _(Mul)                    \
  _(Multinomial)            \
  _(Neg)                    \
  _(NonMaxSuppression)      \
  _(NonZero)                \
  _(Not)                    \
  _(OneHot)                 \
  _(Or)                     \
  _(PRelu)                  \
  _(Pad)                    \
  _(Pow)                    \
  _(QLinearConv)            \
  _(QLinearMatMul)          \
  _(RNN)                    \
  _(RandomNormal)           \
  _(RandomNormalLike)       \
  _(RandomUniform)          \
  _(RandomUniformLike)      \
  _(Reciprocal)             \
  _(ReduceL1)               \
  _(ReduceL2)               \
  _(ReduceLogSum)           \
  _(ReduceLogSumExp)        \
  _(ReduceMax)              \
  _(ReduceMean)             \
  _(ReduceMin)              \
  _(ReduceProd)             \
  _(ReduceSum)              \
  _(ReduceSumSquare)        \
  _(Relu)                   \
  _(Reshape)                \
  _(Resize)                 \
  _(ReverseSequence)        \
  _(RoiAlign)               \
  _(Round)                  \
  _(Scan)                   \
  _(Scatter)                \
  _(ScatterElements)        \
  _(ScatterND)              \
  _(Selu)                   \
  _(Shape)                  \
  _(Shrink)                 \
  _(Sigmoid)                \
  _(Sign)                   \
  _(Sin)                    \
  _(Sinh)                   \
  _(Size)                   \
  _(Slice)                  \
  _(Softmax)                \
  _(Softplus)               \
  _(Softsign)               \
  _(SpaceToDepth)           \
  _(Split)                  \
  _(Sqrt)                   \
  _(Squeeze)                \
  _(StringNormalizer)       \
  _(Sub)                    \
  _(Sum)                    \
  _(Tan)                    \
  _(Tanh)                   \
  _(TfIdfVectorizer)        \
  _(ThresholdedRelu)        \
  _(Tile)                   \
  _(TopK)                   \
  _(Transpose)              \
  _(Unsqueeze)              \
  _(Where)                  \
  _(Xor)

#define FORALL_ATTRIBUTES(_)    \
  _(abs)                        \
  _(acos)                       \
  _(activation_alpha)           \
  _(activation_beta)            \
  _(activations)                \
  _(add)                        \
  _(alpha)                      \
  _(asin)                       \
  _(atan)                       \
  _(atan2)                      \
  _(auto_pad)                   \
  _(axes)                       \
  _(axis)                       \
  _(batch_axis)                 \
  _(beta)                       \
  _(bias)                       \
  _(blocksize)                  \
  _(body)                       \
  _(broadcast)                  \
  _(case_change_action)         \
  _(cat)                        \
  _(ceil)                       \
  _(ceil_mode)                  \
  _(center_point_box)           \
  _(chunk)                      \
  _(clamp)                      \
  _(clip)                       \
  _(consumed_inputs)            \
  _(cos)                        \
  _(cosh)                       \
  _(count_include_pad)          \
  _(detect_positive)            \
  _(detect_negative)            \
  _(device)                     \
  _(dilation)                   \
  _(dilations)                  \
  _(dim)                        \
  _(direction)                  \
  _(div)                        \
  _(dtype)                      \
  _(else_branch)                \
  _(epsilon)                    \
  _(eq)                         \
  _(equal)                      \
  _(exclusive)                  \
  _(expand)                     \
  _(expm1)                      \
  _(exponent)                   \
  _(floor)                      \
  _(fmod)                       \
  _(frac)                       \
  _(gamma)                      \
  _(ge)                         \
  _(group)                      \
  _(gt)                         \
  _(hidden_size)                \
  _(high)                       \
  _(inplace)                    \
  _(input_forget)               \
  _(is_case_sensitive)          \
  _(is_test)                    \
  _(keepdims)                   \
  _(kernel)                     \
  _(kernels)                    \
  _(kernel_shape)               \
  _(lambd)                      \
  _(le)                         \
  _(lerp)                       \
  _(linear_before_reset)        \
  _(lgamma)                     \
  _(locale)                     \
  _(log1p)                      \
  _(low)                        \
  _(lt)                         \
  _(max)                        \
  _(max_gram_length)            \
  _(max_skip_count)             \
  _(mean)                       \
  _(min)                        \
  _(min_gram_length)            \
  _(min_skip_count)             \
  _(mode)                       \
  _(momentum)                   \
  _(mul)                        \
  _(ne)                         \
  _(neg)                        \
  _(ngram_counts)               \
  _(ngram_indexes)              \
  _(num_scan_inputs)            \
  _(ones)                       \
  _(order)                      \
  _(other)                      \
  _(output_height)              \
  _(output_padding)             \
  _(output_shape)               \
  _(output_width)               \
  _(pad)                        \
  _(pads)                       \
  _(perm)                       \
  _(pool_int64s)                \
  _(pool_strings)               \
  _(pooled_shape)               \
  _(pow)                        \
  _(ratio)                      \
  _(reciprocal)                 \
  _(remainder)                  \
  _(reverse)                    \
  _(round)                      \
  _(rsqrt)                      \
  _(sample_size)                \
  _(sampling_ratio)             \
  _(scale)                      \
  _(scales)                     \
  _(scan_input_axes)            \
  _(scan_input_directions)      \
  _(scan_output_axes)           \
  _(scan_output_directions)     \
  _(seed)                       \
  _(shape)                      \
  _(sigmoid)                    \
  _(sin)                        \
  _(sinh)                       \
  _(size)                       \
  _(spatial_scale)              \
  _(split)                      \
  _(stopwords)                  \
  _(storage_order)              \
  _(stride)                     \
  _(strides)                    \
  _(sub)                        \
  _(tan)                        \
  _(tanh)                       \
  _(then_branch)                \
  _(time_axis)                  \
  _(to)                         \
  _(transA)                     \
  _(transB)                     \
  _(trunc)                      \
  _(value)                      \
  _(weights)                    \
  _(zeros)

#define FORALL_BUILTIN_SYMBOLS(_) \
  _(Param)                        \
  _(Return)                       \
  _(Undefined)                    \
  _(Captured)                     \
  FORALL_OPERATORS(_)             \
  FORALL_ATTRIBUTES(_)

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
