
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file provides overrides for Apple's OpenCL CPU implementation. It is a special case compared
// to all other implementations, as it only supports a 1-dimensional work-group size. In addition,
// that work-group size is limited to 1024 (in theory) or much lower (kernel resource dependent).
// Thus, instead of supporting this corner-case in the whole regular flow (starting from the tuner),
// we provide this file with some manual overrides.
//
// Note: These overrides are to make the Apple CPU work and not crash, they are not in any way
// optimized parameters. For decent speed don't use Apple's OpenCL CPU implementation.
//
// =================================================================================================

namespace gpgpu::blas {
namespace database {
// =================================================================================================

const DatabaseEntry XaxpyApple = {
  "Xaxpy", Precision::Any, {"VW", "WGS", "WPT"}, { { kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 8, 1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XdotApple = {
  "Xdot", Precision::Any, {"WGS1", "WGS2"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XgemvApple = {
  "Xgemv", Precision::Any, {"WGS1", "WPT1", "UNROLL1"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XgemvFastApple = {
  "XgemvFast", Precision::Any, {"VW2", "WGS2", "WPT2"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XgemvFastRotApple = {
  "XgemvFastRot", Precision::Any, {"VW3", "WGS3", "WPT3"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XgerApple = {
  "Xger", Precision::Any, {"WGS1", "WGS2", "WPT"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 64, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XtrsvApple = {
  "Xtrsv", Precision::Any, {"TRSV_BLOCK_SIZE"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry XgemmApple = {
  "Xgemm", Precision::Any, {"GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 } } } } } } }
};
const DatabaseEntry XgemmDirectApple = {
  "XgemmDirect", Precision::Any, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry CopyApple = {
  "Copy", Precision::Any, {"COPY_DIMX", "COPY_DIMY", "COPY_VW", "COPY_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry PadApple = {
  "Pad", Precision::Any, {"PAD_DIMX", "PAD_DIMY", "PAD_WPTX", "PAD_WPTY"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry TransposeApple = {
  "Transpose", Precision::Any, {"TRA_DIM", "TRA_PAD", "TRA_SHUFFLE", "TRA_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry PadtransposeApple = {
  "Padtranspose", Precision::Any, {"PADTRA_PAD", "PADTRA_TILE", "PADTRA_WPT"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry InvertApple = {
  "Invert", Precision::Any, {"INTERNAL_BLOCK_SIZE"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};
const DatabaseEntry TrsvRoutineApple = {
  "TrsvRoutine", Precision::Any, {"TRSV_BLOCK_SIZE"}, { {  kDeviceTypeAll, "default", { { "default", { { kDeviceNameDefault, Params{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } } } } } } }
};

// =================================================================================================
} // namespace database
} // namespace gpgpu::blas
