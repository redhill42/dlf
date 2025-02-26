
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Routine base class (see the header for information about the class).
//
// =================================================================================================

#include <string>
#include <vector>
#include <cstdlib>

#include "routine.hpp"

namespace gpgpu { namespace blas {

// The constructor does all heavy work, errors are returned as exceptions
Routine::Routine(const Queue& queue, Event* event, const std::string& name,
                 const std::vector<std::string>& kernel_names, const Precision precision,
                 const std::vector<database::DatabaseEntry>& userDatabase,
                 std::initializer_list<const char*> source) :
    precision_(precision),
    routine_name_(name),
    kernel_names_(kernel_names),
    queue_(queue),
    event_(event),
    context_(queue_.context()),
    device_(context_.device()),
    db_(kernel_names)
{
    InitDatabase(device_, kernel_names, precision, userDatabase, db_);
    InitProgram(source);
}

void Routine::InitProgram(std::initializer_list<const char*> source) {
    const auto context_id = context_.id();
    const auto device_id = device_.id();

    // Determines the identifier for this particular routine call
    auto routine_info = routine_name_;
    for (const auto& kernel_name : kernel_names_) {
        routine_info += "_" + kernel_name + db_(kernel_name).GetValuesString();
    }

    // Queries the cache to see whether or not the program (context-specific)
    // is already there
    bool has_program;
    program_ = ProgramCache::Instance().Get(
        ProgramKeyRef{context_id, device_id, precision_, routine_info},
        &has_program);
    if (has_program) return;

    // Sets the build options from an environment variable (if set)
    auto options = std::vector<std::string>();
    const auto environment_variable = std::getenv("GPGPU_BUILD_OPTIONS");
    if (environment_variable != nullptr) {
        options.push_back(environment_variable);
    }

    // Queries the cache to see whether or not the binary (device-specific)
    // is already there. If it is, a program is created and stored in the cache.
    const auto device_name = GetDeviceName(device_);
    const auto platform_id = device_.platform().id();
    bool has_binary;
    auto binary = BinaryCache::Instance().Get(
        BinaryKeyRef{platform_id, precision_, routine_info, device_name},
        &has_binary);
    if (has_binary) {
        program_ = context_.loadProgram(binary);
        ProgramCache::Instance().Store(
            ProgramKey{context_id, device_id, precision_, routine_info},
            program_);
        return;
    }

    // Otherwise, the kernel will be compiled and program will be built.
    // Both the binary and the program will be added to the cache.

    // Inspects whether or not FP64 is supported in case of double precision
    if ((precision_ == Precision::Double && !PrecisionSupported<double>(device_)) ||
        (precision_ == Precision::ComplexDouble && !PrecisionSupported<double2>(device_))) {
        throw RuntimeErrorCode(StatusCode::kNoDoublePrecision);
    }

    // As above, but for FP16 (half precision)
    if (precision_ == Precision::Half && !PrecisionSupported<half>(device_)) {
        throw RuntimeErrorCode(StatusCode::kNoHalfPrecision);
    }

    // Collects the parameters for this device in the form of defines
    auto source_string = std::string();
    for (const auto& kernel_name : kernel_names_) {
        source_string += db_(kernel_name).GetDefines();
    }

    // Adds routine-specific code to the constructed source string
    for (const char* s : source) {
        source_string += s;
    }

    // Completes the source and compiles the kernel
    program_ = CompileFromSource(source_string, precision_, routine_name_,
                                 device_, context_, options, 0);

    // Store the compiled binary and program in the cache
    BinaryCache::Instance().Store(
        BinaryKey{platform_id, precision_, routine_info, device_name},
        program_.getIR());

    ProgramCache::Instance().Store(
        ProgramKey{context_id, device_id, precision_, routine_info},
        program_);
}

}} // namespace gpgpu::blas
