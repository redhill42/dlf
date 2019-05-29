
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements all the basic functionality for the BLAS routines. This class serves as a
// base class for the actual routines (e.g. Xaxpy, Xgemm). It contains common functionality such as
// compiling the OpenCL kernel, connecting to the database, etc.
//
// =================================================================================================

#ifndef GPGPU_BLAS_ROUTINE_H_
#define GPGPU_BLAS_ROUTINE_H_

#include <string>
#include <vector>
#include <unordered_map>

#include "utilities/utilities.hpp"
#include "cache.hpp"
#include "utilities/buffer_test.hpp"
#include "database/database.hpp"
#include "routines/common.hpp"

namespace gpgpu::blas {
// =================================================================================================

// See comment at top of file for a description of the class
class Routine {
public:

  // Initializes db_, fetching cached database or building one
  static void InitDatabase(
      const Device& device,
      const std::vector<std::string>& kernel_names,
      const Precision precision,
      const std::vector<database::DatabaseEntry>& userDatabase,
      Databases& db)
  {
    const auto platform_id = device.id();
    for (const auto& kernel_name : kernel_names) {
      // Queries the cache to see whether or not the kernel parameter database is already there
      bool has_db;
      db(kernel_name) = DatabaseCache::Instance().Get(
          DatabaseKeyRef{platform_id, device.id(), precision, kernel_name},
          &has_db);
      if (has_db) { continue; }

      // Builds the parameter database for this device and routine set and stores it in the cache
      log_debug("Searching database for kernel '" + kernel_name + "'");
      db(kernel_name) = Database(device, kernel_name, precision, userDatabase);
      DatabaseCache::Instance().Store(
          DatabaseKey{platform_id, device.id(), precision, kernel_name},
          Database{db(kernel_name)});
    }
  }

  // Base class constructor. The user database is an optional extra database to override the
  // built-in database.
  // All heavy preparation work is done inside this constructor.
  // NOTE: the caller must provide the same userDatabase for each combination of device, precision
  // and routine list, otherwise the caching logic will break.
  explicit Routine(const Queue& queue, Event* event, const std::string& name,
                   const std::vector<std::string>& routines, const Precision precision,
                   const std::vector<database::DatabaseEntry>& userDatabase,
                   std::initializer_list<const char*> source);

private:

  // Initializes program_, fetching cached program or building one
  void InitProgram(std::initializer_list<const char *> source);

protected:

  // Non-static variable for the precision
  const Precision precision_;

  // The routine's name and the corresponding kernels
  const std::string routine_name_;
  const std::vector<std::string> kernel_names_;

  // The OpenCL objects, accessible only from derived classes
  Queue queue_;
  Event* event_;
  Context context_;
  Device device_;

  // Compiled program (either retrieved from cache or compiled in slow path)
  Program program_;

  // Connection to the database for all the device-specific parameters
  Databases db_;
};

// =================================================================================================
} // namespace gpgpu::blas

// GPGPU_BLAS_ROUTINE_H_
#endif
