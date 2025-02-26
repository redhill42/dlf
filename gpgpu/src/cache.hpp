
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the caching functionality of compiled binaries and programs.
//
// =================================================================================================

#ifndef GPGPU_BLAS_CACHE_H_
#define GPGPU_BLAS_CACHE_H_

#include <string>
#include <mutex>
#include <map>

#include "utilities/utilities.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

// The generic thread-safe cache. We assume that the Key may be a heavyweight struct that is not
// normally used by the caller, while the Value is either lightweight or ref-counted.
// Hence, searching by non-Key is supported (if there is a corresponding operator<()), and
// on Store() the Key instance is moved from the caller (because it will likely be constructed
// as temporary at the time of Store()).
template <typename Key, typename Value>
class Cache {
public:
  // Cached object is returned by-value to avoid racing with Invalidate().
  // Due to lack of std::optional<>, in case of a cache miss we return a default-constructed
  // Value and set the flag to false.
  template <typename U>
  Value Get(const U& key, bool* in_cache) const;

  // We do not return references to just stored object to avoid racing with Invalidate().
  // Caller is expected to store a temporary.
  void Store(Key&& key, Value value);

  template <typename Creator>
  Value StoreIfAbsent(Key&& key, Creator&& creator) {
      std::lock_guard<std::mutex> lock(cache_mutex_);

      auto it = cache_.find(key);
      if (it != cache_.end()) {
          return it->second;
      }

      auto value = creator();
      cache_.emplace(std::move(key), creator());
      return value;
  }

  void Invalidate();

  // Removes all entries with a given key
  void Remove(const Key& key);

  static Cache<Key, Value>& Instance();

private:
  std::map<Key, Value, std::less<void>> cache_;
  mutable std::mutex cache_mutex_;

  static Cache<Key, Value> instance_;
}; // class Cache

// =================================================================================================

// The key struct for the cache of compiled OpenCL binaries (device name and platform-dependent)
// Order of fields: precision, routine_name, device_name (smaller fields first)
typedef std::tuple<PlatformID, Precision, std::string, std::string> BinaryKey;
typedef std::tuple<const PlatformID&, const Precision&, const std::string&, const std::string&> BinaryKeyRef;

typedef Cache<BinaryKey, std::string> BinaryCache;

extern template class Cache<BinaryKey, std::string>;
extern template std::string BinaryCache::Get(const BinaryKeyRef&, bool*) const;

// =================================================================================================

// The key struct for the cache of compiled OpenCL programs (context-dependent)
// Order of fields: context, device_id, precision, routine_name (smaller fields first)
typedef std::tuple<ContextID, DeviceID, Precision, std::string> ProgramKey;
typedef std::tuple<const ContextID&, const DeviceID&, const Precision&, const std::string&> ProgramKeyRef;

typedef Cache<ProgramKey, Program> ProgramCache;

extern template class Cache<ProgramKey, Program>;
extern template Program ProgramCache::Get(const ProgramKeyRef&, bool*) const;

// =================================================================================================

class Database;

// The key struct for the cache of database maps.
// Order of fields: platform_id, device_id, precision, kernel_name (smaller fields first)
typedef std::tuple<PlatformID, DeviceID, Precision, std::string> DatabaseKey;
typedef std::tuple<const PlatformID&, const DeviceID&, const Precision&, const std::string&> DatabaseKeyRef;

typedef Cache<DatabaseKey, Database> DatabaseCache;

extern template class Cache<DatabaseKey, Database>;
extern template Database DatabaseCache::Get(const DatabaseKeyRef&, bool*) const;

// =================================================================================================
}} // namespace gpgpu::blas

// GPGPU_BLAS_CACHE_H_
#endif
