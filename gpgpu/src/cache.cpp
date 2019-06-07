
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

#include <string>
#include <vector>
#include <mutex>

#include "database/database.hpp"
#include "cache.hpp"

namespace gpgpu { namespace blas {
// =================================================================================================

template <typename Key, typename Value>
template <typename U>
Value Cache<Key, Value>::Get(const U& key, bool* in_cache) const {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (it == cache_.end()) {
    if (in_cache)
      *in_cache = false;
    return Value();
  }

  if (in_cache)
    *in_cache = true;
  return it->second;
}

template <typename Key, typename Value>
void Cache<Key, Value>::Store(Key&& key, Value value) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.emplace(std::move(key), std::move(value));
}

template <typename Key, typename Value>
void Cache<Key, Value>::Remove(const Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.erase(key);
}

template <typename Key, typename Value>
void Cache<Key, Value>::Invalidate() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

template <typename Key, typename Value>
Cache<Key, Value>& Cache<Key, Value>::Instance() {
  return instance_;
}

template <typename Key, typename Value>
Cache<Key, Value> Cache<Key, Value>::instance_;

// =================================================================================================

template class Cache<BinaryKey, std::string>;
template std::string BinaryCache::Get(const BinaryKeyRef&, bool*) const;

// =================================================================================================

template class Cache<ProgramKey, Program>;
template Program ProgramCache::Get(const ProgramKeyRef&, bool*) const;

// =================================================================================================

template class Cache<DatabaseKey, Database>;
template Database DatabaseCache::Get(const DatabaseKeyRef&, bool*) const;

// =================================================================================================
}} // namespace gpgpu::blas
