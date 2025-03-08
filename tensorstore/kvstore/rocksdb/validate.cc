#include "tensorstore/kvstore/rocksdb/validate.h"

#include <string_view>

#include "absl/strings/match.h"
#include "tensorstore/kvstore/generation.h"

namespace tensorstore {
namespace internal_storage_rocksdb {

bool IsValidStorageGeneration(const StorageGeneration& gen) {
  if (StorageGeneration::IsUnknown(gen) || StorageGeneration::IsNoValue(gen))
    return true;

  return false;
}

}  // namespace internal_storage_rocksdb
}  // namespace tensorstore