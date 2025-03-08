#ifndef TENSORSTORE_KVSTORE_ROCKSDB_VALIDATE_H_
#define TENSORSTORE_KVSTORE_ROCKSDB_VALIDATE_H_

#include <string_view>

#include "tensorstore/kvstore/generation.h"

using ::tensorstore::StorageGeneration;

namespace tensorstore {
namespace internal_storage_rocksdb {

bool IsValidStorageGeneration(const StorageGeneration& gen);

}
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_ROCKSDB_VALIDATE_H_