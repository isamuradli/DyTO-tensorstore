// // Copyright 2020 The TensorStore Authors
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //      http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <future>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/context_resource_provider.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/uri_utils.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/generic_coalescing_batch_util.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"
using namespace std;

namespace tensorstore {
namespace internal_rocksdb_kvstore {

namespace jb = tensorstore::internal_json_binding;

using rocksdb::Status;
using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::kvstore::ReadResult;

struct RocksDBSpecData {
  bool create_if_missing = true;  // Create DB if it doesn't exist
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    cout << "RocksDBSpecData" << endl;
    return f(x.create_if_missing, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("create_if_missing",
                 jb::Projection<&RocksDBSpecData::create_if_missing>(
                     jb::DefaultValue([](auto* y) { *y = true; }))),
      jb::Member(DataCopyConcurrencyResource::id,
                 jb::Projection<&RocksDBSpecData::data_copy_concurrency>()));
};

class RocksDBSpec
    : public internal_kvstore::RegisteredDriverSpec<RocksDBSpec,
                                                    RocksDBSpecData> {
 public:
  static constexpr char id[] = "rocksdb";
  RocksDBSpec() { std::cout << "RocksDBSpec" << endl; }

  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<std::string> ToUrl(std::string_view path) const override {
    return absl::StrCat(id, "://", internal::PercentEncodeUriPath(path));
  }
};

class RocksDB
    : public internal_kvstore::RegisteredDriver<RocksDB, RocksDBSpec> {
 public:
  RocksDB() { std::cout << "RocksDB" << endl; }
  ~RocksDB() override {
    cout << "Closing the connection" << endl;
    Close();
  }  // Close the DB when the driver is destroyed

  internal_kvstore_batch::CoalescingOptions GetBatchReadCoalescingOptions()
      const {
    internal_kvstore_batch::CoalescingOptions options;
    options.max_extra_read_bytes = 4095;
    options.target_coalesced_size = 128 * 1024 * 1024;
    return options;
  }

  absl::Status OpenDB();
  void Close();

  Future<ReadResult> Read(Key key, ReadOptions options) override;
  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override;

  absl::Status GetBoundSpecData(RocksDBSpecData& spec) const {
    spec = spec_;
    return absl::OkStatus();
  }

  const Executor& executor() const {
    return spec_.data_copy_concurrency->executor;
  }

  // std::unique_ptr<rocksdb::DB> db_;
  rocksdb::DB* db_;
  RocksDBSpecData spec_;
  mutable absl::Mutex mu_;  // Mutual exclusion for the DB operations

  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);
  absl::Status WriteToRocksDB(const std::string key, std::optional<Value> value,
                              Promise<TimestampedStorageGeneration> promise);
};

// struct ReadTask : public internal::AtomicReferenceCount<ReadTask> {
//   IntrusivePtr<RocksDB> rockdb_ptr;
//   std::string key_name;
//   kvstore::ReadOptions options;
//   Promise<kvstore::ReadResult> promise;

//   ReadTask(IntrusivePtr<RocksDB> rockdb_ptr, std::string key_name,
//            kvstore::ReadOptions options, Promise<kvstore::ReadResult>
//            promise)
//       : rockdb_ptr(std::move(rockdb_ptr)),
//         key_name(std::move(key_name)),
//         options(std::move(options)),
//         promise(std::move(promise)) {}
// };
// // Status
absl::Status RocksDB::OpenDB() {
  rocksdb::Options options;
  options.create_if_missing = spec_.create_if_missing;
  std::string db_path = "testdb";
  Status status = rocksdb::DB::Open(options, db_path, &db_);

  if (!status.ok()) {
    return absl::InternalError(
        absl::StrCat("RocksDB::Open failed: ", status.ToString()));
  }
  return absl::OkStatus();
}

void RocksDB::Close() {
  absl::MutexLock lock(&mu_);
  if (db_) {
    auto s = db_->Close();
    if (!s.ok()) {
      std::cerr << "Error closing RocksDB database: " << s.ToString();
    }
    delete db_;
  }

  std::cout << "DB Closed" << std::endl;
}

Future<kvstore::DriverPtr> RocksDBSpec::DoOpen() const {
  std::cout << "RocksDB::DoOpen Creating pointer to RocksDB" << endl;
  auto driver = internal::MakeIntrusivePtr<RocksDB>();
  driver->spec_ = data_;
  cout << "RocksDBSpec::DoOpen Done" << endl;
  auto status = driver->OpenDB();  // Try to open DB immediately. Errors will be
                                   // caught by the caller.
  return driver;
}

Future<ReadResult> RocksDB::Read(Key key, ReadOptions options) {
  std::cout << "Entering RocksDB::Read " << std::endl;
  absl::Cord value;
  std::string result_value;

  cout << "Key is  " << key << endl;
  std::cout << "Exiting Rocksdb::Read" << std::endl;

  // return std::move(future);

  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> RocksDB::ReadImpl(Key&& key,
                                              ReadOptions&& options) {
  std::cout << "Inside ReadImpl()." << endl;
  std::cout << "Looking for value of key: " << key << endl;
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  if (!db_) {
    promise.SetResult(absl::InternalError("Database not open"));
  }

  std::string value;
  absl::Time time_ = absl::Now();
  std::cout << "db->Get is calling..." << std::endl;
  StorageGeneration generation;
  Status s = db_->Get(rocksdb::ReadOptions(), key, &value);
  if (s.ok()) {
    promise.SetResult(ReadResult::Value(
        absl::Cord(value), TimestampedStorageGeneration{
                               generation.Unknown(),
                               time_}));  // Generation not tracked in RocksDB
    std::cout << "Found key" << std::endl;
  } else if (s.IsNotFound()) {
    promise.SetResult(ReadResult::Missing(absl::Now()));
    std::cout << "Key not found" << std::endl;
  } else {
    promise.SetResult(absl::InternalError(
        absl::StrCat("RocksDB Read failed: ", s.ToString())));
    std::cout << "Error occurred reading from DB: " << s.ToString() << endl;
  }
  std::cout
      << "Returning future to HandleBatchRequestByGenericByteRangeCoalescing"
      << std::endl;
  return std::move(future);
}  // namespace internal_rocksdb_kvstore

Future<TimestampedStorageGeneration> RocksDB::Write(Key key,
                                                    std::optional<Value> value,
                                                    WriteOptions options) {
  std::cout << "Inside RocksDB::Write" << endl;
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();

  absl::Status status = WriteToRocksDB(key, value, std::move(promise));
  if (status.ok()) {
    std::cout << "Write successful" << endl;
  } else {
    std::cout << "Error occurred writing to DB: " << status.ToString() << endl;
  }
  return std::move(future);
}

absl::Status RocksDB::WriteToRocksDB(
    const std::string key, std::optional<Value> value,
    Promise<TimestampedStorageGeneration> promise) {
  if (!db_) {
    std::cout << "No pointer to db! " << endl;
  }

  std::cout << "Writing to RocksDB: Key: " << key << " Value: " << value.value()
            << endl;
  // std::string value_ = absl::StrCat(value.value());
  std::ostringstream oss;
  oss << value.value();
  std::string value_ = oss.str();
  Status status = db_->Put(rocksdb::WriteOptions(), key, value_);
  if (!status.ok()) {
    promise.SetResult(absl::InternalError(
        absl::StrCat("WriteToRocksDB failed: ", status.ToString())));
    std::cout << "Error occurred writing to DB: " << status.ToString() << endl;
    return absl::InternalError(
        absl::StrCat("WriteToRocksDB failed: ", status.ToString()));
  } else {
    std::cout << "Sending back Write Promise" << endl;
    promise.SetResult(TimestampedStorageGeneration{});
    // add time stamp
  }
  return absl::OkStatus();
}

}  // namespace internal_rocksdb_kvstore
}  // namespace tensorstore
   // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_rocksdb_kvstore::RocksDB)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_rocksdb_kvstore::RocksDBSpec>
    registration;
}  // namespace