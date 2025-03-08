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
#include "tensorstore/kvstore/rocksdb/validate.h"
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

using rocksdb::Status;
using ::tensorstore::internal::DataCopyConcurrencyResource;
using ::tensorstore::internal_storage_rocksdb::IsValidStorageGeneration;
using ::tensorstore::kvstore::ReadResult;

namespace tensorstore {
namespace {

TimestampedStorageGeneration GenerationNow(StorageGeneration generation) {
  return TimestampedStorageGeneration{std::move(generation), absl::Now()};
}

namespace jb = tensorstore::internal_json_binding;

struct RocksDBSpecData {
  bool create_if_missing = false;
  std::string database_name;
  Context::Resource<DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    cout << "RocksDBSpecData" << endl;
    return f(x.create_if_missing, x.data_copy_concurrency, x.database_name);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("create_if_missing",
                 jb::Projection<&RocksDBSpecData::create_if_missing>(
                     jb::DefaultValue([](auto* y) { *y = true; }))),
      jb::Member("database_name",
                 jb::Projection<&RocksDBSpecData::database_name>(
                     jb::DefaultValue([](auto* y) { *y = "testdb"; }))),
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
    return absl::StrCat(id, "://", path);
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
  std::atomic<uint64_t> next_generation_number_ = 0;  // Added the counter

  Future<ReadResult> ReadImpl(Key&& key, ReadOptions&& options);
  absl::Status WriteToRocksDB(const std::string key, const std::string gen_key,
                              std::optional<Value> value,
                              Promise<TimestampedStorageGeneration> promise,
                              WriteOptions options);
};

absl::Status RocksDB::OpenDB() {
  rocksdb::Options options;
  options.create_if_missing = spec_.create_if_missing;
  std::cout << "Opening DB with Name: " << spec_.database_name << endl;
  Status status = rocksdb::DB::Open(options, spec_.database_name, &db_);
  std::cout << "Status: " << status.ToString() << std::endl;
  if (!status.ok()) {
    std::cerr << "Error opening RocksDB database: " << status.ToString();
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
  std::cout << "------" << data_.create_if_missing << "-----------" << endl;
  driver->spec_ = data_;
  cout << "RocksDBSpec::DoOpen Done" << endl;
  auto status = driver->OpenDB();  // Try to open DB immediately. Errors will be
                                   // caught by the caller.

  if (!status.ok()) {
    return status;
  } else {
    return driver;
  }
}

Future<ReadResult> RocksDB::Read(Key key, ReadOptions options) {
  std::cout << "\nEntering RocksDB::Read " << std::endl;
  absl::Cord value;

  if (!IsValidStorageGeneration(options.generation_conditions.if_equal) ||
      !IsValidStorageGeneration(options.generation_conditions.if_not_equal)) {
    return absl::InvalidArgumentError("Malformed Generation");
  }

  return internal_kvstore_batch::HandleBatchRequestByGenericByteRangeCoalescing(
      *this, std::move(key), std::move(options));
}

Future<kvstore::ReadResult> RocksDB::ReadImpl(Key&& key,
                                              ReadOptions&& options) {
  std::cout << "Inside ReadImpl. ";

  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  if (!db_) {
    promise.SetResult(absl::InternalError("Database not open"));
  }

  std::string value;
  std::string gen_key = key + "_generation";  // Key to store generation
  std::string gen_number;  // stores the version of the key-value pair
  Status status = db_->Get(rocksdb::ReadOptions(), key, &value);
  Status status_gen = db_->Get(rocksdb::ReadOptions(), gen_key, &gen_number);

  std::cout << "Key is : " << key << " Value is : " << value
            << " Generation number is : " << gen_number << endl;

  // Key found
  if (status.ok() && status_gen.ok()) {
    promise.SetResult(ReadResult::Value(
        absl::Cord(value),
        GenerationNow(StorageGeneration::FromUint64(
            std::stoi(gen_number)))));  // Generation not tracked in RocksDB
    // std::cout << "Found key" << std::endl;
  }
  // Ket not found
  else if (status.IsNotFound()) {
    promise.SetResult(
        ReadResult::Missing(GenerationNow(StorageGeneration::NoValue())));
    std::cout << "Key not found" << std::endl;
  } else {
    promise.SetResult(absl::InternalError(
        absl::StrCat("RocksDB Read failed: ", status.ToString())));
    std::cout << "Error occurred reading from DB: " << status.ToString()
              << endl;
  }

  return std::move(future);
}  // namespace internal_rocksdb_kvstore

Future<TimestampedStorageGeneration> RocksDB::Write(Key key,
                                                    std::optional<Value> value,
                                                    WriteOptions options) {
  std::cout << "Inside RocksDB::Write" << endl;
  auto [promise, future] =
      PromiseFuturePair<TimestampedStorageGeneration>::Make();

  std::string gen_key = key + "_generation";  // Key to store generation

  absl::Status status =
      WriteToRocksDB(key, gen_key, value, std::move(promise), options);

  if (status.ok()) {
    std::cout << "Write successful" << endl;
  } else {
    std::cout << "Error occurred writing to DB: " << status.ToString() << endl;
  }
  return std::move(future);
}

absl::Status RocksDB::WriteToRocksDB(
    const std::string key, const std::string gen_key,
    std::optional<Value> value, Promise<TimestampedStorageGeneration> promise,
    WriteOptions options) {
  if (!db_) {
    std::cout << "No pointer to db! " << endl;
  }

  std::cout << "Writing : Key:" << key << " Value: " << value.value() << endl;
  absl::MutexLock lock(&mu_);  // Acquire mutex to protect counter

  std::string current_generation;
  auto status = db_->Get(rocksdb::ReadOptions(), gen_key, &current_generation);

  if (status.IsNotFound()) {
    // Key does not exist
    if (!options.generation_conditions.MatchesNoValue()) {
      promise.SetResult(GenerationNow(StorageGeneration::Unknown()));
      return absl::InternalError(
          absl::StrCat("Unknown generation", status.ToString()));
    }
    if (!value) {
      promise.SetResult(GenerationNow(StorageGeneration::NoValue()));
      return absl::InternalError(absl::StrCat("No value", status.ToString()));
    }

    // Converts Cord type to string
    std::ostringstream oss;
    oss << value.value();
    std::string value_str = oss.str();

    // std::string value_str = absl::Cord(*value).ToString();
    std::stringstream ss;
    uint64_t generation_number = next_generation_number_++;
    ss << generation_number;  // increment the generation number upon
                              // updating the data
    auto status =
        db_->Put(rocksdb::WriteOptions(), key, value_str);  // write value
    auto generation_status = db_->Put(rocksdb::WriteOptions(), gen_key,
                                      ss.str());  // write generation

    if (!status.ok() || !generation_status.ok()) {
      promise.SetResult(absl::InternalError(
          absl::StrCat("RocksDB Put failed: ", status.ToString())));
      return absl::InternalError(
          absl::StrCat("RocksDB Put failed: ", status.ToString()));
    }
    promise.SetResult(
        GenerationNow(StorageGeneration::FromUint64(generation_number)));
    return absl::OkStatus();
  }

  // Key exists
  if (!options.generation_conditions.Matches(
          StorageGeneration::FromUint64(std::stoi(current_generation)))) {
    promise.SetResult(GenerationNow(StorageGeneration::Unknown()));
    return absl::FailedPreconditionError("Generation Unknown");
  }

  // No value is given, delete the key
  if (!value) {
    auto status = db_->Delete(rocksdb::WriteOptions(), key);
    auto generation_status = db_->Delete(rocksdb::WriteOptions(), gen_key);

    if (!status.ok() || !generation_status.ok()) {
      promise.SetResult(absl::InternalError(
          absl::StrCat("RocksDB Delete failed: ", status.ToString())));
      return absl::InternalError(
          absl::StrCat("RocksDB Delete failed: ", status.ToString()));
    }
    promise.SetResult(GenerationNow(StorageGeneration::NoValue()));
    // return absl::OkStatus();
    return absl::InternalError(
        absl::StrCat("No value: Deleting key", status.ToString()));
  }

  // Convert Cord type to string
  std::ostringstream oss;
  oss << value.value();
  std::string value_str = oss.str();

  // std::string value_str = absl::Cord(*value).ToString();
  std::stringstream ss;
  uint64_t generation_number = next_generation_number_++;
  std::cout << "Generation number is : " << generation_number << endl;
  ss << generation_number;  // increment the generation number upon
                            // updating the data
  auto write_status =
      db_->Put(rocksdb::WriteOptions(), key, value_str);  // write value
  auto generation_status = db_->Put(rocksdb::WriteOptions(), gen_key,
                                    ss.str());  // write generation

  if (!write_status.ok() || !generation_status.ok()) {
    promise.SetResult(absl::InternalError(
        absl::StrCat("RocksDB Put failed: ", write_status.ToString())));
    return absl::InternalError(
        absl::StrCat("RocksDB Put failed: ", write_status.ToString()));
  }
  promise.SetResult(
      GenerationNow(StorageGeneration::FromUint64(generation_number)));
  return absl::OkStatus();
}

}  // namespace
}  // namespace tensorstore
   // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::RocksDB)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::RocksDBSpec>
    registration;
}  // namespace