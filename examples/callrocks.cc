#include <rocksdb/db.h>
#include <rocksdb/options.h>

#include <iostream>
using rocksdb::Status;

int main() {
  rocksdb::DB* db;
  rocksdb::Options options;
  options.create_if_missing = true;

  std::string db_path = "testdb";
  rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db);

  std::string value;
  Status s =
      db->Get(rocksdb::ReadOptions(), "output.zarr/1.1_generation", &value);
  if (s.ok()) {
    std::cout << "Value: " << value << std::endl;
  } else {
    std::cerr << "Error reading: " << s.ToString() << std::endl;
  }
  return 1;
}

// #include <iostream>
// #include <string>
// #include <rocksdb/db.h>
// #include <rocksdb/options.h>

// int main() {
//     rocksdb::DB* db;
//     rocksdb::Options options;
//     options.create_if_missing = true;

//     std::string db_path = "testdb"; // Replace with your RocksDB path
//     rocksdb::Status s = rocksdb::DB::Open(options, db_path, &db);
//     if (!s.ok()) {
//         std::cerr << "Failed to open database: " << s.ToString() <<
//         std::endl; return 1;
//     }

//     rocksdb::Iterator* it = db->NewIterator(rocksdb::ReadOptions());

//     // 1. Iterate through all keys:
//     std::cout << "Iterating through all keys:" << std::endl;
//     for (it->SeekToFirst(); it->Valid(); it->Next()) {
//         std::string key = it->key().ToString();
//         // You can also access the value: std::string value =
//         it->value().ToString(); std::cout << key << std::endl; // Or do
//         something else with the key
//     }

//     // 2. Iterate from a specific key (prefix scan):
//     std::string prefix = "my_prefix_"; // Example prefix
//     std::cout << "\nIterating from prefix '" << prefix << "':" << std::endl;
//     for (it->Seek(prefix); it->Valid() && it->key().starts_with(prefix);
//     it->Next()) {
//         std::string key = it->key().ToString();
//         std::cout << key << std::endl;
//     }

//     // 3. Reverse iteration (from last to first):
//     std::cout << "\nReverse iteration:" << std::endl;
//     for (it->SeekToLast(); it->Valid(); it->Prev()) {
//         std::string key = it->key().ToString();
//         std::cout << key << std::endl;
//     }

//     delete it; // Important: Delete the iterator when you're done with it.
//     delete db;
//     return 0;
// }