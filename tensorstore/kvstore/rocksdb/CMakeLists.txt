cmake_minimum_required(VERSION 3.10)
project(RocksDBExample)

set(CMAKE_CXX_STANDARD 17)

# Find RocksDB
find_package(RocksDB REQUIRED)

add_executable(rocksdb_testingCode sampleToUseRocksDBAPI.cc)
target_link_libraries(rocksdb_testingCode RocksDB::rocksdb)
