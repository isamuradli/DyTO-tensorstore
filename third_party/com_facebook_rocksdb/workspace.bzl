load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

def repo():
    maybe(
        third_party_http_archive,
        name = "com_facebook_rocksdb",
        strip_prefix = "rocksdb-8.0.0",
        urls = [
            "https://github.com/facebook/rocksdb/archive/refs/tags/v8.0.0.tar.gz", 
        ],
        sha256 = "05ff6b0e89bffdf78b5a9d6fca46cb06bde6189f5787b9eeaef0511b782c1033",
        build_file = Label("//third_party:com_facebook_rocksdb/rocksdb.BUILD.bazel"),
        system_build_file = Label("//third_party:com_facebook_rocksdb/system.BUILD.bazel"),
        
    )
