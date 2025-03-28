workspace(
    name = "tensorstore",
)

# Load rules_foreign_cc
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

maybe(
    http_archive,
    name = "rules_foreign_cc",
    urls = ["https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.5.0.tar.gz"],
    strip_prefix = "rules_foreign_cc-0.5.0",
    sha256 = "ed8068eea78cdccb8a638b3b4b49dd333db01ec83879f3fcd460cc8b6aeaa0c6",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")
rules_foreign_cc_dependencies()

load("//:external.bzl", "tensorstore_dependencies")
tensorstore_dependencies()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

register_toolchains("@local_config_python//:py_toolchain")

# Register proto toolchains.
load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()

# Register apple_support toolchains, which are needed for cross-compilaton
# macOS. Unfortunately this (small) repo will have to be downloaded in all
# cases, even though it is only needed on macOS when cross-compiling.
load("@build_bazel_apple_support//crosstool:setup.bzl", "apple_cc_configure")

apple_cc_configure()

# Define LLVM toolchain used for extracting C++ API documentation information
load("@toolchains_llvm//toolchain:rules.bzl", "llvm_toolchain")

llvm_toolchain(
    name = "llvm_toolchain",
    # https://github.com/bazel-contrib/toolchains_llvm/blob/master/toolchain/internal/llvm_distributions.bzl
    llvm_versions = {
        # Note: Older versions are built against older glibc, which is needed
        # for compatibility with manylinux containers.
        "": "15.0.6",
        "darwin-aarch64": "15.0.7",
        "darwin-x86_64": "15.0.7",
    },
    extra_target_compatible_with = {
        "": ["@//docs:docs_toolchain_value"],
    },
)

load("@llvm_toolchain//:toolchains.bzl", "llvm_register_toolchains")

llvm_register_toolchains()
