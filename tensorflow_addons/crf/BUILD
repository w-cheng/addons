# Description:
#   Contains classes to construct a CRF layer
#   APIs here are meant to evolve over time.

licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

py_library(
    name = "crf",
    srcs = [
        "__init__.py",
        "crf.py"
    ],
    srcs_version = "PY2AND3"
)

py_test(
    name = "crf_test",
    size = "medium",
    srcs = ["crf_test.py"],
    deps = [
        ":crf",
    ],
)