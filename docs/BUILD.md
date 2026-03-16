# Build Guide

## Prerequisites

The local build expects these system dependencies to already be installed:

- C++14 compiler
- CMake
- OpenCV 4
- Eigen3
- Pangolin
- epoxy
- pybind11 headers
- Python 3.10+
- `pkg-config`

## Build Command

Use low parallelism. ORB-SLAM3 and its third-party dependencies are memory-heavy to compile.

```bash
cd mapping_3d
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

What the build script does:

1. builds `dependency/ORB_SLAM3/Thirdparty/DBoW2`
2. builds `dependency/ORB_SLAM3/Thirdparty/g2o`
3. builds `dependency/ORB_SLAM3/Thirdparty/Sophus`
4. builds `dependency/ORB_SLAM3/lib/libORB_SLAM3.so`
5. builds `pyorbslam3/_orbslam3*.so`

## Smoke Test

```bash
cd mapping_3d
python3 test_orbslam3_import.py
```

Expected output:

```text
pyorbslam3 import ok
```

## Clean Rebuild

If you need to force a clean native rebuild:

```bash
rm -rf dependency/ORB_SLAM3/build
rm -rf dependency/ORB_SLAM3/lib
rm -rf dependency/ORB_SLAM3/Thirdparty/DBoW2/build
rm -rf dependency/ORB_SLAM3/Thirdparty/DBoW2/lib
rm -rf dependency/ORB_SLAM3/Thirdparty/g2o/build
rm -rf dependency/ORB_SLAM3/Thirdparty/g2o/lib
rm -rf dependency/ORB_SLAM3/Thirdparty/Sophus/build
rm -f pyorbslam3/_orbslam3*.so
MAKE_JOBS=2 bash ./build_orbslam3_python.sh
```

## Notes

- The vendored ORB-SLAM3 build is pinned to safer `-O2` flags and disabled Eigen static alignment/vectorization to avoid the runtime corruption seen with the default aggressive build.
- `dependency/ORB_SLAM3/Vocabulary/ORBvoc.txt` is unpacked from `ORBvoc.txt.tar.gz` during build if needed.
