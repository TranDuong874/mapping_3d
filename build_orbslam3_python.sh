#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="${SCRIPT_DIR}/dependency/ORB_SLAM3"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MAKE_JOBS="${MAKE_JOBS:-2}"
EXT_SUFFIX="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_config_var("EXT_SUFFIX"))
PY
)"
PYTHON_INCLUDE="$("$PYTHON_BIN" - <<'PY'
import sysconfig
print(sysconfig.get_paths()["include"])
PY
)"
OUTPUT_DIR="${SCRIPT_DIR}/pyorbslam3"
OUTPUT_FILE="${OUTPUT_DIR}/_orbslam3${EXT_SUFFIX}"

if [[ ! -f "${ORB_ROOT}/lib/libORB_SLAM3.so" ]]; then
  (cd "${ORB_ROOT}" && MAKE_JOBS="${MAKE_JOBS}" bash build.sh)
fi

mkdir -p "${OUTPUT_DIR}"

OPENCV_CFLAGS="$(pkg-config --cflags opencv4)"
OPENCV_LIBS="$(pkg-config --libs opencv4)"
EPOXY_LIBS="$(pkg-config --libs epoxy)"

g++ \
  -std=c++14 -O2 -Wall -Wextra -shared -fPIC \
  -DCOMPILEDWITHC14 -DHAVE_EIGEN -DHAVE_EPOXY -DPANGO_DEFAULT_WIN_URI=\"wayland\" -D_LINUX_ \
  -DEIGEN_DONT_VECTORIZE -DEIGEN_DONT_ALIGN_STATICALLY -DEIGEN_MAX_STATIC_ALIGN_BYTES=0 \
  -I"${PYTHON_INCLUDE}" \
  -I/usr/include/pybind11 \
  ${OPENCV_CFLAGS} \
  -I"${ORB_ROOT}" \
  -I"${ORB_ROOT}/include" \
  -I"${ORB_ROOT}/include/CameraModels" \
  -I"${ORB_ROOT}/Thirdparty/Sophus" \
  -I/usr/include/eigen3 \
  "${SCRIPT_DIR}/orbslam3_bindings.cpp" \
  -L"${ORB_ROOT}/lib" \
  -L"${ORB_ROOT}/Thirdparty/DBoW2/lib" \
  -L"${ORB_ROOT}/Thirdparty/g2o/lib" \
  -Wl,-rpath,"${ORB_ROOT}/lib:${ORB_ROOT}/Thirdparty/DBoW2/lib:${ORB_ROOT}/Thirdparty/g2o/lib:/usr/local/lib" \
  ${OPENCV_LIBS} \
  ${EPOXY_LIBS} \
  -lORB_SLAM3 -lDBoW2 -lg2o \
  -o "${OUTPUT_FILE}"

echo "Built ${OUTPUT_FILE}"
