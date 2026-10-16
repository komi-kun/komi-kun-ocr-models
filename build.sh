#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
#
# Builds komi-kun's PaddleOCR native artifact:
#   * libpaddle_ocr.so for every supported Android ABI (our JNI over Paddle-Lite)
#   * an optional offline host harness used to validate the pipeline without a device
#   * the converted Paddle-Lite models + recognition dictionary
#   * manifest.json (per-file SHA-256 / size / ABI / SPDX)
#
# Everything is pinned to upstream releases in deps.lock.json. Nothing here is vendored into the
# app repo; the app only consumes the produced manifest.
#
# Usage:
#   ./build.sh                 # Android libs for every ABI + models + manifest
#   ./build.sh --host          # also build the host conformance harness and run it on test pages
#   ./build.sh --abi arm64-v8a # restrict to one ABI
#   ./build.sh --clean
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DL="$ROOT/dl"
DEPS="$ROOT/deps"
WORK="$ROOT/build"
OUT="$ROOT/out"

NDK_VERSION_DEFAULT="29.0.14206865"
ANDROID_API=26

LITE_TAG="v2.14-rc"
LITE_BASE="https://github.com/PaddlePaddle/Paddle-Lite/releases/download/$LITE_TAG"
OCV_TAG="v36"
OCV_BASE="https://github.com/nihui/opencv-mobile/releases/download/$OCV_TAG"
OCV_VERSION="4.13.0"
# PP-OCRv5 mobile models (det + unified rec covering Japanese), as officially pre-converted to the
# Paddle-Lite naive-buffer format by the PaddlePaddle team for `Paddle-Lite-Demo`'s
# `feature/paddle-x` branch (the "PaddleX-Lite-Deploy" of PaddleOCR's on-device guide). PaddleOCR's
# own PP-OCRv5 inference tars ship PaddlePaddle 3.x PIR (`inference.json`), which `opt` 2.14-rc
# cannot parse, so we consume the official `.nb` bytes directly and pin their SHA-256 in
# deps.lock.json. The `.nb` header records `v2.14-rc`, i.e. the exact opt/runtime we link.
V5_NB_BASE="https://paddlelite-demo.bj.bcebos.com/paddle-x/ocr/models"
V5_LABELS_URL="https://paddlelite-demo.bj.bcebos.com/demo/ocr/labels/labels.tar.gz"

# ABIs built from a Paddle-Lite prebuilt release. `x86_64` has no upstream prebuilt (Paddle-Lite
# ships armv7/armv8 for Android only) and would additionally need x86-targeted models -- see
# README.md.
ALL_ABIS=(arm64-v8a armeabi-v7a)

BUILD_HOST=0
CLEAN=0
MANIFEST_ONLY=0
ABIS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) BUILD_HOST=1 ;;
    --clean) CLEAN=1 ;;
    --manifest-only) MANIFEST_ONLY=1 ;;
    --abi) shift; ABIS+=("$1") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
[[ ${#ABIS[@]} -eq 0 ]] && ABIS=("${ALL_ABIS[@]}")

if [[ $CLEAN -eq 1 ]]; then rm -rf "$WORK" "$OUT"; fi
mkdir -p "$DL" "$DEPS" "$WORK" "$OUT/lib" "$OUT/models"

# ---------------------------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------------------------
find_ndk() {
  if [[ -n "${ANDROID_NDK_HOME:-}" && -d "$ANDROID_NDK_HOME" ]]; then echo "$ANDROID_NDK_HOME"; return; fi
  for base in "${ANDROID_SDK_ROOT:-}" "${ANDROID_HOME:-}" "$HOME/Android/Sdk" "$HOME/Library/Android/sdk"; do
    [[ -z "$base" ]] && continue
    if [[ -d "$base/ndk/$NDK_VERSION_DEFAULT" ]]; then echo "$base/ndk/$NDK_VERSION_DEFAULT"; return; fi
    if [[ -d "$base/ndk" ]]; then echo "$(find "$base/ndk" -maxdepth 1 -mindepth 1 -type d | sort | tail -1)"; return; fi
  done
  echo ""
}

# ---------------------------------------------------------------------------------------------
# downloads (pinned)
# ---------------------------------------------------------------------------------------------
fetch() { # url outfile
  local url="$1" out="$2"
  if [[ -s "$out" ]]; then echo "[skip] $(basename "$out")"; return; fi
  echo "[get ] $url"
  curl -fSL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

download_deps() {
  for abi in "${ABIS[@]}"; do
    case "$abi" in
      arm64-v8a) fetch "$LITE_BASE/inference_lite_lib.android.armv8.clang.c++_shared.with_extra.tar.gz" "$DL/lite.android.armv8.tar.gz" ;;
      armeabi-v7a) fetch "$LITE_BASE/inference_lite_lib.android.armv7.clang.c++_shared.with_extra.tar.gz" "$DL/lite.android.armv7.tar.gz" ;;
      *) echo "no Paddle-Lite prebuilt for $abi" >&2; exit 1 ;;
    esac
  done
  fetch "$OCV_BASE/opencv-mobile-$OCV_VERSION-android.zip" "$DL/opencv-mobile-android.zip"
  fetch "https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.hpp" "$DL/clipper.hpp"
  fetch "https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.cpp" "$DL/clipper.cpp"
  fetch "$V5_NB_BASE/PP-OCRv5_mobile_det.tar.gz" "$DL/ppocrv5_mobile_det.tar.gz"
  fetch "$V5_NB_BASE/PP-OCRv5_mobile_rec.tar.gz" "$DL/ppocrv5_mobile_rec.tar.gz"
  fetch "$V5_LABELS_URL" "$DL/labels.tar.gz"

  if [[ $BUILD_HOST -eq 1 ]]; then
    fetch "$LITE_BASE/inference_lite_lib.x86.linux.tar.gz" "$DL/lite.x86.linux.tar.gz"
    fetch "$OCV_BASE/opencv-mobile-$OCV_VERSION-ubuntu-2404.zip" "$DL/opencv-mobile-ubuntu.zip"
  fi

  extract() { # tar/zip src dest
    local src="$1" dest="$2"
    if [[ -d "$dest" && -n "$(ls -A "$dest" 2>/dev/null)" ]]; then return; fi
    mkdir -p "$dest"
    case "$src" in
      *.tar.gz) tar -xf "$src" -C "$dest" ;;
      *.zip) unzip -q -o "$src" -d "$dest" ;;
    esac
  }
  for abi in "${ABIS[@]}"; do
    case "$abi" in
      arm64-v8a) extract "$DL/lite.android.armv8.tar.gz" "$DEPS/lite.android.armv8" ;;
      armeabi-v7a) extract "$DL/lite.android.armv7.tar.gz" "$DEPS/lite.android.armv7" ;;
    esac
  done
  extract "$DL/opencv-mobile-android.zip" "$DEPS/ocv-android"
  mkdir -p "$ROOT/third_party/clipper"
  cp -f "$DL/clipper.hpp" "$DL/clipper.cpp" "$ROOT/third_party/clipper/"
  if [[ $BUILD_HOST -eq 1 ]]; then
    extract "$DL/lite.x86.linux.tar.gz" "$DEPS/lite.x86.linux"
    extract "$DL/opencv-mobile-ubuntu.zip" "$DEPS/ocv-ubuntu"
  fi
}

# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
stage_models() {
  local tmp="$WORK/models-native"
  mkdir -p "$tmp"
  [[ -f "$tmp/PP-OCRv5_mobile_det.nb" ]] || tar -xf "$DL/ppocrv5_mobile_det.tar.gz" -C "$tmp"
  [[ -f "$tmp/PP-OCRv5_mobile_rec.nb" ]] || tar -xf "$DL/ppocrv5_mobile_rec.tar.gz" -C "$tmp"
  [[ -f "$tmp/ppocr_keys_ocrv5.txt" ]] || tar -xf "$DL/labels.tar.gz" -C "$tmp" ppocr_keys_ocrv5.txt

  # Published under komi-kun's own flat names; byte-identical to the official PaddlePaddle `.nb`
  # (sha256 pinned in deps.lock.json and re-verified against the release manifest at runtime).
  cp -f "$tmp/PP-OCRv5_mobile_det.nb" "$OUT/models/ppocrv5_det.nb"
  cp -f "$tmp/PP-OCRv5_mobile_rec.nb" "$OUT/models/ppocrv5_rec.nb"
  cp -f "$tmp/ppocr_keys_ocrv5.txt" "$OUT/models/ppocrv5_dict.txt"
}

# ---------------------------------------------------------------------------------------------
# android libraries
# ---------------------------------------------------------------------------------------------
lite_dir_for() {
  case "$1" in
    arm64-v8a) echo "$DEPS/lite.android.armv8/inference_lite_lib.android.armv8.clang.c++_shared.with_extra/cxx" ;;
    armeabi-v7a) echo "$DEPS/lite.android.armv7/inference_lite_lib.android.armv7.clang.c++_shared.with_extra/cxx" ;;
  esac
}

build_android() {
  local ndk="$1"
  local tc="$ndk/build/cmake/android.toolchain.cmake"
  local ocv="$DEPS/ocv-android/opencv-mobile-$OCV_VERSION-android/sdk/native"
  for abi in "${ABIS[@]}"; do
    local lite; lite="$(lite_dir_for "$abi")"
    local extra=""
    [[ -f "$ocv/3rdparty/libs/$abi/libkleidicv.a" ]] && extra="$ocv/3rdparty/libs/$abi/libkleidicv.a;$ocv/3rdparty/libs/$abi/libkleidicv_hal.a;$ocv/3rdparty/libs/$abi/libkleidicv_thread.a"
    echo "[build] libpaddle_ocr.so ($abi)"
    cmake -S "$ROOT" -B "$WORK/android-$abi" \
      -DCMAKE_TOOLCHAIN_FILE="$tc" \
      -DANDROID_ABI="$abi" \
      -DANDROID_PLATFORM="android-$ANDROID_API" \
      -DANDROID_STL=c++_shared \
      -DCMAKE_BUILD_TYPE=Release \
      -DKOMIKUN_LITE_INCLUDE="$lite/include" \
      -DKOMIKUN_LITE_LIB="$lite/lib/libpaddle_api_light_bundled.a" \
      -DKOMIKUN_OPENCV_INCLUDE="$ocv/jni/include" \
      -DKOMIKUN_OPENCV_LIB_DIR="$ocv/staticlibs/$abi" \
      -DKOMIKUN_OPENCV_EXTRA_LIBS="$extra" \
      >/dev/null
    cmake --build "$WORK/android-$abi" -j"$(nproc)" >/dev/null
    mkdir -p "$OUT/lib/$abi"
    cp -f "$WORK/android-$abi/libpaddle_ocr.so" "$OUT/lib/$abi/libpaddle_ocr.so"
    "$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$OUT/lib/$abi/libpaddle_ocr.so" 2>/dev/null || true
  done
}

# ---------------------------------------------------------------------------------------------
# host harness (validation only; never shipped)
# ---------------------------------------------------------------------------------------------
build_host() {
  local lite="$DEPS/lite.x86.linux/inference_lite_lib/cxx"
  local mklml="$DEPS/lite.x86.linux/inference_lite_lib/third_party/mklml/lib"
  local ocv="$DEPS/ocv-ubuntu/opencv-mobile-$OCV_VERSION-ubuntu-2404"
  echo "[build] paddle_ocr_host (validation harness)"
  cmake -S "$ROOT" -B "$WORK/host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DKOMIKUN_LITE_INCLUDE="$lite/include" \
    -DKOMIKUN_LITE_LIB="$lite/lib/libpaddle_api_light_bundled.a" \
    -DKOMIKUN_OPENCV_INCLUDE="$ocv/include/opencv4" \
    -DKOMIKUN_OPENCV_LIB_DIR="$ocv/lib" \
    -DKOMIKUN_HOST_OPENMP_LIB_DIR="$mklml" \
    >/dev/null
  cmake --build "$WORK/host" -j"$(nproc)" >/dev/null
  cp -f "$WORK/host/paddle_ocr_host" "$OUT/paddle_ocr_host"
  echo "[host ] run: LD_LIBRARY_PATH=$mklml $OUT/paddle_ocr_host <det.nb> <rec.nb> <dict> <page.ppm>"
}

# ---------------------------------------------------------------------------------------------
# NDK runtime libraries (libomp.so + libc++_shared.so)
# ---------------------------------------------------------------------------------------------
stage_runtime_libs() {
  local ndk="$1"
  local tcdir="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
  local clang_ver; clang_ver="$(ls "$tcdir/lib/clang" | sort -V | tail -1)"
  for abi in "${ABIS[@]}"; do
    local triple clang_arch
    case "$abi" in
      arm64-v8a) triple="aarch64-linux-android"; clang_arch="aarch64" ;;
      armeabi-v7a) triple="arm-linux-androideabi"; clang_arch="arm" ;;
    esac
    mkdir -p "$OUT/lib/$abi"
    cp -f "$tcdir/sysroot/usr/lib/$triple/libc++_shared.so" "$OUT/lib/$abi/libc++_shared.so"
    cp -f "$tcdir/lib/clang/$clang_ver/lib/linux/$clang_arch/libomp.so" "$OUT/lib/$abi/libomp.so"
  done
}

# ---------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------
sha256() { sha256sum "$1" | cut -d' ' -f1; }
size_of() { stat -c '%s' "$1"; }

# SPDX ids: our shim and the PaddleOCR/Paddle-Lite inputs are Apache-2.0; the NDK runtime
# libraries are Apache-2.0 WITH LLVM-exception (libc++, the LLVM OpenMP runtime).
spdx_for() { case "$1" in libc++_shared.so|libomp.so) echo "Apache-2.0 WITH LLVM-exception" ;; *) echo "Apache-2.0" ;; esac; }

MODELS=(ppocrv5_det.nb ppocrv5_rec.nb ppocrv5_dict.txt)

json_entries() { # [$1=1 to omit libc++_shared.so]
  local omit_libcxx="${1:-0}"
  local first=1 abi
  for abi in "${ABIS[@]}"; do
    for so in libpaddle_ocr.so libomp.so libc++_shared.so; do
      [[ "$so" == "libc++_shared.so" && "$omit_libcxx" == "1" ]] && continue
      local f="$OUT/lib/$abi/$so"
      [[ -f "$f" ]] || continue
      [[ $first -eq 0 ]] && echo ","
      first=0
      # Release assets are a flat namespace (GitHub rewrites '/' to '.'), so `path` is the asset
      # name; the build-time fetch task builds the URL from it.
      printf '    { "abi": "%s", "name": "%s", "path": "%s_%s", "sha256": "%s", "size": %s, "spdx": "%s" }' \
        "$abi" "$so" "$abi" "$so" "$(sha256 "$f")" "$(size_of "$f")" "$(spdx_for "$so")"
    done
  done
  for m in "${MODELS[@]}"; do
    local f="$OUT/models/$m"
    [[ -f "$f" ]] || continue
    [[ $first -eq 0 ]] && echo ","
    first=0
    printf '    { "name": "%s", "path": "%s", "sha256": "%s", "size": %s, "spdx": "Apache-2.0" }' \
      "$m" "$m" "$(sha256 "$f")" "$(size_of "$f")"
  done
}

BASE_URL="https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2"

# The release manifest is what `ModelDownloader` fetches at runtime; the app manifest is the copy
# that lives at `core/ocr/paddle-manifest.json` and drives the build-time fetch task.
write_manifests() {
  {
    echo "{"
    echo "  \"release\": \"ocr-models-v2\","
    echo "  \"baseUrl\": \"$BASE_URL\","
    echo "  \"files\": ["
    json_entries
    echo
    echo "  ]"
    echo "}"
  } > "$OUT/manifest.json"

  {
    echo "{"
    echo "  \"_comment\": ["
    echo "    \"Consumed by :core:ocr:fetchPaddleNativeLibs. sha256 must match the file published at\","
    echo "    \"<baseUrl>/<path>. Regenerate with build.sh --manifest-only if the release is republished.\","
    echo "    \"libc++_shared.so is deliberately NOT listed: AGP already packages the NDK copy for any\","
    echo "    \"module built with -DANDROID_STL=c++_shared (:core:dictionary), and a second copy in\","
    echo "    \"jniLibs fails the app merge with a duplicate-path error. Re-add the release's\","
    echo "    \"libc++_shared.so entries only if no other module provides it.\""
    echo "  ],"
    echo "  \"release\": \"ocr-models-v2\","
    echo "  \"baseUrl\": \"$BASE_URL\","
    echo "  \"abi\": [$(printf '"%s", ' "${ABIS[@]}" | sed 's/, $//')],"
    echo "  \"files\": ["
    json_entries 1
    echo
    echo "  ],"
    echo "  \"models\": {"
    echo "    \"det\": [\"ppocrv5_det.nb\"],"
    echo "    \"rec\": [\"ppocrv5_rec.nb\"],"
    echo "    \"dict\": [\"ppocrv5_dict.txt\"]"
    echo "  }"
    echo "}"
  } > "$OUT/app-paddle-manifest.json"
  echo "[done] $OUT/manifest.json + $OUT/app-paddle-manifest.json"
}

main() {
  if [[ $MANIFEST_ONLY -eq 1 ]]; then
    write_manifests
    return 0
  fi
  local ndk=""
  if [[ ${#ABIS[@]} -gt 0 ]]; then ndk="$(find_ndk)"; fi
  download_deps
  stage_models
  if [[ -n "$ndk" ]]; then
    build_android "$ndk"
    stage_runtime_libs "$ndk"
  else
    echo "[warn] no Android NDK found; skipping .so build" >&2
  fi
  [[ $BUILD_HOST -eq 1 ]] && build_host
  write_manifests
  echo
  echo "Artifacts in $OUT:"
  find "$OUT" -type f | sort
  if [[ $BUILD_HOST -eq 1 ]]; then
    echo
    echo "[info] host harness built at $OUT/paddle_ocr_host (toolchain check only)."
    echo "[info] PP-OCRv5 ships only as arm-targeted .nb, so the offline conformance run"
    echo "[info] (which needs x86-targeted models) is skipped for this release; see README."
  fi
}

main
