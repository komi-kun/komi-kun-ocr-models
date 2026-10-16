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
PADDLEOCR_MODEL_BASE="https://paddleocr.bj.bcebos.com/PP-OCRv4"
PADDLEOCR_DICT_BASE="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v2.7.0/ppocr/utils/dict"

# ABIs built from a Paddle-Lite prebuilt release. `x86_64` has no upstream prebuilt (Paddle-Lite
# ships armv7/armv8 for Android only) and would additionally need x86-targeted models -- see
# README.md.
ALL_ABIS=(arm64-v8a armeabi-v7a)

BUILD_HOST=0
CLEAN=0
ABIS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) BUILD_HOST=1 ;;
    --clean) CLEAN=1 ;;
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
  fetch "$LITE_BASE/opt_linux_x86" "$DL/opt_linux_x86"
  fetch "$OCV_BASE/opencv-mobile-$OCV_VERSION-android.zip" "$DL/opencv-mobile-android.zip"
  fetch "https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.hpp" "$DL/clipper.hpp"
  fetch "https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.cpp" "$DL/clipper.cpp"
  fetch "$PADDLEOCR_MODEL_BASE/chinese/ch_PP-OCRv4_det_infer.tar" "$DL/ch_PP-OCRv4_det_infer.tar"
  fetch "$PADDLEOCR_MODEL_BASE/multilingual/japan_PP-OCRv4_rec_infer.tar" "$DL/japan_PP-OCRv4_rec_infer.tar"
  fetch "$PADDLEOCR_DICT_BASE/japan_dict.txt" "$DL/japan_dict.txt"

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
convert_models() {
  local opt="$DL/opt_linux_x86"
  chmod +x "$opt"
  local tmp="$WORK/modelconv"
  mkdir -p "$tmp"
  [[ -d "$tmp/ch_PP-OCRv4_det_infer" ]] || tar -xf "$DL/ch_PP-OCRv4_det_infer.tar" -C "$tmp"
  [[ -d "$tmp/japan_PP-OCRv4_rec_infer" ]] || tar -xf "$DL/japan_PP-OCRv4_rec_infer.tar" -C "$tmp"

  # PaddleOCR's models are Apache-2.0; `opt` converts Paddle inference graphs to the Paddle-Lite
  # naive-buffer format the mobile runtime loads. `valid_targets=arm` is PaddleOCR's own choice
  # for the official `.nb` models; the resulting graph is ARM-specific (see README).
  "$opt" --model_dir="$tmp/ch_PP-OCRv4_det_infer" --optimize_out="$OUT/models/ppocrv4_det" \
         --optimize_out_type=naive_buffer --valid_targets=arm >/dev/null
  "$opt" --model_dir="$tmp/japan_PP-OCRv4_rec_infer" --optimize_out="$OUT/models/ppocrv4_japan_rec" \
         --optimize_out_type=naive_buffer --valid_targets=arm >/dev/null
  cp -f "$DL/japan_dict.txt" "$OUT/models/ppocrv4_japan_dict.txt"

  # Validation aid only: x86_64-targeted conversions of the same models, so the host harness can
  # exercise the identical engine sources off-device. NOT published (see README).
  if [[ $BUILD_HOST -eq 1 ]]; then
    mkdir -p "$OUT/models-x86"
    "$opt" --model_dir="$tmp/ch_PP-OCRv4_det_infer" --optimize_out="$OUT/models-x86/ppocrv4_det" \
           --optimize_out_type=naive_buffer --valid_targets=x86 >/dev/null
    "$opt" --model_dir="$tmp/japan_PP-OCRv4_rec_infer" --optimize_out="$OUT/models-x86/ppocrv4_japan_rec" \
           --optimize_out_type=naive_buffer --valid_targets=x86 >/dev/null
    cp -f "$DL/japan_dict.txt" "$OUT/models-x86/ppocrv4_japan_dict.txt"
  fi
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

MODELS=(ppocrv4_det.nb ppocrv4_japan_rec.nb ppocrv4_japan_dict.txt)

json_entries() { # produce JSON entries into stdout
  local first=1 path abi
  for abi in "${ABIS[@]}"; do
    for so in libpaddle_ocr.so libomp.so libc++_shared.so; do
      local f="$OUT/lib/$abi/$so"
      [[ -f "$f" ]] || continue
      [[ $first -eq 0 ]] && echo ","
      first=0
      printf '    { "abi": "%s", "name": "%s", "path": "lib/%s/%s", "sha256": "%s", "size": %s, "spdx": "%s" }' \
        "$abi" "$so" "$abi" "$so" "$(sha256 "$f")" "$(size_of "$f")" "$(spdx_for "$so")"
    done
  done
  for m in "${MODELS[@]}"; do
    local f="$OUT/models/$m"
    [[ -f "$f" ]] || continue
    [[ $first -eq 0 ]] && echo ","
    first=0
    printf '    { "name": "%s", "path": "models/%s", "sha256": "%s", "size": %s, "spdx": "Apache-2.0" }' \
      "$m" "$m" "$(sha256 "$f")" "$(size_of "$f")"
  done
}

BASE_URL="https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v1"

# The release manifest is what `ModelDownloader` fetches at runtime; the app manifest is the copy
# that lives at `core/ocr/paddle-manifest.json` and drives the build-time fetch task.
write_manifests() {
  {
    echo "{"
    echo "  \"release\": \"ocr-models-v1\","
    echo "  \"baseUrl\": \"$BASE_URL\","
    echo "  \"abi\": [$(printf '"%s", ' "${ABIS[@]}" | sed 's/, $//')],"
    echo "  \"files\": ["
    json_entries
    echo
    echo "  ]"
    echo "}"
  } > "$OUT/manifest.json"

  {
    echo "{"
    echo "  \"_comment\": ["
    echo "    \"Consumed by :core:ocr:fetchPaddleNativeLibs (contract §1.7). sha256 must match the files\"."
    echo "    \"published in $BASE_URL. See the integration notes in the artifact repo README.\""
    echo "  ],"
    echo "  \"release\": \"ocr-models-v1\","
    echo "  \"baseUrl\": \"$BASE_URL\","
    echo "  \"abi\": [$(printf '"%s", ' "${ABIS[@]}" | sed 's/, $//')],"
    echo "  \"files\": ["
    json_entries
    echo
    echo "  ],"
    echo "  \"models\": {"
    echo "    \"det\": [\"ppocrv4_det.nb\"],"
    echo "    \"rec\": [\"ppocrv4_japan_rec.nb\"],"
    echo "    \"dict\": [\"ppocrv4_japan_dict.txt\"]"
    echo "  }"
    echo "}"
  } > "$OUT/app-paddle-manifest.json"
  echo "[done] $OUT/manifest.json + $OUT/app-paddle-manifest.json"
}

main() {
  local ndk=""
  if [[ ${#ABIS[@]} -gt 0 ]]; then ndk="$(find_ndk)"; fi
  download_deps
  convert_models
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
    if command -v python3 >/dev/null && python3 "$ROOT/tests/make_pages.py" "$ROOT/tests" >/dev/null 2>&1; then
      echo "Host harness smoke test (x86 validation models, synthetic page):"
      "$OUT/paddle_ocr_host" "$OUT/models-x86/ppocrv4_det.nb" "$OUT/models-x86/ppocrv4_japan_rec.nb" \
        "$OUT/models-x86/ppocrv4_japan_dict.txt" "$ROOT/tests/page_horizontal.ppm" 2>/dev/null | head -6 || true
    else
      echo "[warn] Pillow/CJK font unavailable; skipping the host smoke test" >&2
    fi
  fi
}

main
