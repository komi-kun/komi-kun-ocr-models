#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
#
# Builds komi-kun's PaddleOCR native artifact on ONNX Runtime:
#   * libpaddle_ocr.so for every supported Android ABI (our JNI shim over ONNX Runtime + OpenCV)
#   * the ONNX Runtime runtime library, as published by Microsoft (checksum-pinned, unmodified)
#   * the PP-OCRv6-small ONNX models + recognition dictionary
#   * an optional offline host harness used to validate the pipeline without a device
#   * manifest.json (per-file SHA-256 / size / ABI / SPDX) + app-paddle-manifest.json
#
# Everything is pinned to upstream releases in deps.lock.json. Nothing here is vendored into the
# app repo; the app only consumes the produced manifests.
#
# Usage:
#   ./build.sh                 # Android libs for every ABI + models + manifests
#   ./build.sh --host          # also build the host conformance harness
#   ./build.sh --abi arm64-v8a # restrict to one ABI
#   ./build.sh --manifest-only # only regenerate the manifests from an existing out/
#   ./build.sh --clean
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DL="$ROOT/dl"
DEPS="$ROOT/deps"
WORK="$ROOT/build"
OUT="$ROOT/out"

NDK_VERSION_DEFAULT="29.0.14206865"
ANDROID_API=26

ORT_VERSION="1.31.0"
ORT_AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/$ORT_VERSION/onnxruntime-android-$ORT_VERSION.aar"
ORT_LINUX_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"

OCV_TAG="v36"
OCV_VERSION="4.13.0"
OCV_BASE="https://github.com/nihui/opencv-mobile/releases/download/$OCV_TAG"

CLIPPER_HPP="https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.hpp"
CLIPPER_CPP="https://paddle-inference-dist.cdn.bcebos.com/PaddleLite/Clipper/clipper.cpp"

# PP-OCRv6 small: the smallest PP-OCRv6 recogniser that still covers Japanese kana (v6 *tiny*
# has no kana and must not be used). Pinned to a HuggingFace revision in deps.lock.json.
HF_BASE="https://huggingface.co"
V6_DET_REPO="PaddlePaddle/PP-OCRv6_small_det_onnx"
V6_REC_REPO="PaddlePaddle/PP-OCRv6_small_rec_onnx"

# ABIs are the app's single ABI set (contract §1.5). ONNX Runtime publishes all three, and the
# ONNX models are architecture-independent, so x86_64 is a first-class target here.
ALL_ABIS=(arm64-v8a armeabi-v7a x86_64)

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

# ---------------------------------------------------------------------------------------------
# pinned-input helpers (deps.lock.json is the single source of truth)
# ---------------------------------------------------------------------------------------------
json_get() { # <dot.path> -> value (null -> empty)
  python3 - "$ROOT/deps.lock.json" "$1" <<'PY'
import json, sys
node = json.load(open(sys.argv[1]))
for part in sys.argv[2].split("."):
    node = node.get(part) if isinstance(node, dict) else None
    if node is None:
        break
print("" if node is None else node)
PY
}

# deps.lock.json is parsed, so it must be valid JSON (no `//` comments - use `_comment`).
python3 -c "import json,sys;json.load(open(sys.argv[1]))" "$ROOT/deps.lock.json" \
  || { echo "[FAIL] deps.lock.json is not valid JSON" >&2; exit 1; }

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

sha256() { sha256sum "$1" | cut -d' ' -f1; }
size_of() { stat -c '%s' "$1"; }

fetch_pinned() { # url outfile expected_sha256
  local url="$1" out="$2" expected="$3"
  if [[ -z "$expected" ]]; then
    echo "[FAIL] no pinned sha256 for $(basename "$out") (deps.lock.json lookup failed)" >&2
    exit 1
  fi
  if [[ -s "$out" ]] && [[ "$(sha256 "$out")" == "$expected" ]]; then
    echo "[skip] $(basename "$out")"
    return
  fi
  echo "[get ] $url"
  curl -fSL --retry 3 -o "$out.part" "$url"
  local actual; actual="$(sha256 "$out.part")"
  if [[ "$actual" != "$expected" ]]; then
    rm -f "$out.part"
    echo "[FAIL] sha256 mismatch for $(basename "$out"): got $actual want $expected" >&2
    exit 1
  fi
  mv "$out.part" "$out"
}

# ---------------------------------------------------------------------------------------------
# downloads (pinned)
# ---------------------------------------------------------------------------------------------
download_deps() {
  fetch_pinned "$ORT_AAR_URL" "$DL/onnxruntime-android-$ORT_VERSION.aar" \
    "$(json_get onnxruntime.android_aar.sha256)"
  fetch_pinned "$ORT_LINUX_URL" "$DL/onnxruntime-linux-x64-$ORT_VERSION.tgz" \
    "$(json_get onnxruntime.linux_x86_64.sha256)"
  fetch_pinned "$OCV_BASE/opencv-mobile-$OCV_VERSION-android.zip" "$DL/opencv-mobile-android.zip" \
    "$(json_get opencv_mobile.android.sha256)"
  fetch_pinned "$CLIPPER_HPP" "$DL/clipper.hpp" "$(json_get clipper.sha256_hpp)"
  fetch_pinned "$CLIPPER_CPP" "$DL/clipper.cpp" "$(json_get clipper.sha256_cpp)"
  if [[ $BUILD_HOST -eq 1 ]]; then
    fetch_pinned "$OCV_BASE/opencv-mobile-$OCV_VERSION-ubuntu-2404.zip" "$DL/opencv-mobile-ubuntu.zip" \
      "$(json_get opencv_mobile.linux_x86_64.sha256)"
  fi
}

extract_deps() {
  mkdir -p "$DEPS"
  if [[ ! -d "$DEPS/ort-aar" ]]; then
    mkdir -p "$DEPS/ort-aar"
    unzip -q -o "$DL/onnxruntime-android-$ORT_VERSION.aar" -d "$DEPS/ort-aar"
  fi
  # The per-ABI runtime bytes are pinned individually (see deps.lock.json); the AAR itself is
  # verified at download time, this re-verifies what is actually staged/published.
  for abi in "${ABIS[@]}"; do
    local src="$DEPS/ort-aar/jni/$abi/libonnxruntime.so"
    local expected; expected="$(json_get "onnxruntime.android.$abi.sha256")"
    if [[ ! -f "$src" ]]; then
      echo "[FAIL] $src missing from the ORT AAR" >&2
      exit 1
    fi
    if [[ "$(sha256 "$src")" != "$expected" ]]; then
      echo "[FAIL] sha256 mismatch for $abi/libonnxruntime.so" >&2
      exit 1
    fi
    echo "[ok  ] $abi/libonnxruntime.so $expected"
  done
  if [[ $BUILD_HOST -eq 1 && ! -d "$DEPS/ort-linux" ]]; then
    mkdir -p "$DEPS/ort-linux"
    tar -xzf "$DL/onnxruntime-linux-x64-$ORT_VERSION.tgz" -C "$DEPS/ort-linux" --strip-components=1
  fi
  if [[ ! -d "$DEPS/ocv-android" ]]; then
    mkdir -p "$DEPS/ocv-android"
    unzip -q -o "$DL/opencv-mobile-android.zip" -d "$DEPS/ocv-android"
  fi
  if [[ $BUILD_HOST -eq 1 && ! -d "$DEPS/ocv-ubuntu" ]]; then
    mkdir -p "$DEPS/ocv-ubuntu"
    unzip -q -o "$DL/opencv-mobile-ubuntu.zip" -d "$DEPS/ocv-ubuntu"
  fi
  mkdir -p "$ROOT/third_party/clipper"
  cp -f "$DL/clipper.hpp" "$DL/clipper.cpp" "$ROOT/third_party/clipper/"
}

# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------
stage_models() {
  local det_rev rec_rev det_sha rec_sha
  det_rev="$(json_get models.det.revision)"
  rec_rev="$(json_get models.rec.revision)"
  det_sha="$(json_get models.det.sha256)"
  rec_sha="$(json_get models.rec.sha256)"
  fetch_pinned "$HF_BASE/$V6_DET_REPO/resolve/$det_rev/inference.onnx" \
    "$DL/ppocrv6_small_det.onnx" "$det_sha"
  fetch_pinned "$HF_BASE/$V6_REC_REPO/resolve/$rec_rev/inference.onnx" \
    "$DL/ppocrv6_small_rec.onnx" "$rec_sha"
  fetch_pinned "$HF_BASE/$V6_REC_REPO/resolve/$rec_rev/inference.yml" "$DL/v6_small_rec_inference.yml" \
    "$(json_get models.rec.yaml_sha256)"
  fetch_pinned "$HF_BASE/$V6_DET_REPO/resolve/$det_rev/inference.yml" "$DL/v6_small_det_inference.yml" \
    "$(json_get models.det.yaml_sha256)"

  cp -f "$DL/ppocrv6_small_det.onnx" "$OUT/models/ppocrv6_small_det.onnx"
  cp -f "$DL/ppocrv6_small_rec.onnx" "$OUT/models/ppocrv6_small_rec.onnx"
  python3 "$ROOT/tools/extract_dict.py" "$DL/v6_small_rec_inference.yml" \
    "$OUT/models/ppocrv6_small_dict.txt"

  local dict_sha; dict_sha="$(json_get models.dict.sha256)"
  local actual; actual="$(sha256 "$OUT/models/ppocrv6_small_dict.txt")"
  if [[ "$actual" != "$dict_sha" ]]; then
    echo "[FAIL] extracted dict sha256 mismatch: got $actual want $dict_sha" >&2
    exit 1
  fi
  echo "[dict] ppocrv6_small_dict.txt $actual"
}

# ---------------------------------------------------------------------------------------------
# android libraries
# ---------------------------------------------------------------------------------------------
build_android() {
  local ndk="$1"
  local tc="$ndk/build/cmake/android.toolchain.cmake"
  local ocv="$DEPS/ocv-android/opencv-mobile-$OCV_VERSION-android/sdk/native"
  for abi in "${ABIS[@]}"; do
    local extra=""
    [[ -f "$ocv/3rdparty/libs/$abi/libkleidicv.a" ]] && extra="$ocv/3rdparty/libs/$abi/libkleidicv.a;$ocv/3rdparty/libs/$abi/libkleidicv_hal.a;$ocv/3rdparty/libs/$abi/libkleidicv_thread.a"
    echo "[build] libpaddle_ocr.so ($abi)"
    cmake -S "$ROOT" -B "$WORK/android-$abi" \
      -DCMAKE_TOOLCHAIN_FILE="$tc" \
      -DANDROID_ABI="$abi" \
      -DANDROID_PLATFORM="android-$ANDROID_API" \
      -DANDROID_STL=c++_shared \
      -DCMAKE_BUILD_TYPE=Release \
      -DKOMIKUN_ORT_INCLUDE="$DEPS/ort-aar/headers" \
      -DKOMIKUN_OPENCV_INCLUDE="$ocv/jni/include" \
      -DKOMIKUN_OPENCV_LIB_DIR="$ocv/staticlibs/$abi" \
      -DKOMIKUN_OPENCV_EXTRA_LIBS="$extra" \
      >/dev/null
    cmake --build "$WORK/android-$abi" -j"$(nproc)" >/dev/null
    mkdir -p "$OUT/lib/$abi"
    cp -f "$WORK/android-$abi/libpaddle_ocr.so" "$OUT/lib/$abi/libpaddle_ocr.so"
    "$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded "$OUT/lib/$abi/libpaddle_ocr.so" 2>/dev/null || true
    # The ORT runtime is republished byte-for-byte, exactly as Microsoft built it.
    cp -f "$DEPS/ort-aar/jni/$abi/libonnxruntime.so" "$OUT/lib/$abi/libonnxruntime.so"
  done
}

# ---------------------------------------------------------------------------------------------
# host harness (validation only; never shipped)
# ---------------------------------------------------------------------------------------------
build_host() {
  local ort="$DEPS/ort-linux"
  local ocv="$DEPS/ocv-ubuntu/opencv-mobile-$OCV_VERSION-ubuntu-2404"
  echo "[build] paddle_ocr_host (validation harness)"
  cmake -S "$ROOT" -B "$WORK/host" \
    -DCMAKE_BUILD_TYPE=Release \
    -DKOMIKUN_ORT_INCLUDE="$ort/include" \
    -DKOMIKUN_ORT_LIB_DIR="$ort/lib" \
    -DKOMIKUN_OPENCV_INCLUDE="$ocv/include/opencv4" \
    -DKOMIKUN_OPENCV_LIB_DIR="$ocv/lib" \
    >/dev/null
  cmake --build "$WORK/host" -j"$(nproc)" >/dev/null
  cp -f "$WORK/host/paddle_ocr_host" "$OUT/paddle_ocr_host"
  echo "[host ] run: $OUT/paddle_ocr_host info $OUT/models/ppocrv6_small_det.onnx $OUT/models/ppocrv6_small_rec.onnx $OUT/models/ppocrv6_small_dict.txt"
}

# ---------------------------------------------------------------------------------------------
# NDK runtime libraries (libc++_shared.so + libomp.so — both are DT_NEEDED of our shim)
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
      x86_64) triple="x86_64-linux-android"; clang_arch="x86_64" ;;
    esac
    mkdir -p "$OUT/lib/$abi"
    cp -f "$tcdir/sysroot/usr/lib/$triple/libc++_shared.so" "$OUT/lib/$abi/libc++_shared.so"
    # opencv-mobile's `parallel_for_` is what needs OpenMP (ONNX Runtime has its own pool), but
    # `libpaddle_ocr.so` still carries a DT_NEEDED on it, so it must ship alongside the shim.
    cp -f "$tcdir/lib/clang/$clang_ver/lib/linux/$clang_arch/libomp.so" "$OUT/lib/$abi/libomp.so"
  done
}

# ---------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------
# SPDX ids: our shim is Apache-2.0; the ORT runtime is MIT; OpenCV/opencv-mobile and the PP-OCRv6
# models are Apache-2.0; the NDK libc++ is Apache-2.0 WITH LLVM-exception.
spdx_for() {
  case "$1" in
    libc++_shared.so|libomp.so) echo "Apache-2.0 WITH LLVM-exception" ;;
    libonnxruntime.so) echo "MIT" ;;
    *) echo "Apache-2.0" ;;
  esac
}

MODELS=(ppocrv6_small_det.onnx ppocrv6_small_rec.onnx ppocrv6_small_dict.txt)
NATIVE_LIBS=(libpaddle_ocr.so libonnxruntime.so libomp.so libc++_shared.so)
RELEASE="ocr-models-v3"
BASE_URL="https://github.com/komi-kun/komi-kun-ocr-models/releases/download/$RELEASE"

json_entries() { # [$1=1 to omit libc++_shared.so]
  local omit_libcxx="${1:-0}"
  local first=1 abi so m f
  for abi in "${ALL_ABIS[@]}"; do
    for so in "${NATIVE_LIBS[@]}"; do
      [[ "$so" == "libc++_shared.so" && "$omit_libcxx" == "1" ]] && continue
      f="$OUT/lib/$abi/$so"
      [[ -f "$f" ]] || continue
      [[ $first -eq 0 ]] && echo ","
      first=0
      printf '    { "abi": "%s", "name": "%s", "path": "%s_%s", "sha256": "%s", "size": %s, "spdx": "%s" }' \
        "$abi" "$so" "$abi" "$so" "$(sha256 "$f")" "$(size_of "$f")" "$(spdx_for "$so")"
    done
  done
  for m in "${MODELS[@]}"; do
    f="$OUT/models/$m"
    [[ -f "$f" ]] || continue
    [[ $first -eq 0 ]] && echo ","
    first=0
    printf '    { "name": "%s", "path": "%s", "sha256": "%s", "size": %s, "spdx": "%s" }' \
      "$m" "$m" "$(sha256 "$f")" "$(size_of "$f")" "$(spdx_for "$m")"
  done
}

# The release manifest is what `ModelDownloader` fetches at runtime; the app manifest is the copy
# that lives at `core/ocr/paddle-manifest.json` and drives the build-time fetch task.
write_manifests() {
  {
    echo "{"
    echo "  \"release\": \"$RELEASE\","
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
    echo "  \"release\": \"$RELEASE\","
    echo "  \"baseUrl\": \"$BASE_URL\","
    echo "  \"abi\": [$(printf '"%s", ' "${ALL_ABIS[@]}" | sed 's/, $//')],"
    echo "  \"files\": ["
    json_entries 1
    echo
    echo "  ],"
    echo "  \"models\": {"
    echo "    \"det\": [\"ppocrv6_small_det.onnx\"],"
    echo "    \"rec\": [\"ppocrv6_small_rec.onnx\"],"
    echo "    \"dict\": [\"ppocrv6_small_dict.txt\"]"
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
  extract_deps
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
}

main
