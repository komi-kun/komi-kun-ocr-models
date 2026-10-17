// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// Dynamic loader for ONNX Runtime's ABI-stable C API.
//
// We do **not** link `libonnxruntime.so` at build time. ORT's C API is explicitly designed for
// dynamic loading: `onnxruntime_c_api.h` declares `OrtGetApiBase()` and the whole API surface is a
// versioned vtable (`OrtApiBase::GetApi(ORT_API_VERSION)`), so a shim compiled once against the
// header keeps working with any runtime that provides the same-or-newer API version. Loading at
// runtime also means the shim has no `DT_NEEDED` on ORT, so it never fails to `dlopen`/`System.load`
// just because the runtime is missing — `Engine::Init()` simply returns false and OCR degrades
// gracefully, exactly like the previous Paddle-Lite artifact.
//
// Search order (first hit wins):
//   1. an explicit override path (env `KOMIKUN_ORT_LIB`), used by the host harness;
//   2. `libonnxruntime.so` next to *this* shared library (covers the downloaded
//      `filesDir/paddle_ocr/lib/<abi>/` layout);
//   3. the bare soname `libonnxruntime.so` (covers the APK `jniLibs` layout, where the app's
//      linker namespace search path already contains `lib/<abi>/`).
#pragma once

#include <string>

#include "onnxruntime_c_api.h"

namespace komikun::ocr::ort {

/// Loads `libonnxruntime.so` once per process and returns its `OrtApi` vtable, or `nullptr` when
/// no candidate could be loaded. Subsequent calls return the cached result (never re-probes).
const OrtApi* Api();

/// Human-readable reason the last load attempt failed (empty when [Api] succeeded).
const std::string& LoadError();

/// The runtime version string reported by the loaded library, or "" when not loaded.
std::string RuntimeVersion();

}  // namespace komikun::ocr::ort
