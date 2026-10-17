// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
#include "ort_runtime.h"

#include <dlfcn.h>

#include <cstdlib>
#include <mutex>
#include <vector>

namespace komikun::ocr::ort {
namespace {

const OrtApi* g_api = nullptr;
void* g_handle = nullptr;
std::string g_version;
std::string g_error;
std::once_flag g_once;

/// A symbol that lives in *this* shared library and is never inlined or dropped, so `dladdr` can
/// answer "where am I loaded from?".
extern "C" __attribute__((noinline, used)) void KomikunOcrOwnObjectAnchor() {}

/// Directory part of `path`, or "" when there is no separator.
std::string DirName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return {};
  return path.substr(0, slash);
}

/// The directory containing this shared library, resolved through `dladdr`. Lets the shim find a
/// `libonnxruntime.so` that sits next to it without knowing the app's `filesDir`.
std::string OwnDirectory() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&KomikunOcrOwnObjectAnchor), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return DirName(info.dli_fname);
}

std::vector<std::string> Candidates() {
  std::vector<std::string> out;
  // 1. explicit override (host harness / debugging).
  if (const char* env = std::getenv("KOMIKUN_ORT_LIB"); env != nullptr && *env != '\0') {
    out.emplace_back(env);
  }
  // 2. next to this library (the downloaded `filesDir/paddle_ocr/lib/<abi>/` layout).
  const std::string own_dir = OwnDirectory();
  if (!own_dir.empty()) out.push_back(own_dir + "/libonnxruntime.so");
  // 3. the bare soname (the APK `jniLibs` layout: the app linker namespace already contains
  //    `lib/<abi>/`).
  out.emplace_back("libonnxruntime.so");
  return out;
}

void LoadOnce() {
  for (const std::string& candidate : Candidates()) {
    // RTLD_LOCAL: nothing outside this shim consumes ORT symbols. RTLD_NOW so a mismatched
    // runtime fails here rather than mid-inference.
    void* handle = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr) {
      const char* err = dlerror();
      g_error = candidate + ": " + (err != nullptr ? err : "dlopen failed");
      continue;
    }
    auto get_api_base = reinterpret_cast<const OrtApiBase* (*)()>(dlsym(handle, "OrtGetApiBase"));
    if (get_api_base == nullptr) {
      g_error = candidate + ": OrtGetApiBase not found";
      dlclose(handle);
      continue;
    }
    const OrtApiBase* api_base = get_api_base();
    const OrtApi* api = api_base != nullptr ? api_base->GetApi(ORT_API_VERSION) : nullptr;
    if (api == nullptr) {
      g_error = candidate + ": no OrtApi for API version " + std::to_string(ORT_API_VERSION);
      dlclose(handle);
      continue;
    }
    g_handle = handle;
    g_api = api;
    const char* version =
        api_base->GetVersionString != nullptr ? api_base->GetVersionString() : nullptr;
    g_version = version != nullptr ? version : "";
    g_error.clear();
    return;
  }
  if (g_error.empty()) g_error = "libonnxruntime.so not found";
}

}  // namespace

const OrtApi* Api() {
  std::call_once(g_once, LoadOnce);
  return g_api;
}

const std::string& LoadError() { return g_error; }

std::string RuntimeVersion() {
  std::call_once(g_once, LoadOnce);
  return g_version;
}

}  // namespace komikun::ocr::ort
