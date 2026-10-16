// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// JNI shim exposing komi-kun's `mihon.core.ocr.paddle.PaddleOcrEngine` native surface:
//
//   private external fun nativeInit(detParam, detBin, recParam, recBin, dict, numThreads): Boolean
//   private external fun nativeProcess(bitmap, thresh, boxThresh, unclipRatio, filterFurigana,
//                                      numThreads): String
//   private external fun nativeRelease()
//
// The returned String is the `TIMING|`/`LINE|` text protocol parsed by
// `PaddleOcrEngine.parseOutput`. A single process-wide engine instance is kept (matching the
// Kotlin side's companion-wide locking), guarded by a mutex.
#include <jni.h>

#include <android/bitmap.h>
#include <android/log.h>

#include <opencv2/imgproc.hpp>

#include <mutex>
#include <string>
#include <sys/stat.h>

#include "ocr_engine.h"

#define LOG_TAG "paddle_ocr"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

komikun::ocr::Engine g_engine;
std::mutex g_mutex;

std::string JStringToStd(JNIEnv* env, jstring value) {
  if (value == nullptr) return {};
  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) return {};
  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

bool FileExists(const std::string& path) {
  struct stat st {};
  return !path.empty() && ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// Accepts either the combined naive-buffer model at the primary path, its `.nb` sibling, or the
// secondary path. Lets the artifact survive the historical `*.param`/`*.bin` naming.
std::string ResolveModel(const std::string& primary, const std::string& secondary) {
  if (FileExists(primary)) return primary;
  if (FileExists(primary + ".nb")) return primary + ".nb";
  if (FileExists(secondary)) return secondary;
  return primary;
}

jstring EmptyString(JNIEnv* env) { return env->NewStringUTF(""); }

}  // namespace

extern "C" {

JNIEXPORT jboolean JNICALL
Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeInit(  // NOLINT(readability-identifier-naming)
    JNIEnv* env, jobject /*thiz*/, jstring det_param, jstring det_bin, jstring rec_param,
    jstring rec_bin, jstring dict_path, jint num_threads) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_engine.ready()) return JNI_TRUE;

  const std::string det = ResolveModel(JStringToStd(env, det_param), JStringToStd(env, det_bin));
  const std::string rec = ResolveModel(JStringToStd(env, rec_param), JStringToStd(env, rec_bin));
  const std::string dict = JStringToStd(env, dict_path);

  LOGI("nativeInit det=%s rec=%s dict=%s threads=%d", det.c_str(), rec.c_str(), dict.c_str(),
       static_cast<int>(num_threads));

  const bool ok = g_engine.Init(det, rec, dict, static_cast<int>(num_threads));
  if (!ok) {
    LOGE("nativeInit failed (det=%s rec=%s dict=%s)", det.c_str(), rec.c_str(), dict.c_str());
    g_engine.Release();
  }
  return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeProcess(  // NOLINT(readability-identifier-naming)
    JNIEnv* env, jobject /*thiz*/, jobject bitmap, jfloat thresh, jfloat box_thresh,
    jfloat unclip_ratio, jboolean /*filter_furigana*/, jint /*num_threads*/) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_engine.ready()) {
    LOGE("nativeProcess called before a successful nativeInit");
    return EmptyString(env);
  }

  AndroidBitmapInfo info{};
  if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS) {
    LOGE("AndroidBitmap_getInfo failed");
    return EmptyString(env);
  }
  if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
    LOGE("unsupported bitmap format: %d", info.format);
    return EmptyString(env);
  }

  void* pixels = nullptr;
  if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
      pixels == nullptr) {
    LOGE("AndroidBitmap_lockPixels failed");
    return EmptyString(env);
  }

  // ARGB_8888 is laid out as R,G,B,A bytes; hand the engine BGR (cv::imread order), which is what
  // PaddleOCR's mobile post-processing reference expects.
  const cv::Mat rgba(static_cast<int>(info.height), static_cast<int>(info.width), CV_8UC4, pixels);
  cv::Mat bgr;
  cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
  AndroidBitmap_unlockPixels(env, bitmap);

  komikun::ocr::Options options;
  options.det_db_thresh = thresh;
  options.det_db_box_thresh = box_thresh;
  options.det_db_unclip_ratio = unclip_ratio;

  komikun::ocr::Timings timings;
  const auto lines = g_engine.Run(bgr, options, &timings);
  const std::string protocol =
      komikun::ocr::BuildProtocol(lines, timings, static_cast<int>(info.width),
                                  static_cast<int>(info.height));
  return env->NewStringUTF(protocol.c_str());
}

JNIEXPORT void JNICALL
Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeRelease(  // NOLINT(readability-identifier-naming)
    JNIEnv* /*env*/, jobject /*thiz*/) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_engine.Release();
  LOGI("nativeRelease");
}

}  // extern "C"
