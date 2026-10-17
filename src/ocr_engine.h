// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// PaddleOCR (ONNX Runtime) text detection + recognition engine for komi-kun.
//
// This is the portable, JNI-free core: it loads a DB text-detection ONNX graph and a CRNN
// text-recognition ONNX graph and runs the standard PaddleOCR mobile pipeline over an in-memory
// image. Inference goes through ONNX Runtime's ABI-stable C API, loaded dynamically (see
// `ort_runtime.h`); the graphs are ONNX exports of the official PaddlePaddle PP-OCR models, so they
// are architecture-independent and the same files run on every ABI.
//
// The detection post-processing (`BoxesFromBitmap`/`FilterTagDetRes`/`Unclip`) and the recognition
// crop/resize come from PaddleOCR's Apache-2.0 `deploy/lite` reference implementation, vendored
// under `third_party/paddleocr_lite/`. The engine itself (model loading, orchestration,
// preprocessing parity, protocol output) is ours.
//
// Preprocessing is deliberately byte-for-byte the same as the validated benchmark
// (`/tmp/opencode/ocr-bench/common.py`), against which PP-OCRv6-small was selected:
//   * detection: longest side <= 960, both sides rounded to a multiple of 32 (`round`, not truncate),
//     BGR, `/255` then `(x - mean) / std` with mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224,
//     0.225]`,
//   * recognition: fixed `rec_image_height = 48`, width `ceil(48 * w / h)` capped at 3200, BGR,
//     `(x / 255 - 0.5) / 0.5`,
//   * greedy CTC decode with dict index 0 = blank and a trailing space class.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

struct OrtApi;
struct OrtEnv;
struct OrtMemoryInfo;
struct OrtRunOptions;
struct OrtSession;

namespace komikun::ocr {

/// Runtime-tunable detector thresholds. Mirrors the values `PaddleOcrEngine` passes to JNI.
struct Options {
  float det_db_thresh = 0.3f;
  float det_db_box_thresh = 0.5f;
  float det_db_unclip_ratio = 1.6f;
  bool det_db_use_dilate = false;
  bool det_use_polygon_score = true;
  int max_side_len = 960;
  int rec_image_height = 48;
};

/// One recognised text line, in original-image pixel coordinates.
struct Line {
  std::string text;
  float score = 0.f;
  /// Quad corners in reading order: top-left, top-right, bottom-right, bottom-left.
  std::array<std::array<float, 2>, 4> quad{};
  bool vertical = false;
  /// Reading-order rank (manga order; informational for the Kotlin consumer).
  int order = 0;
};

/// Per-stage wall-clock timings (milliseconds) reported through the `TIMING|` protocol line.
struct Timings {
  double prep = 0.0;  // detector pre-processing (resize + normalise)
  double det = 0.0;   // detector inference
  double post = 0.0;  // detector post-processing (box extraction)
  double crop = 0.0;  // perspective crop of every detected box
  double rec = 0.0;   // recognition pre-processing + inference + CTC decode
  int raw_boxes = 0;  // number of boxes that survived detection filtering
};

class Engine {
 public:
  Engine() = default;
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /// Loads the detection + recognition ONNX models and the recognition dictionary.
  bool Init(const std::string& det_model,
            const std::string& rec_model,
            const std::string& dict_path,
            int num_threads);

  bool ready() const { return det_ != nullptr && rec_ != nullptr && !dict_.empty(); }

  void Release();

  /// Runs full-page OCR. @p bgr is a CV_8UC3 image in BGR order (as `cv::imread` produces).
  std::vector<Line> Run(const cv::Mat& bgr, const Options& options, Timings* timings);

#ifdef KOMIKUN_OCR_HOST_TEST
  /// Host-only parity hook (never compiled into the shipped `.so`): runs recognition on a single
  /// crop through the exact same preprocessing/decode path as [Run]. Used by the off-device
  /// conformance harness to compare against the recorded benchmark outputs crop-by-crop.
  bool RecognizeCropParity(const cv::Mat& bgr_crop, std::string* text, float* score);
#endif

 private:
  /// Creates the ORT env / memory info / both sessions. Returns false with `ort_error()` set.
  bool CreateSessions();

  /// Runs the detector on @p bgr; writes the probability map to @p pred_map and the resize ratios
  /// to @p ratio_hw (h then w). Records the pre-processing and inference times in @p prep_ms /
  /// @p det_ms. Returns false on failure.
  bool RunDetector(const cv::Mat& bgr, const Options& options, cv::Mat* pred_map,
                   std::vector<float>* ratio_hw, double* prep_ms, double* det_ms);

  /// Runs recognition on one already-oriented crop (landscape, as `GetRotateCropImage` returns).
  bool RecognizeCrop(const cv::Mat& crop, std::string* text, float* score);

  /// ORT objects. `env_`, `memory_`, `run_options_` are created lazily and kept for the process
  /// lifetime (re-creating them per `Init()` would only churn thread pools); the sessions are
  /// created by [Init] and dropped by [Release].
  const OrtApi* api_ = nullptr;
  OrtEnv* env_ = nullptr;
  OrtMemoryInfo* memory_ = nullptr;
  OrtRunOptions* run_options_ = nullptr;
  OrtSession* det_ = nullptr;
  OrtSession* rec_ = nullptr;
  std::string det_path_;
  std::string rec_path_;
  std::string det_input_;
  std::string rec_input_;
  std::string det_output_;
  std::string rec_output_;
  std::string ort_error_;
  std::vector<std::string> dict_;  // index 0 == CTC blank, last == space
  int threads_ = 4;
};

/// Serialises the engine result into the `TIMING|`/`LINE|` text protocol consumed by the Kotlin
/// `PaddleOcrEngine.parseOutput`. Coordinates are normalised to [0,1] against @p img_w/@p img_h.
std::string BuildProtocol(const std::vector<Line>& lines,
                          const Timings& timings,
                          int img_w,
                          int img_h);

}  // namespace komikun::ocr
