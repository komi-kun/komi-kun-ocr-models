// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// PaddleOCR (Paddle-Lite) text detection + recognition engine for komi-kun.
//
// This is the portable, JNI-free core: it loads a DB text-detection model and a CRNN
// text-recognition model converted to Paddle-Lite naive-buffer format (`opt` output) and runs the
// standard PaddleOCR mobile pipeline over an in-memory image. The detection post-processing
// (`BoxesFromBitmap`/`FilterTagDetRes`/`Unclip`) and the recognition crop/resize come from
// PaddleOCR's Apache-2.0 `deploy/lite` reference implementation, vendored under
// `third_party/paddleocr_lite/`. The engine itself (model loading, orchestration, portability
// fixes, shape handling, protocol output) is ours.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "paddle_api.h"

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
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /// Loads the detection + recognition models and the recognition dictionary.
  bool Init(const std::string& det_model,
            const std::string& rec_model,
            const std::string& dict_path,
            int num_threads);

  bool ready() const { return det_ != nullptr && rec_ != nullptr && !dict_.empty(); }

  void Release();

  /// Runs full-page OCR. @p bgr is a CV_8UC3 image in BGR order (as `cv::imread` produces).
  std::vector<Line> Run(const cv::Mat& bgr, const Options& options, Timings* timings);

 private:
  using Predictor = std::shared_ptr<paddle::lite_api::PaddlePredictor>;

  Predictor LoadModel(const std::string& path) const;

  /// Paddle-Lite's mobile predictor returns wrong results (and can crash) when a `Run()` uses a
  /// different input shape than the previous one. Loading a model is only ~5 ms, so the predictor
  /// is rebuilt whenever the next inference needs a different shape. `Run()` also derives a single
  /// recognition width for the whole page (PaddleOCR's `predict_rec` `max_wh_ratio`) so the
  /// recognition shape is stable across all boxes of a page.
  void EnsureShape(bool is_detector, long height, long width);

  std::shared_ptr<paddle::lite_api::PaddlePredictor> det_;
  std::shared_ptr<paddle::lite_api::PaddlePredictor> rec_;
  std::string det_path_;
  std::string rec_path_;
  std::vector<std::string> dict_;  // index 0 == CTC blank, last == space
  int threads_ = 4;
  long det_h_ = 0;
  long det_w_ = 0;
  long rec_h_ = 0;
  long rec_w_ = 0;
};

/// Serialises the engine result into the `TIMING|`/`LINE|` text protocol consumed by the Kotlin
/// `PaddleOcrEngine.parseOutput`. Coordinates are normalised to [0,1] against @p img_w/@p img_h.
std::string BuildProtocol(const std::vector<Line>& lines,
                          const Timings& timings,
                          int img_w,
                          int img_h);

}  // namespace komikun::ocr
