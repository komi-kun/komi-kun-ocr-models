// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
#include "ocr_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "crnn_process.h"    // vendored from PaddleOCR deploy/lite (Apache-2.0)
#include "db_post_process.h" // vendored from PaddleOCR deploy/lite (Apache-2.0)

// NOTE: when linking the *bundled* Paddle-Lite static library
// (`libpaddle_api_light_bundled.a`) the op/kernel registrars are already inside the archive, so
// `paddle_use_ops.h` / `paddle_use_kernels.h` must NOT be included (they would be defined twice).

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

using paddle::lite_api::CreatePaddlePredictor;
using paddle::lite_api::MobileConfig;
using paddle::lite_api::PaddlePredictor;

namespace komikun::ocr {
namespace {

using Clock = std::chrono::steady_clock;

/// Upper bound on the shared recognition input width (keeps one pathological wide box from
/// forcing every other crop on the page through the same huge tensor).
constexpr int kMaxRecWidth = 1600;

double Ms(const Clock::time_point& a, const Clock::time_point& b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------------------------
// Pre-processing
// ---------------------------------------------------------------------------------------------

// Fills an NCHW float tensor from interleaved NHWC data, applying per-channel mean/scale. The ARM
// NEON path mirrors PaddleOCR's `NeonMeanScale`; other targets use the scalar fallback (so the
// exact same code builds for x86_64 and the host conformance harness).
void MeanScale(const float* din, float* dout, int size,
               const std::vector<float>& mean, const std::vector<float>& scale) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  const float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  const float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  const float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  const float32x4_t vscale0 = vdupq_n_f32(scale[0]);
  const float32x4_t vscale1 = vdupq_n_f32(scale[1]);
  const float32x4_t vscale2 = vdupq_n_f32(scale[2]);

  float* dout_c0 = dout;
  float* dout_c1 = dout + size;
  float* dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    vst1q_f32(dout_c0, vmulq_f32(vsubq_f32(vin3.val[0], vmean0), vscale0));
    vst1q_f32(dout_c1, vmulq_f32(vsubq_f32(vin3.val[1], vmean1), vscale1));
    vst1q_f32(dout_c2, vmulq_f32(vsubq_f32(vin3.val[2], vmean2), vscale2));
    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; ++i) {
    *(dout_c0++) = (*(din++) - mean[0]) * scale[0];
    *(dout_c1++) = (*(din++) - mean[1]) * scale[1];
    *(dout_c2++) = (*(din++) - mean[2]) * scale[2];
  }
#else
  for (int i = 0; i < size; ++i) {
    dout[i] = (din[i * 3 + 0] - mean[0]) * scale[0];
    dout[size + i] = (din[i * 3 + 1] - mean[1]) * scale[1];
    dout[size * 2 + i] = (din[i * 3 + 2] - mean[2]) * scale[2];
  }
#endif
}

// Resizes the page so its longest side is <= max_size_len and both sides are a multiple of 32
// (the DB backbone's stride). Records the h/w scale factors for mapping boxes back.
cv::Mat DetResize(const cv::Mat& img, int max_size_len, std::vector<float>& ratio_hw) {
  const int w = img.cols;
  const int h = img.rows;

  float ratio = 1.f;
  const int max_wh = std::max(w, h);
  if (max_wh > max_size_len) {
    ratio = static_cast<float>(max_size_len) / static_cast<float>(max_wh);
  }

  const auto round32 = [](int v) {
    if (v % 32 == 0) return v;
    if (v / 32 < 1) return 32;
    return (v / 32) * 32;
  };

  const int resize_h = std::max(round32(static_cast<int>(static_cast<float>(h) * ratio)), 32);
  const int resize_w = std::max(round32(static_cast<int>(static_cast<float>(w) * ratio)), 32);

  cv::Mat out;
  cv::resize(img, out, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return out;
}

// Width the recognition model sees for a given crop (PaddleOCR pads/truncates to this width).
int RecNaturalWidth(const cv::Mat& crop, int rec_image_height) {
  if (crop.rows <= 0) return 0;
  return static_cast<int>(
      std::ceil(static_cast<float>(rec_image_height) * crop.cols / crop.rows));
}

// Resizes a crop to `rec_image_height` (preserving aspect) and pads to `target_width`, exactly
// like PaddleOCR's recognition pre-processing (`predict_rec`'s shared `max_wh_ratio`).
cv::Mat RecResizeTo(const cv::Mat& img, int target_width, int rec_image_height) {
  const float ratio = static_cast<float>(img.cols) / static_cast<float>(img.rows);
  const int resize_w = std::min(target_width,
                                static_cast<int>(std::ceil(rec_image_height * ratio)));
  cv::Mat out;
  cv::resize(img, out, cv::Size(std::max(resize_w, 1), rec_image_height), 0.f, 0.f,
             cv::INTER_LINEAR);
  if (target_width > out.cols) {
    cv::copyMakeBorder(out, out, 0, 0, 0, target_width - out.cols, cv::BORDER_CONSTANT,
                       {127, 127, 127});
  }
  return out;
}

std::map<std::string, double> DbConfig(const Options& options) {
  return {
      {"max_side_len", static_cast<double>(options.max_side_len)},
      {"det_db_thresh", static_cast<double>(options.det_db_thresh)},
      {"det_db_box_thresh", static_cast<double>(options.det_db_box_thresh)},
      {"det_db_unclip_ratio", static_cast<double>(options.det_db_unclip_ratio)},
      {"det_db_use_dilate", options.det_db_use_dilate ? 1.0 : 0.0},
      {"det_use_polygon_score", options.det_use_polygon_score ? 1.0 : 0.0},
  };
}

float Dist(const std::vector<int>& a, const std::vector<int>& b) {
  const float dx = static_cast<float>(a[0] - b[0]);
  const float dy = static_cast<float>(a[1] - b[1]);
  return std::sqrt(dx * dx + dy * dy);
}

bool IsVertical(const std::vector<std::vector<int>>& box) {
  const float width = Dist(box[0], box[1]);
  const float height = Dist(box[0], box[3]);
  return height >= width * 1.5f;
}

// Manga reading order: columns right-to-left for vertical text (the common case), plain
// top-to-bottom/left-to-right otherwise.
std::vector<int> ReadingOrder(const std::vector<std::vector<std::vector<int>>>& boxes) {
  const size_t n = boxes.size();
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  int vertical_count = 0;
  for (const auto& box : boxes) {
    if (IsVertical(box)) ++vertical_count;
  }
  const bool manga_vertical = vertical_count * 2 >= static_cast<int>(n);

  const auto center_x = [&](int i) { return (boxes[i][0][0] + boxes[i][2][0]) / 2.f; };
  const auto center_y = [&](int i) { return (boxes[i][0][1] + boxes[i][2][1]) / 2.f; };

  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (manga_vertical) {
      if (std::abs(center_x(a) - center_x(b)) > 4.f) return center_x(a) > center_x(b);
      return center_y(a) < center_y(b);
    }
    if (std::abs(center_y(a) - center_y(b)) > 4.f) return center_y(a) < center_y(b);
    return center_x(a) < center_x(b);
  });
  return idx;
}

void Sanitize(std::string* text) {
  for (char& c : *text) {
    if (c == '|' || c == '\n' || c == '\r') c = ' ';
  }
}

// Greedy CTC decode over a [T, C] logit matrix. `dict` index 0 is the CTC blank.
std::string DecodeCtc(const float* pred, int T, int C, const std::vector<std::string>& dict,
                      float* out_score) {
  std::string result;
  float score_sum = 0.f;
  int count = 0;
  int last_index = 0;
  for (int n = 0; n < T; ++n) {
    const float* row = pred + static_cast<size_t>(n) * C;
    int argmax = 0;
    float max_value = row[0];
    for (int c = 1; c < C; ++c) {
      if (row[c] > max_value) {
        max_value = row[c];
        argmax = c;
      }
    }
    if (argmax > 0 && !(n > 0 && argmax == last_index)) {
      score_sum += max_value;
      ++count;
      if (argmax < static_cast<int>(dict.size())) result += dict[argmax];
    }
    last_index = argmax;
  }
  *out_score = count > 0 ? score_sum / static_cast<float>(count) : 0.f;
  return result;
}

}  // namespace

// ---------------------------------------------------------------------------------------------

std::vector<std::string> ReadDictionary(const std::string& path) {
  std::vector<std::string> dict;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    dict.push_back(line);
  }
  return dict;
}

Engine::Predictor Engine::LoadModel(const std::string& path) const {
  MobileConfig config;
  config.set_model_from_file(path);
  config.set_threads(threads_);
  config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
  return CreatePaddlePredictor<MobileConfig>(config);
}

void Engine::EnsureShape(bool is_detector, long height, long width) {
  auto& predictor = is_detector ? det_ : rec_;
  const long last_h = is_detector ? det_h_ : rec_h_;
  const long last_w = is_detector ? det_w_ : rec_w_;
  if (predictor != nullptr && last_h >= 0 && (height != last_h || width != last_w)) {
    predictor = LoadModel(is_detector ? det_path_ : rec_path_);
  }
  if (is_detector) {
    det_h_ = height;
    det_w_ = width;
  } else {
    rec_h_ = height;
    rec_w_ = width;
  }
}

bool Engine::Init(const std::string& det_model, const std::string& rec_model,
                  const std::string& dict_path, int num_threads) {
  threads_ = num_threads > 0 ? num_threads : 4;
  det_path_ = det_model;
  rec_path_ = rec_model;
  det_h_ = det_w_ = rec_h_ = rec_w_ = -1;

  det_ = LoadModel(det_model);
  rec_ = LoadModel(rec_model);
  if (det_ == nullptr || rec_ == nullptr) {
    Release();
    return false;
  }

  const std::vector<std::string> raw = ReadDictionary(dict_path);
  if (raw.empty()) {
    Release();
    return false;
  }
  dict_.clear();
  dict_.reserve(raw.size() + 2);
  dict_.emplace_back("#");  // CTC blank
  dict_.insert(dict_.end(), raw.begin(), raw.end());
  dict_.emplace_back(" ");  // trailing space class (PaddleOCR convention)
  return true;
}

void Engine::Release() {
  det_.reset();
  rec_.reset();
  dict_.clear();
  det_h_ = det_w_ = rec_h_ = rec_w_ = -1;
}

std::vector<Line> Engine::Run(const cv::Mat& bgr, const Options& options, Timings* timings) {
  Timings local;
  std::vector<Line> lines;
  if (!ready() || bgr.empty()) {
    if (timings) *timings = local;
    return lines;
  }

  // ---------------------------------------------------------------- detection
  const auto t0 = Clock::now();
  std::vector<float> ratio_hw;
  cv::Mat resized = DetResize(bgr, options.max_side_len, ratio_hw);
  cv::Mat det_fp;
  resized.convertTo(det_fp, CV_32FC3, 1.0 / 255.f);

  EnsureShape(true, det_fp.rows, det_fp.cols);

  {
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(det_->GetInput(0));
    input_tensor->Resize({1, 3, det_fp.rows, det_fp.cols});
    float* data0 = input_tensor->mutable_data<float>();
    MeanScale(reinterpret_cast<const float*>(det_fp.data), data0, det_fp.rows * det_fp.cols,
              {0.485f, 0.456f, 0.406f}, {1 / 0.229f, 1 / 0.224f, 1 / 0.225f});
  }
  const auto t1 = Clock::now();
  local.prep += Ms(t0, t1);

  det_->Run();
  const auto t2 = Clock::now();
  local.det += Ms(t1, t2);

  std::unique_ptr<const paddle::lite_api::Tensor> det_out(det_->GetOutput(0));
  const float* outptr = det_out->data<float>();
  const auto out_shape = det_out->shape();
  const int out_h = static_cast<int>(out_shape[2]);
  const int out_w = static_cast<int>(out_shape[3]);

  cv::Mat pred_map(out_h, out_w, CV_32F);
  cv::Mat bit_map(out_h, out_w, CV_8UC1);
  float* pred_ptr = pred_map.ptr<float>();
  uint8_t* bit_ptr = bit_map.ptr<uint8_t>();
  for (int i = 0; i < out_h * out_w; ++i) {
    const float v = outptr[i];
    pred_ptr[i] = v;
    bit_ptr[i] = static_cast<uint8_t>(std::min(255.f, std::max(0.f, v * 255.f)));
  }

  cv::Mat binary;
  cv::threshold(bit_map, binary, static_cast<double>(options.det_db_thresh) * 255.0, 255,
                cv::THRESH_BINARY);
  if (options.det_db_use_dilate) {
    cv::Mat dilation;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::dilate(binary, dilation, element);
    binary = dilation;
  }

  auto boxes = BoxesFromBitmap(pred_map, binary, DbConfig(options));
  auto filtered = FilterTagDetRes(boxes, ratio_hw[0], ratio_hw[1], bgr);
  local.raw_boxes = static_cast<int>(filtered.size());
  local.post += Ms(t2, Clock::now());

  // ---------------------------------------------------------------- recognition
  const std::vector<int> reading = ReadingOrder(filtered);

  // Crop every box once, then run recognition with a single shared input width for the whole page
  // (PaddleOCR's `predict_rec` `max_wh_ratio`). A stable shape keeps Paddle-Lite's mobile
  // predictor correct across all boxes.
  struct Job {
    cv::Mat crop;
    int order;
    std::vector<std::vector<int>> box;
  };
  std::vector<Job> jobs;
  jobs.reserve(filtered.size());
  int target_width = 0;
  for (size_t rank = 0; rank < reading.size(); ++rank) {
    const int box_index = reading[rank];
    const auto tc0 = Clock::now();
    cv::Mat crop = GetRotateCropImage(bgr, filtered[box_index]);
    local.crop += Ms(tc0, Clock::now());
    if (crop.empty()) continue;
    const int natural = RecNaturalWidth(crop, options.rec_image_height);
    if (natural <= 0) continue;
    target_width = std::max(target_width, natural);
    jobs.push_back({std::move(crop), static_cast<int>(rank), filtered[box_index]});
  }
  if (!jobs.empty()) {
    target_width = std::min(target_width, kMaxRecWidth);
    EnsureShape(false, options.rec_image_height, target_width);
  }

  for (const Job& job : jobs) {
    cv::Mat rec_resized = RecResizeTo(job.crop, target_width, options.rec_image_height);
    cv::Mat rec_fp;
    rec_resized.convertTo(rec_fp, CV_32FC3, 1.0 / 255.f);

    const auto tr0 = Clock::now();
    std::unique_ptr<paddle::lite_api::Tensor> input_tensor(rec_->GetInput(0));
    input_tensor->Resize({1, 3, rec_fp.rows, rec_fp.cols});
    float* data0 = input_tensor->mutable_data<float>();
    MeanScale(reinterpret_cast<const float*>(rec_fp.data), data0, rec_fp.rows * rec_fp.cols,
              {0.5f, 0.5f, 0.5f}, {2.0f, 2.0f, 2.0f});
    rec_->Run();

    std::unique_ptr<const paddle::lite_api::Tensor> rec_out(rec_->GetOutput(0));
    const float* predict_batch = rec_out->data<float>();
    const auto shape = rec_out->shape();
    const int T = static_cast<int>(shape[1]);
    const int C = static_cast<int>(shape[2]);

    float score = 0.f;
    std::string text = DecodeCtc(predict_batch, T, C, dict_, &score);
    local.rec += Ms(tr0, Clock::now());

    Sanitize(&text);
    if (text.empty()) continue;

    Line line;
    line.text = std::move(text);
    line.score = score;
    line.order = job.order;
    for (int p = 0; p < 4; ++p) {
      line.quad[p] = {static_cast<float>(job.box[p][0]), static_cast<float>(job.box[p][1])};
    }
    line.vertical = IsVertical(job.box);
    lines.push_back(std::move(line));
  }

  // Emit in reading order.
  std::stable_sort(lines.begin(), lines.end(),
                   [](const Line& a, const Line& b) { return a.order < b.order; });

  if (timings) *timings = local;
  return lines;
}

std::string BuildProtocol(const std::vector<Line>& lines, const Timings& timings, int img_w,
                          int img_h) {
  const double total = timings.prep + timings.det + timings.post + timings.crop + timings.rec;
  std::string out;
  out.reserve(lines.size() * 96 + 64);

  char head[192];
  std::snprintf(head, sizeof(head), "TIMING|%ld|%ld|%ld|%ld|%ld|%ld|%d\n",
                static_cast<long>(total), static_cast<long>(timings.prep),
                static_cast<long>(timings.det), static_cast<long>(timings.post),
                static_cast<long>(timings.crop), static_cast<long>(timings.rec),
                timings.raw_boxes);
  out += head;

  const float iw = img_w > 0 ? static_cast<float>(img_w) : 1.f;
  const float ih = img_h > 0 ? static_cast<float>(img_h) : 1.f;

  for (size_t i = 0; i < lines.size(); ++i) {
    const Line& line = lines[i];
    float xmin = line.quad[0][0], xmax = line.quad[0][0];
    float ymin = line.quad[0][1], ymax = line.quad[0][1];
    for (const auto& p : line.quad) {
      xmin = std::min(xmin, p[0]);
      xmax = std::max(xmax, p[0]);
      ymin = std::min(ymin, p[1]);
      ymax = std::max(ymax, p[1]);
    }

    char buf[4096];
    std::snprintf(
        buf, sizeof(buf),
        "LINE|%d|%.4f|%.6f,%.6f,%.6f,%.6f|%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f|%d|%d|%s\n",
        line.order, line.score, xmin / iw, ymin / ih, xmax / iw, ymax / ih, line.quad[0][0] / iw,
        line.quad[0][1] / ih, line.quad[1][0] / iw, line.quad[1][1] / ih, line.quad[2][0] / iw,
        line.quad[2][1] / ih, line.quad[3][0] / iw, line.quad[3][1] / ih, line.vertical ? 1 : 0, 0,
        line.text.c_str());
    out += buf;
  }
  return out;
}

}  // namespace komikun::ocr
