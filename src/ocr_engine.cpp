// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
#include "ocr_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include "crnn_process.h"    // vendored from PaddleOCR deploy/lite (Apache-2.0)
#include "db_post_process.h" // vendored from PaddleOCR deploy/lite (Apache-2.0)
#include "ort_runtime.h"

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace komikun::ocr {
namespace {

using Clock = std::chrono::steady_clock;

/// Upper bound on the recognition input width. Identical to the validated benchmark's `max_w`
/// (`rec_preprocess(..., max_w=3200)`); keeps one pathological wide crop from allocating a huge
/// tensor.
constexpr int kMaxRecWidth = 3200;

double Ms(const Clock::time_point& a, const Clock::time_point& b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// ---------------------------------------------------------------------------------------------
// ONNX Runtime handles
// ---------------------------------------------------------------------------------------------

/// RAII for `OrtValue`.
struct ValueGuard {
  const OrtApi* api;
  OrtValue* value;
  ~ValueGuard() {
    if (value != nullptr) api->ReleaseValue(value);
  }
};

/// RAII for `OrtTensorTypeAndShapeInfo`.
struct ShapeGuard {
  const OrtApi* api;
  OrtTensorTypeAndShapeInfo* info;
  ~ShapeGuard() {
    if (info != nullptr) api->ReleaseTensorTypeAndShapeInfo(info);
  }
};

/// RAII for `OrtSessionOptions`.
struct SessionOptionsGuard {
  const OrtApi* api;
  OrtSessionOptions* options;
  ~SessionOptionsGuard() {
    if (options != nullptr) api->ReleaseSessionOptions(options);
  }
};

// ---------------------------------------------------------------------------------------------
// Pre-processing (byte-for-byte the validated benchmark, /tmp/opencode/ocr-bench/common.py)
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
// (the DB backbone's stride), rounding to the nearest multiple exactly like PaddleOCR's
// `DetResizeForTest` and the benchmark. Records the h/w scale factors for mapping boxes back.
cv::Mat DetResize(const cv::Mat& img, int max_size_len, std::vector<float>& ratio_hw) {
  const int w = img.cols;
  const int h = img.rows;

  float ratio = 1.f;
  const int max_wh = std::max(w, h);
  if (max_wh > max_size_len) {
    ratio = static_cast<float>(max_size_len) / static_cast<float>(max_wh);
  }

  const auto round32 = [](double v) {
    return std::max(32, static_cast<int>(std::lround(v / 32.0)) * 32);
  };

  const int resize_h = round32(static_cast<double>(h) * ratio);
  const int resize_w = round32(static_cast<double>(w) * ratio);

  cv::Mat out;
  cv::resize(img, out, cv::Size(resize_w, resize_h));

  ratio_hw.push_back(static_cast<float>(resize_h) / static_cast<float>(h));
  ratio_hw.push_back(static_cast<float>(resize_w) / static_cast<float>(w));
  return out;
}

// Resizes a crop to `rec_image_height` preserving the aspect ratio (PaddleOCR's recognition
// pre-processing, and the benchmark's `rec_preprocess`). Width is `ceil(h_rec * w / h)` clamped to
// `[1, kMaxRecWidth]` — no padding: ONNX Runtime accepts the dynamic width directly.
cv::Mat RecResize(const cv::Mat& img, int rec_image_height) {
  if (img.rows <= 0 || img.cols <= 0) return {};
  const double ratio = static_cast<double>(img.cols) / static_cast<double>(img.rows);
  int resize_w = static_cast<int>(std::ceil(static_cast<double>(rec_image_height) * ratio));
  resize_w = std::clamp(resize_w, 1, kMaxRecWidth);
  cv::Mat out;
  cv::resize(img, out, cv::Size(resize_w, rec_image_height), 0.f, 0.f, cv::INTER_LINEAR);
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
// Engine lifecycle
// ---------------------------------------------------------------------------------------------

namespace {

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

}  // namespace

Engine::~Engine() { Release(); }

void Engine::Release() {
  if (api_ != nullptr) {
    if (det_ != nullptr) api_->ReleaseSession(det_);
    if (rec_ != nullptr) api_->ReleaseSession(rec_);
  }
  det_ = nullptr;
  rec_ = nullptr;
  dict_.clear();
}

/// Creates one ORT session and records its (single) input and output names.
namespace {

bool CreateSession(const OrtApi* api, OrtEnv* env, const std::string& path, int threads,
                   OrtSession** out_session, std::string* out_input, std::string* out_output,
                   std::string* out_error) {
  SessionOptionsGuard options_guard{api, nullptr};
  OrtSessionOptions* options = nullptr;
  auto fail = [&](OrtStatus* status) {
    if (status == nullptr) return false;
    *out_error = api->GetErrorMessage(status);
    api->ReleaseStatus(status);
    return true;
  };

  if (fail(api->CreateSessionOptions(&options))) return false;
  options_guard.options = options;
  if (fail(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL))) return false;
  if (fail(api->SetIntraOpNumThreads(options, threads))) return false;
  // ORT's own default log severity is WARNING; keep the app log quiet.
  if (fail(api->SetSessionLogSeverityLevel(options, 3 /* ORT_LOGGING_LEVEL_ERROR */))) return false;

  OrtSession* session = nullptr;
  if (fail(api->CreateSession(env, path.c_str(), options, &session))) return false;

  OrtAllocator* allocator = nullptr;
  if (fail(api->GetAllocatorWithDefaultOptions(&allocator))) {
    api->ReleaseSession(session);
    return false;
  }
  const auto name_at = [&](size_t index, bool input, std::string* out) {
    char* name = nullptr;
    OrtStatus* status = input ? api->SessionGetInputName(session, index, allocator, &name)
                              : api->SessionGetOutputName(session, index, allocator, &name);
    if (fail(status)) return false;
    *out = name != nullptr ? name : "";
    if (name != nullptr && api->AllocatorFree(allocator, name) != nullptr) {
      // AllocatorFree only fails for an allocator that cannot free; the default one never does.
    }
    return true;
  };
  if (!name_at(0, /*input=*/true, out_input) || !name_at(0, /*input=*/false, out_output)) {
    api->ReleaseSession(session);
    return false;
  }
  *out_session = session;
  return true;
}

}  // namespace

bool Engine::CreateSessions() {
  api_ = ort::Api();
  if (api_ == nullptr) {
    ort_error_ = ort::LoadError();
    return false;
  }

  // The env owns the (intra-op) thread pools and is created once per process.
  if (env_ == nullptr) {
    if (api_->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "komikun_ocr", &env_) != nullptr) {
      ort_error_ = "CreateEnv failed";
      return false;
    }
  }
  if (memory_ == nullptr) {
    if (api_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_) != nullptr) {
      ort_error_ = "CreateCpuMemoryInfo failed";
      return false;
    }
  }
  if (run_options_ == nullptr) {
    if (api_->CreateRunOptions(&run_options_) != nullptr) {
      ort_error_ = "CreateRunOptions failed";
      return false;
    }
  }

  std::string det_error;
  if (!CreateSession(api_, env_, det_path_, threads_, &det_, &det_input_, &det_output_,
                     &det_error)) {
    ort_error_ = "det session: " + det_error;
    Release();
    return false;
  }
  std::string rec_error;
  if (!CreateSession(api_, env_, rec_path_, threads_, &rec_, &rec_input_, &rec_output_,
                     &rec_error)) {
    ort_error_ = "rec session: " + rec_error;
    Release();
    return false;
  }
  return true;
}

bool Engine::Init(const std::string& det_model, const std::string& rec_model,
                  const std::string& dict_path, int num_threads) {
  if (ready()) return true;

  threads_ = num_threads > 0 ? num_threads : 4;
  det_path_ = det_model;
  rec_path_ = rec_model;

  const std::vector<std::string> raw = ReadDictionary(dict_path);
  if (raw.empty()) {
    ort_error_ = "empty or missing dictionary: " + dict_path;
    return false;
  }
  dict_.clear();
  dict_.reserve(raw.size() + 2);
  dict_.emplace_back("#");  // CTC blank
  dict_.insert(dict_.end(), raw.begin(), raw.end());
  dict_.emplace_back(" ");  // trailing space class (PaddleOCR convention)

  return CreateSessions();
}

// ---------------------------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------------------------

bool Engine::RunDetector(const cv::Mat& bgr, const Options& options, cv::Mat* pred_map,
                         std::vector<float>* ratio_hw, double* prep_ms, double* det_ms) {
  const auto t0 = Clock::now();
  cv::Mat resized = DetResize(bgr, options.max_side_len, *ratio_hw);
  cv::Mat det_fp;
  resized.convertTo(det_fp, CV_32FC3, 1.0 / 255.f);

  const int height = det_fp.rows;
  const int width = det_fp.cols;
  std::vector<float> input(static_cast<size_t>(3) * height * width);
  MeanScale(reinterpret_cast<const float*>(det_fp.data), input.data(), height * width,
            {0.485f, 0.456f, 0.406f}, {1 / 0.229f, 1 / 0.224f, 1 / 0.225f});
  *prep_ms = Ms(t0, Clock::now());

  const int64_t shape[4] = {1, 3, height, width};
  OrtValue* value = nullptr;
  if (api_->CreateTensorWithDataAsOrtValue(memory_, input.data(), input.size() * sizeof(float),
                                           shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           &value) != nullptr) {
    ort_error_ = "det CreateTensorWithDataAsOrtValue failed";
    return false;
  }
  ValueGuard value_guard{api_, value};

  const auto t1 = Clock::now();
  const char* input_names[1] = {det_input_.c_str()};
  const char* output_names[1] = {det_output_.c_str()};
  OrtValue* outputs[1] = {nullptr};
  if (api_->Run(det_, run_options_, input_names, &value, 1, output_names, 1, outputs) != nullptr) {
    ort_error_ = "det Run failed";
    return false;
  }
  ValueGuard output_guard{api_, outputs[0]};
  *det_ms = Ms(t1, Clock::now());

  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (api_->GetTensorTypeAndShape(outputs[0], &info) != nullptr) {
    ort_error_ = "det GetTensorTypeAndShape failed";
    return false;
  }
  ShapeGuard info_guard{api_, info};
  size_t dims_count = 0;
  if (api_->GetDimensionsCount(info, &dims_count) != nullptr || dims_count < 4) {
    ort_error_ = "det unexpected output rank";
    return false;
  }
  std::vector<int64_t> dims(dims_count);
  if (api_->GetDimensions(info, dims.data(), dims_count) != nullptr) {
    ort_error_ = "det GetDimensions failed";
    return false;
  }
  const int out_h = static_cast<int>(dims[2]);
  const int out_w = static_cast<int>(dims[3]);

  float* data = nullptr;
  if (api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)) != nullptr ||
      data == nullptr) {
    ort_error_ = "det GetTensorMutableData failed";
    return false;
  }
  *pred_map = cv::Mat(out_h, out_w, CV_32F);
  std::memcpy(pred_map->ptr<float>(), data,
              static_cast<size_t>(out_h) * out_w * sizeof(float));
  return true;
}

bool Engine::RecognizeCrop(const cv::Mat& crop, std::string* text, float* score) {
  const cv::Mat resized = RecResize(crop, 48);
  if (resized.empty()) return false;
  cv::Mat rec_fp;
  resized.convertTo(rec_fp, CV_32FC3, 1.0 / 255.f);

  const int height = rec_fp.rows;
  const int width = rec_fp.cols;
  std::vector<float> input(static_cast<size_t>(3) * height * width);
  MeanScale(reinterpret_cast<const float*>(rec_fp.data), input.data(), height * width,
            {0.5f, 0.5f, 0.5f}, {2.0f, 2.0f, 2.0f});

  const int64_t shape[4] = {1, 3, height, width};
  OrtValue* value = nullptr;
  if (api_->CreateTensorWithDataAsOrtValue(memory_, input.data(), input.size() * sizeof(float),
                                           shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           &value) != nullptr) {
    ort_error_ = "rec CreateTensorWithDataAsOrtValue failed";
    return false;
  }
  ValueGuard value_guard{api_, value};

  const char* input_names[1] = {rec_input_.c_str()};
  const char* output_names[1] = {rec_output_.c_str()};
  OrtValue* outputs[1] = {nullptr};
  if (api_->Run(rec_, run_options_, input_names, &value, 1, output_names, 1, outputs) != nullptr) {
    ort_error_ = "rec Run failed";
    return false;
  }
  ValueGuard output_guard{api_, outputs[0]};

  OrtTensorTypeAndShapeInfo* info = nullptr;
  if (api_->GetTensorTypeAndShape(outputs[0], &info) != nullptr) {
    ort_error_ = "rec GetTensorTypeAndShape failed";
    return false;
  }
  ShapeGuard info_guard{api_, info};
  size_t dims_count = 0;
  if (api_->GetDimensionsCount(info, &dims_count) != nullptr || dims_count < 3) {
    ort_error_ = "rec unexpected output rank";
    return false;
  }
  std::vector<int64_t> dims(dims_count);
  if (api_->GetDimensions(info, dims.data(), dims_count) != nullptr) {
    ort_error_ = "rec GetDimensions failed";
    return false;
  }
  const int steps = static_cast<int>(dims[1]);
  const int classes = static_cast<int>(dims[2]);

  float* data = nullptr;
  if (api_->GetTensorMutableData(outputs[0], reinterpret_cast<void**>(&data)) != nullptr ||
      data == nullptr) {
    ort_error_ = "rec GetTensorMutableData failed";
    return false;
  }
  *text = DecodeCtc(data, steps, classes, dict_, score);
  return true;
}

#ifdef KOMIKUN_OCR_HOST_TEST
bool Engine::RecognizeCropParity(const cv::Mat& bgr_crop, std::string* text, float* score) {
  if (!ready() || bgr_crop.empty()) return false;
  return RecognizeCrop(bgr_crop, text, score);
}
#endif

std::vector<Line> Engine::Run(const cv::Mat& bgr, const Options& options, Timings* timings) {
  Timings local;
  std::vector<Line> lines;
  if (!ready() || bgr.empty()) {
    if (timings) *timings = local;
    return lines;
  }

  // ---------------------------------------------------------------- detection
  cv::Mat pred_map;
  std::vector<float> ratio_hw;
  if (!RunDetector(bgr, options, &pred_map, &ratio_hw, &local.prep, &local.det)) {
    if (timings) *timings = local;
    return lines;
  }
  const auto t2 = Clock::now();

  cv::Mat bit_map(pred_map.size(), CV_8UC1);
  {
    float* pred_ptr = pred_map.ptr<float>();
    uint8_t* bit_ptr = bit_map.ptr<uint8_t>();
    const int total = pred_map.rows * pred_map.cols;
    for (int i = 0; i < total; ++i) {
      bit_ptr[i] = static_cast<uint8_t>(
          std::min(255.f, std::max(0.f, pred_ptr[i] * 255.f)));
    }
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

  struct Job {
    cv::Mat crop;
    int order;
    std::vector<std::vector<int>> box;
  };
  std::vector<Job> jobs;
  jobs.reserve(filtered.size());
  for (size_t rank = 0; rank < reading.size(); ++rank) {
    const int box_index = reading[rank];
    const auto tc0 = Clock::now();
    cv::Mat crop = GetRotateCropImage(bgr, filtered[box_index]);
    local.crop += Ms(tc0, Clock::now());
    if (crop.empty()) continue;
    jobs.push_back({std::move(crop), static_cast<int>(rank), filtered[box_index]});
  }

  for (const Job& job : jobs) {
    const auto tr0 = Clock::now();
    float score = 0.f;
    std::string text;
    const bool ok = RecognizeCrop(job.crop, &text, &score);
    local.rec += Ms(tr0, Clock::now());
    if (!ok) continue;

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
