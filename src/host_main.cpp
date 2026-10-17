// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// Host conformance harness (NOT shipped in the Android artifact). Runs the exact same engine
// sources as `libpaddle_ocr.so`, against the Linux x86_64 ONNX Runtime build, so the detection /
// recognition / protocol pipeline can be validated without a device.
//
// Because ONNX Runtime is portable and the models are architecture-independent ONNX graphs, this
// harness exercises *the same code and the same model bytes* the device runs — unlike the
// Paddle-Lite artifact, whose `.nb` models were `valid_targets=arm` and therefore could not be
// loaded on the host at all.
//
// Usage:
//   paddle_ocr_host full   <det.onnx> <rec.onnx> <dict.txt> <page.ppm> [page.ppm ...]
//       Full pipeline per page; prints the `TIMING|`/`LINE|` protocol for each.
//
//   paddle_ocr_host parity <det.onnx> <rec.onnx> <dict.txt> <list.txt>
//       Recognition-only parity mode. `list.txt` has one `<crop.ppm>\t<0|1>` line per crop
//       (1 = vertical/tategaki, which is rotated 90 deg CCW exactly as the benchmark does).
//       Prints `TEXT\t<crop.ppm>\t<text>` so a script can diff against the recorded benchmark.
//
//   paddle_ocr_host info   <det.onnx> <rec.onnx> <dict.txt>
//       Prints the loaded ORT version and confirms both sessions initialise.
//
// Images are binary PPM (P6) so no image-codec dependency is needed; PIL/OpenCV can write them:
//   `python3 -c "import cv2;cv2.imwrite('x.ppm', cv2.imread('x.png'))"`.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "ocr_engine.h"
#include "ort_runtime.h"

namespace {

bool LoadPpm(const std::string& path, cv::Mat* out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string magic;
  int width = 0;
  int height = 0;
  int maxval = 0;
  in >> magic >> width >> height >> maxval;
  if (!in || magic != "P6" || maxval != 255 || width <= 0 || height <= 0) return false;
  in.get();  // single whitespace after maxval
  cv::Mat rgb(height, width, CV_8UC3);
  in.read(reinterpret_cast<char*>(rgb.data), static_cast<std::streamsize>(rgb.total() * 3));
  if (!in) return false;
  cv::cvtColor(rgb, *out, cv::COLOR_RGB2BGR);
  return true;
}

int Usage() {
  std::cerr << "usage:\n"
            << "  paddle_ocr_host full   <det.onnx> <rec.onnx> <dict.txt> <page.ppm> [...]\n"
            << "  paddle_ocr_host parity <det.onnx> <rec.onnx> <dict.txt> <list.txt>\n"
            << "  paddle_ocr_host info   <det.onnx> <rec.onnx> <dict.txt>\n";
  return 2;
}

/// Shared: engine + ORT banner on stderr.
bool InitEngine(const char* const* argv, komikun::ocr::Engine* engine, int threads) {
  if (!engine->Init(argv[2], argv[3], argv[4], threads)) {
    std::cerr << "engine init failed\n";
    return false;
  }
  std::cerr << "onnxruntime " << komikun::ocr::ort::RuntimeVersion() << "\n";
  return true;
}

int RunFull(int argc, char** argv, int threads) {
  komikun::ocr::Engine engine;
  if (!InitEngine(argv, &engine, threads)) return 1;

  for (int a = 5; a < argc; ++a) {
    cv::Mat bgr;
    if (!LoadPpm(argv[a], &bgr)) {
      std::cerr << "failed to read PPM image: " << argv[a] << "\n";
      return 1;
    }
    komikun::ocr::Options options;
    komikun::ocr::Timings timings;
    const auto lines = engine.Run(bgr, options, &timings);
    std::cout << "### " << argv[a] << "\n"
              << komikun::ocr::BuildProtocol(lines, timings, bgr.cols, bgr.rows);
    std::cerr << argv[a] << " " << bgr.cols << "x" << bgr.rows << " -> " << lines.size()
              << " lines\n";
  }
  return 0;
}

int RunParity(int argc, char** argv, int threads) {
  if (argc < 6) return Usage();
  komikun::ocr::Engine engine;
  if (!InitEngine(argv, &engine, threads)) return 1;

  std::ifstream list(argv[5]);
  if (!list) {
    std::cerr << "cannot read crop list: " << argv[5] << "\n";
    return 1;
  }
  std::string line;
  while (std::getline(list, line)) {
    if (line.empty()) continue;
    const size_t tab = line.find('\t');
    const std::string path = tab == std::string::npos ? line : line.substr(0, tab);
    const bool vertical = tab != std::string::npos && line.find('1', tab + 1) != std::string::npos;
    cv::Mat crop;
    if (!LoadPpm(path, &crop)) {
      std::cerr << "failed to read crop: " << path << "\n";
      return 1;
    }
    if (vertical) {
      cv::Mat rotated;
      cv::rotate(crop, rotated, cv::ROTATE_90_COUNTERCLOCKWISE);
      crop = rotated;
    }
    std::string text;
    float score = 0.f;
    if (!engine.RecognizeCropParity(crop, &text, &score)) {
      std::cerr << "recognition failed for " << path << "\n";
      return 1;
    }
    // `|` and newlines would break the record separator; the engine sanitises them the same way.
    for (char& c : text) {
      if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    }
    std::cout << "TEXT\t" << path << "\t" << text << "\n";
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) return Usage();
  const std::string mode = argv[1];
  const int threads = getenv("KOMIKUN_OCR_THREADS") != nullptr
                          ? std::atoi(getenv("KOMIKUN_OCR_THREADS"))
                          : 4;

  if (mode == "full") {
    if (argc < 6) return Usage();
    return RunFull(argc, argv, threads);
  }
  if (mode == "parity") return RunParity(argc, argv, threads);
  if (mode == "info") {
    komikun::ocr::Engine engine;
    if (!engine.Init(argv[2], argv[3], argv[4], threads)) {
      std::cerr << "engine init failed\n";
      return 1;
    }
    std::cout << "onnxruntime " << komikun::ocr::ort::RuntimeVersion() << "\n";
    std::cout << "det=" << argv[2] << " rec=" << argv[3] << " dict=" << argv[4] << "\n";
    return 0;
  }
  return Usage();
}
