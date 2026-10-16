// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 komi-kun contributors
//
// Host conformance harness (NOT shipped in the Android artifact). Runs the exact same engine
// sources as `libpaddle_ocr.so` against the x86_64 Paddle-Lite build so the detection /
// recognition / protocol pipeline can be validated without a device.
//
//   paddle_ocr_host <det.nb> <rec.nb> <dict.txt> <image.ppm> [numThreads]
//
// Reads a binary PPM (P6) so no image-codec dependency is needed. ImageMagick / PIL can write
// one: `magick in.png out.ppm`, or `PIL.Image.save("out.ppm")`.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/imgproc.hpp>

#include "ocr_engine.h"

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: " << argv[0] << " <det.nb> <rec.nb> <dict.txt> <image.ppm> [threads]\n";
    return 2;
  }
  const std::string det = argv[1];
  const std::string rec = argv[2];
  const std::string dict = argv[3];
    const int threads = argc > 5 ? std::atoi(argv[5]) : 4;

  komikun::ocr::Engine engine;
  if (!engine.Init(det, rec, dict, threads)) {
    std::cerr << "engine init failed\n";
    return 1;
  }

  for (int a = 4; a < argc; ++a) {
    cv::Mat bgr;
    if (!LoadPpm(argv[a], &bgr)) {
      std::cerr << "failed to read PPM image: " << argv[a] << "\n";
      return 1;
    }
    komikun::ocr::Options options;
    komikun::ocr::Timings timings;
    const auto lines = engine.Run(bgr, options, &timings);
    std::cout << "### " << argv[a] << "\n" << komikun::ocr::BuildProtocol(lines, timings, bgr.cols, bgr.rows);
    std::cerr << argv[a] << " " << bgr.cols << "x" << bgr.rows << " -> " << lines.size() << " lines\n";
  }
  return 0;
}
