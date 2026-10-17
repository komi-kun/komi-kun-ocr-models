# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
#
# Builds `libpaddle_ocr.so` (Android, via the NDK toolchain) or `paddle_ocr_host` (the offline
# conformance harness) from the same engine sources.
#
# Inference runs on ONNX Runtime, loaded **dynamically** at runtime (see `src/ort_runtime.h`), so
# there is no ORT library to link here — only its headers are needed. OpenCV (image ops) and
# Clipper (DB box unclipping) are still linked statically.
#
# Required cache variables (filled in by build.sh):
#   KOMIKUN_ORT_INCLUDE       dir containing onnxruntime_c_api.h
#   KOMIKUN_OPENCV_INCLUDE    dir with opencv2/
#   KOMIKUN_OPENCV_LIB_DIR    dir with libopencv_core.a + libopencv_imgproc.a
#   KOMIKUN_OPENCV_EXTRA_LIBS optional extra static libs (e.g. opencv-mobile's KleidiCV HAL)
#   KOMIKUN_ORT_LIB_DIR       host only: dir with libonnxruntime.so (baked into the rpath)
cmake_minimum_required(VERSION 3.22)
project(paddle_ocr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(KOMIKUN_ORT_INCLUDE "" CACHE PATH "dir containing onnxruntime_c_api.h")
set(KOMIKUN_OPENCV_INCLUDE "" CACHE PATH "opencv-mobile include dir")
set(KOMIKUN_OPENCV_LIB_DIR "" CACHE PATH "opencv-mobile static-lib dir")
set(KOMIKUN_OPENCV_EXTRA_LIBS "" CACHE STRING "extra static libs")
set(KOMIKUN_ORT_LIB_DIR "" CACHE PATH "host: dir with libonnxruntime.so (rpath)")

set(KOMIKUN_OCR_SOURCES
    src/ocr_engine.cpp
    src/ort_runtime.cpp
    third_party/paddleocr_lite/crnn_process.cc
    third_party/paddleocr_lite/db_post_process.cc
    third_party/clipper/clipper.cpp)

set(KOMIKUN_OCR_INCLUDES
    src
    third_party/paddleocr_lite
    third_party/clipper
    ${KOMIKUN_ORT_INCLUDE}
    ${KOMIKUN_OPENCV_INCLUDE})

if(ANDROID)
  add_library(paddle_ocr SHARED src/ocr_jni.cpp ${KOMIKUN_OCR_SOURCES})
  target_include_directories(paddle_ocr PRIVATE ${KOMIKUN_OCR_INCLUDES})

  # OpenMP is still needed: opencv-mobile's `parallel_for_` calls `omp_get_thread_num` /
  # `omp_get_max_threads`. ONNX Runtime itself owns its own thread pool (SetIntraOpNumThreads).
  target_compile_options(paddle_ocr PRIVATE
      -Oz -fopenmp -ffunction-sections -fdata-sections -fvisibility=hidden
      -Wno-deprecated-declarations -Wno-unused-parameter)

  target_link_options(paddle_ocr PRIVATE
      -fopenmp
      -Wl,--gc-sections
      -Wl,--exclude-libs,ALL
      -Wl,-z,max-page-size=16384)

  # `libonnxruntime.so` is deliberately NOT linked: `src/ort_runtime.cpp` dlopen()s it, so the
  # shim has no DT_NEEDED on it and still loads when the runtime is absent.
  target_link_libraries(paddle_ocr PRIVATE
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_imgproc.a
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_core.a
      ${KOMIKUN_OPENCV_EXTRA_LIBS}
      jnigraphics android log z dl m)
else()
  add_executable(paddle_ocr_host src/host_main.cpp ${KOMIKUN_OCR_SOURCES})
  target_include_directories(paddle_ocr_host PRIVATE ${KOMIKUN_OCR_INCLUDES})
  # The parity hook exists only in the host harness; it is never compiled into the shipped `.so`.
  target_compile_definitions(paddle_ocr_host PRIVATE KOMIKUN_OCR_HOST_TEST)
  target_compile_options(paddle_ocr_host PRIVATE -fopenmp)
  target_link_options(paddle_ocr_host PRIVATE -fopenmp)
  target_link_libraries(paddle_ocr_host PRIVATE
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_imgproc.a
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_core.a
      ${KOMIKUN_OPENCV_EXTRA_LIBS}
      pthread dl)
  if(KOMIKUN_ORT_LIB_DIR)
    target_link_options(paddle_ocr_host PRIVATE "-Wl,--disable-new-dtags" "-Wl,-rpath,${KOMIKUN_ORT_LIB_DIR}")
  endif()
endif()
