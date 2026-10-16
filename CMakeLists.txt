# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
#
# Builds `libpaddle_ocr.so` (Android, via the NDK toolchain) or `paddle_ocr_host` (the offline
# conformance harness) from the same engine sources.
#
# Required cache variables (filled in by build.sh):
#   KOMIKUN_LITE_INCLUDE      dir with paddle_api.h / paddle_place.h
#   KOMIKUN_LITE_LIB          libpaddle_api_light_bundled.a
#   KOMIKUN_OPENCV_INCLUDE    dir with opencv2/
#   KOMIKUN_OPENCV_LIB_DIR    dir with libopencv_core.a + libopencv_imgproc.a
#   KOMIKUN_OPENCV_EXTRA_LIBS optional extra static libs (e.g. opencv-mobile's KleidiCV HAL)
cmake_minimum_required(VERSION 3.22)
project(paddle_ocr CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(KOMIKUN_LITE_INCLUDE "" CACHE PATH "Paddle-Lite include dir")
set(KOMIKUN_LITE_LIB "" CACHE FILEPATH "libpaddle_api_light_bundled.a")
set(KOMIKUN_OPENCV_INCLUDE "" CACHE PATH "opencv-mobile include dir")
set(KOMIKUN_OPENCV_LIB_DIR "" CACHE PATH "opencv-mobile static-lib dir")
set(KOMIKUN_OPENCV_EXTRA_LIBS "" CACHE STRING "extra static libs")
set(KOMIKUN_HOST_OPENMP_LIB_DIR "" CACHE PATH "host: dir with Paddle-Lite's bundled libiomp5 (rpath)")

set(KOMIKUN_OCR_SOURCES
    src/ocr_engine.cpp
    third_party/paddleocr_lite/crnn_process.cc
    third_party/paddleocr_lite/db_post_process.cc
    third_party/clipper/clipper.cpp)

set(KOMIKUN_OCR_INCLUDES
    src
    third_party/paddleocr_lite
    third_party/clipper
    ${KOMIKUN_LITE_INCLUDE}
    ${KOMIKUN_OPENCV_INCLUDE})

if(ANDROID)
  add_library(paddle_ocr SHARED src/ocr_jni.cpp ${KOMIKUN_OCR_SOURCES})
  target_include_directories(paddle_ocr PRIVATE ${KOMIKUN_OCR_INCLUDES})

  target_compile_options(paddle_ocr PRIVATE
      -Oz -fopenmp -ffunction-sections -fdata-sections -fvisibility=hidden
      -Wno-deprecated-declarations -Wno-unused-parameter)

  target_link_options(paddle_ocr PRIVATE
      -fopenmp
      -Wl,--gc-sections
      -Wl,--exclude-libs,ALL
      -Wl,-z,max-page-size=16384)

  target_link_libraries(paddle_ocr PRIVATE
      ${KOMIKUN_LITE_LIB}
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_imgproc.a
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_core.a
      ${KOMIKUN_OPENCV_EXTRA_LIBS}
      jnigraphics android log z dl m)
else()
  get_filename_component(KOMIKUN_LITE_LIB_DIR "${KOMIKUN_LITE_LIB}" DIRECTORY)
  add_executable(paddle_ocr_host src/host_main.cpp ${KOMIKUN_OCR_SOURCES})
  target_include_directories(paddle_ocr_host PRIVATE ${KOMIKUN_OCR_INCLUDES})
  target_compile_options(paddle_ocr_host PRIVATE -fopenmp)
  target_link_options(paddle_ocr_host PRIVATE -fopenmp)
  # The host harness links Paddle-Lite's *shared* library: the prebuilt x86_64 static bundle
  # binds the OpenMP symbols to the system libgomp while Paddle-Lite expects its bundled
  # libiomp5, which corrupts its workspace. Android links the static bundle (single runtime).
  target_link_libraries(paddle_ocr_host PRIVATE
      -L${KOMIKUN_LITE_LIB_DIR} -lpaddle_light_api_shared
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_imgproc.a
      ${KOMIKUN_OPENCV_LIB_DIR}/libopencv_core.a
      ${KOMIKUN_OPENCV_EXTRA_LIBS}
      pthread dl)
  target_link_options(paddle_ocr_host PRIVATE "-Wl,--disable-new-dtags" "-Wl,-rpath,${KOMIKUN_LITE_LIB_DIR}")
  if(KOMIKUN_HOST_OPENMP_LIB_DIR)
    target_link_options(paddle_ocr_host PRIVATE "-Wl,-rpath,${KOMIKUN_HOST_OPENMP_LIB_DIR}")
  endif()
endif()
