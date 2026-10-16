# Vendored third-party sources

Everything in this directory is copied verbatim (or with the minimal patches noted below) from
permissively-licensed upstream projects. None of it is komi-kun's code.

## `paddleocr_lite/`

PaddleOCR's Apache-2.0 mobile reference implementation
(<https://github.com/PaddlePaddle/PaddleOCR>, `deploy/lite/` at tag `v2.7.0`):

| File | Role |
|---|---|
| `crnn_process.{cc,h}` | CRNN crop/resize + dictionary reader |
| `db_post_process.{cc,h}` | DB detection post-processing (contours, min-area boxes, unclip) |
| `cls_process.{cc,h}` | text-direction classifier helpers (kept for reference; not compiled) |

Patches applied:

* `#include "opencv2/imgcodecs.hpp"` removed from `crnn_process.h` and `db_post_process.h` — the
  artifact uses opencv-mobile, which does not ship `imgcodecs`, and never reads/writes image files.

## `clipper/`

Clipper 6.4.2 by Angus Johnson (<http://www.angusj.com>), Boost Software License 1.0, fetched from
Paddle-Lite's mirror. Used by `db_post_process.cc` for polygon offsetting ("unclip").
