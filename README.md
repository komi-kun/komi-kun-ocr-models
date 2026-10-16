# komi-kun-ocr-models

Reproducible, permissively-licensed native OCR artifact for **komi-kun**'s PaddleOCR pipeline
(`:core:ocr`, `mihon.core.ocr.paddle.PaddleOcrEngine`).

This repository **is** the artifact's source. It builds `libpaddle_ocr.so` — a small JNI shim over
[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) + [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
(Apache-2.0) — plus the converted models, and publishes them as the `ocr-models-v1` GitHub release
together with a `manifest.json` carrying per-file SHA-256, size, ABI and SPDX id.

It replaces Chimahon's opaque `paddle-ocr.zip` (`v2.5`), whose `libpaddle_ocr.so` source is not
published. Nothing here is derived from Chimahon: the JNI wrapper, the model conversion and the
build are ours; only PaddleOCR's Apache-2.0 `deploy/lite` post-processing is vendored.

## Published artifacts (`ocr-models-v1`)

```
manifest.json
lib/arm64-v8a/libpaddle_ocr.so        lib/arm64-v8a/libomp.so        lib/arm64-v8a/libc++_shared.so
lib/armeabi-v7a/libpaddle_ocr.so      lib/armeabi-v7a/libomp.so      lib/armeabi-v7a/libc++_shared.so
models/ppocrv4_det.nb
models/ppocrv4_japan_rec.nb
models/ppocrv4_japan_dict.txt
```

Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v1`

## What `libpaddle_ocr.so` implements

The exact JNI surface declared by
`core/ocr/src/main/java/mihon/core/ocr/paddle/PaddleOcrEngine.kt`:

| Symbol | Behaviour |
|---|---|
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeInit(detParam, detBin, recParam, recBin, dict, threads)` | Loads the detection + recognition models and the dictionary. `detBin`/`recBin` are accepted for interface compatibility and used only as a fallback path. |
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeProcess(bitmap, thresh, boxThresh, unclipRatio, filterFurigana, threads)` | Runs DB detection + CRNN recognition over an `ARGB_8888` `Bitmap` and returns the `TIMING|`/`LINE|` protocol string. |
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeRelease()` | Frees the predictors. |

Output protocol (parsed by `PaddleOcrEngine.parseOutput`):

```
TIMING|<total>|<prep>|<det>|<post>|<crop>|<rec>|<rawBoxCount>
LINE|<order>|<score>|<L>,<T>,<R>,<B>|<tlx>,<tly>,<trx>,<try>,<brx>,<bry>,<blx>,<bly>|<isVertical>|<isFurigana>|<text>
```

All coordinates are normalised to `[0,1]` over the input image; the quad is `tl,tr,br,bl` and drives
`PaddleOcrEngine.quadRotation`. `isFurigana` is always `0` — ruby filtering is the merger's job.

## Pipeline

DB text detection → per-box perspective crop (`GetRotateCropImage`, which transposes tall crops so
vertical Japanese columns are read horizontally) → CRNN + CTC recognition, using a **single shared
recognition width per page** derived from the page's `max_wh_ratio`, exactly like PaddleOCR's
`predict_rec`.

### Paddle-Lite shape handling (important)

Paddle-Lite's mobile predictor returns **wrong results** (and can segfault) when a `Run()` uses an
input shape other than the previous one. Reproduced on x86_64 with `libpaddle_light_api_shared.so`
(v2.14-rc): the first inference after loading a model is always correct, subsequent inferences with a
changed shape are not.

The engine therefore:

* derives one recognition width for the whole page and pads every crop to it (upstream behaviour), and
* rebuilds a predictor whenever the next inference needs a different shape — loading a model costs
  ~5–10 ms, so this is cheaper than risking a corrupt result.

If you change the engine, keep those invariants.

## ABI support

| ABI | Status | Why |
|---|---|---|
| `arm64-v8a` | shipped | Paddle-Lite prebuilt + `valid_targets=arm` models |
| `armeabi-v7a` | shipped | Paddle-Lite prebuilt + `valid_targets=arm` models |
| `x86_64` | **not shipped** | Paddle-Lite publishes **no Android x86_64 prebuilt** (only armv7/armv8), and `opt`-converted `.nb` models are target-specific: a `valid_targets=arm` model aborts on x86 and vice-versa. Shipping an ARM-only `.so` under an `x86_64` entry would crash, so the ABI is omitted until both halves exist. |

Adding `x86_64` requires (a) building Paddle-Lite from source for Android x86_64 and (b) publishing a
second, `valid_targets=x86` model set plus an ABI-aware model selection step in `ModelDownloader` /
`PaddleOcrEngine`.

## Models

Converted from the **official PaddleOCR Apache-2.0 inference models** with Paddle-Lite's `opt`
(`naive_buffer`, `valid_targets=arm`):

| File | Source | Notes |
|---|---|---|
| `ppocrv4_det.nb` | `ch_PP-OCRv4_det_infer` | PP-OCRv4 DB detector (supports Chinese/English/multilingual text detection) |
| `ppocrv4_japan_rec.nb` | `japan_PP-OCRv4_rec_infer` | PP-OCRv4 Japanese CRNN recogniser (`rec_image_height=48`) |
| `ppocrv4_japan_dict.txt` | `ppocr/utils/dict/japan_dict.txt` @ v2.7.0 | 4399 entries; the engine prepends `#` (CTC blank) and appends a space class |

No `manga_*_fp16` fine-tune is used: its provenance/licence is unverified (implementation contract
§8.3), so the artifact sticks to official models.

## Build

```bash
./build.sh            # Android .so for every supported ABI + models + manifests
./build.sh --host     # also build the off-device validation harness and smoke-test it
./build.sh --abi arm64-v8a
./build.sh --clean
```

Requirements: `cmake >= 3.22`, `ninja`/`make`, `curl`, `unzip`, and the **Android NDK
`29.0.14206865`** (`ANDROID_NDK_HOME`, or `$ANDROID_SDK_ROOT/ndk/...`). Everything else is
downloaded and pinned by `deps.lock.json`.

`build.sh` writes:

* `out/lib/<abi>/{libpaddle_ocr.so,libomp.so,libc++_shared.so}`
* `out/models/{ppocrv4_det.nb,ppocrv4_japan_rec.nb,ppocrv4_japan_dict.txt}`
* `out/manifest.json` — the release manifest, consumed at runtime by `ModelDownloader`
* `out/app-paddle-manifest.json` — the copy that belongs at `core/ocr/paddle-manifest.json`

`libomp.so` and `libc++_shared.so` come from the NDK (Apache-2.0 WITH LLVM-exception); `libc++_shared.so`
is identical to the one AGP already packages for a `-DANDROID_STL=c++_shared` module — see the app
integration notes in `INTEGRATION.md`.

## Off-device validation

`--host` builds `paddle_ocr_host` against Paddle-Lite's x86_64 Linux build and validates the
**identical engine sources** on synthetic Japanese pages (`tests/make_pages.py`). Because converted
models are target-specific, `--host` also produces `out/models-x86/` **for the harness only** (never
published).

Validated: correct detection + recognition of all five horizontal lines of a synthetic page and of a
vertical manga-style page, including across pages of different sizes. Not validated: execution on an
ARM device (no ARM emulator/qemu in the build environment) — the ARM `.so`s and the `arm`-targeted
models were verified only structurally (ELF/ABI, JNI symbols, `DT_NEEDED`).

## Licensing

* This repository's own code (`src/`, `build.sh`, `CMakeLists.txt`): **Apache-2.0**.
* Vendored `third_party/paddleocr_lite/`: PaddleOCR `deploy/lite`, Apache-2.0.
* `third_party/clipper/`: Clipper 6.4.2, Boost Software License 1.0.
* Inputs: Paddle-Lite and PaddleOCR/deploy-lite (Apache-2.0), opencv-mobile/OpenCV (Apache-2.0),
  Android NDK runtime libraries (Apache-2.0 WITH LLVM-exception).

See `LICENSE` and `NOTICE`.
