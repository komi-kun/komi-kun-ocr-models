# komi-kun-ocr-models

Reproducible, permissively-licensed native OCR artifact for **komi-kun**'s PaddleOCR pipeline
(`:core:ocr`, `mihon.core.ocr.paddle.PaddleOcrEngine`).

This repository **is** the artifact's source. It builds `libpaddle_ocr.so` — a small JNI shim over
[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite) + [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
(Apache-2.0) — plus the converted models, and publishes them as the `ocr-models-v2` GitHub release
together with a `manifest.json` carrying per-file SHA-256, size, ABI and SPDX id.

It replaces Chimahon's opaque `paddle-ocr.zip` (`v2.5`), whose `libpaddle_ocr.so` source is not
published. Nothing here is derived from Chimahon: the JNI wrapper, the model staging and the
build are ours; only PaddleOCR's Apache-2.0 `deploy/lite` post-processing is vendored.

## Releases

| Release | Models | Notes |
|---|---|---|
| **`ocr-models-v2`** (current) | **PP-OCRv5 mobile** — `ppocrv5_det.nb` + unified `ppocrv5_rec.nb` + `ppocrv5_dict.txt` | The latest PP-OCR generation that runs on Paddle-Lite. The recogniser is a **single** unified model covering Simplified/Traditional Chinese, Pinyin, English **and Japanese**. |
| `ocr-models-v1` | PP-OCRv4 — `ppocrv4_det.nb` + `ppocrv4_japan_rec.nb` + `ppocrv4_japan_dict.txt` | Detection + a dedicated Japanese rec model. |

Both releases use the **same** `libpaddle_ocr.so` sources and the same **Paddle-Lite v2.14-rc**
runtime, and both are **arm-only** (`arm64-v8a`, `armeabi-v7a`).

## Published artifacts (`ocr-models-v2`)

Release assets are a flat namespace (GitHub rewrites `/` to `.`), so the ABI is a name prefix:

```
manifest.json
arm64-v8a_libpaddle_ocr.so        arm64-v8a_libomp.so        arm64-v8a_libc++_shared.so
armeabi-v7a_libpaddle_ocr.so      armeabi-v7a_libomp.so      armeabi-v7a_libc++_shared.so
ppocrv5_det.nb
ppocrv5_rec.nb
ppocrv5_dict.txt
```

Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2`
Each entry's `path` in `manifest.json` / `app-paddle-manifest.json` is the asset name.

App-side wiring (the patches that make `:core:ocr` consume this): **[INTEGRATION.md](INTEGRATION.md)**.

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

The engine is **model-agnostic**: swapping v4 → v5 changed only the three model file names and their
contents, not the `.so`, the JNI surface, the input shape or the dictionary convention.

## Pipeline

DB text detection → per-box perspective crop (`GetRotateCropImage`, which transposes tall crops so
vertical Japanese columns are read horizontally) → CRNN + CTC recognition, using a **single shared
recognition width per page** derived from the page's `max_wh_ratio`, exactly like PaddleOCR's
`predict_rec`. Both PP-OCRv4 and PP-OCRv5 recognition models are CTC (`rec_image_height = 48`).

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

`ocr-models-v2` ships the **official PaddlePaddle PP-OCRv5 mobile** models, already converted to the
Paddle-Lite naive-buffer format **for Paddle-Lite v2.14-rc** (the `.nb` header literally records
`v2.14-rc`). They come from the `feature/paddle-x` branch of
[`Paddle-Lite-Demo`](https://github.com/PaddlePaddle/Paddle-Lite-Demo) — the "PaddleX-Lite-Deploy"
of PaddleOCR's on-device deployment guide.

| File | Source | Notes |
|---|---|---|
| `ppocrv5_det.nb` | `PP-OCRv5_mobile_det.nb` | PP-OCRv5 mobile DB detector (handwriting, vertical, rotated, curved, multi-language). |
| `ppocrv5_rec.nb` | `PP-OCRv5_mobile_rec.nb` | **Unified** PP-OCRv5 mobile recogniser: Simplified + Traditional Chinese, Pinyin, English and **Japanese** in one model (`rec_image_height = 48`, output classes `C = 18385`). This is the PP-OCRv5 Japanese model — there is no separate `japan_PP-OCRv5_*` rec. |
| `ppocrv5_dict.txt` | `ppocr_keys_ocrv5.txt` | 18 383 entries. The engine prepends `#` (CTC blank) and appends a space class, exactly like PaddleOCR's `CTCLabelDecode`. |

### Why not convert them ourselves?

PP-OCRv5's official inference models are published only as **PaddlePaddle 3.x PIR**
(`inference.json` + `inference.pdiparams`), which Paddle-Lite's `opt` 2.14-rc **cannot parse** — it
requires a `.pdmodel` and fails with `Unsupported model format!` / `idx >= blocks.size()`
([Paddle-Lite#10684](https://github.com/PaddlePaddle/Paddle-Lite/issues/10684)). The only
Paddle-Lite-compatible PP-OCRv5 models are PaddlePaddle's own pre-converted `.nb`, which we consume
byte-for-byte and pin by SHA-256. No `manga_*_fp16` fine-tune is used (its provenance/licence is
unverified, implementation contract §8.3).

## Build

```bash
./build.sh            # Android .so for every supported ABI + models + manifests
./build.sh --host     # also build the off-device harness (toolchain check)
./build.sh --abi arm64-v8a
./build.sh --clean
```

Requirements: `cmake >= 3.22`, `ninja`/`make`, `curl`, `unzip`, and the **Android NDK
`29.0.14206865`** (`ANDROID_NDK_HOME`, or `$ANDROID_SDK_ROOT/ndk/...`). Everything else is
downloaded and pinned by `deps.lock.json`.

`build.sh` writes:

* `out/lib/<abi>/{libpaddle_ocr.so,libomp.so,libc++_shared.so}`
* `out/models/{ppocrv5_det.nb,ppocrv5_rec.nb,ppocrv5_dict.txt}`
* `out/manifest.json` — the release manifest, consumed at runtime by `ModelDownloader`
* `out/app-paddle-manifest.json` — the copy that belongs at `core/ocr/paddle-manifest.json`

`libomp.so` and `libc++_shared.so` come from the NDK (Apache-2.0 WITH LLVM-exception); `libc++_shared.so`
is identical to the one AGP already packages for a `-DANDROID_STL=c++_shared` module — see the app
integration notes in `INTEGRATION.md`.

## Validation

The JNI shim, the DB/CRNN orchestration and the model pipeline are identical to `ocr-models-v1`,
which was validated on a synthetic page (horizontal and vertical Japanese) with the host harness.

`ocr-models-v2` was validated as follows:

* **Model contract (executed here, on the host):** the official `PP-OCRv5_mobile_rec` model, run
  through ONNX Runtime, has input `[N,3,48,W]` and output `[N,T,18385]` — confirming the unchanged
  `rec_image_height = 48` and the `['blank'] + ppocr_keys_ocrv5 + [' ']` dictionary convention the
  engine assumes — and recognised four synthetic Japanese lines (kanji, hiragana, katakana) exactly.
* **Runtime match:** the `.nb` files were produced by Paddle-Lite **v2.14-rc**'s `opt` (the header
  records the version) and are consumed by the **same** v2.14-rc `libpaddle_api_light_bundled.a`
  this repository links; PaddlePaddle's own on-device demo runs these exact files on-device.

**Not validated:** `ocr-models-v2` was **not** executed on an ARM device here (no device/emulator
available), and the host conformance harness *cannot* stand in this time: the published models are
`valid_targets=arm`, so loading them under the x86_64 host runtime aborts by design (confirmed:
`SIGABRT`) — the same target-specificity documented under ABI support. `build.sh --host` therefore
only builds the harness (toolchain check), it does not run it. Re-run the v5 models on a real
arm64 device (or add an `x86` model set) before relying on them in production.

## Reproducibility

The build is fully scripted and every input is pinned in `deps.lock.json`, but the `.so` outputs are
**not** byte-identical across runs (NDK artifacts carry build ids). The models, however, *are* the
fixed official `.nb` bytes, pinned by SHA-256. The manifest pins the bytes that were actually
published, and both consumers — the Gradle `fetchPaddleNativeLibs` task and the app's
`ModelDownloader` — verify SHA-256 on every fetch. After republishing, regenerate both manifests with
`./build.sh --manifest-only` and update the app's `core/ocr/paddle-manifest.json`.

## Licensing

* This repository's own code (`src/`, `build.sh`, `CMakeLists.txt`): **Apache-2.0**.
* Vendored `third_party/paddleocr_lite/`: PaddleOCR `deploy/lite`, Apache-2.0.
* `third_party/clipper/`: Clipper 6.4.2, Boost Software License 1.0.
* Inputs: Paddle-Lite and PaddleOCR PP-OCRv5 models (Apache-2.0), opencv-mobile/OpenCV (Apache-2.0),
  Android NDK runtime libraries (Apache-2.0 WITH LLVM-exception).

See `LICENSE` and `NOTICE`.
