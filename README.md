# komi-kun-ocr-models

Reproducible, permissively-licensed native OCR artifact for **komi-kun**'s OCR pipeline
(`:core:ocr`, `mihon.core.ocr.paddle.PaddleOcrEngine`).

This repository **is** the artifact's source. It builds `libpaddle_ocr.so` — a small JNI shim over
[ONNX Runtime](https://github.com/microsoft/onnxruntime) (MIT) + [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
Apache-2.0 detection/recognition models — plus the ONNX models and the ONNX Runtime shared library,
and publishes them as the `ocr-models-v3` GitHub release together with a `manifest.json` carrying
per-file SHA-256, size, ABI and SPDX id.

It replaces Chimahon's opaque `paddle-ocr.zip` (`v2.5`), whose `libpaddle_ocr.so` source is not
published. Nothing here is derived from Chimahon: the JNI shim, the model staging and the build are
ours; only PaddleOCR's Apache-2.0 `deploy/lite` post-processing is vendored.

## Releases

| Release | Status | Backend | Models | ABIs |
|---|---|---|---|---|
| **`ocr-models-v3`** | **current** | **ONNX Runtime 1.31.0** | **PP-OCRv6 small** — `ppocrv6_small_det.onnx` + `ppocrv6_small_rec.onnx` + `ppocrv6_small_dict.txt` | `arm64-v8a`, `armeabi-v7a`, **`x86_64`** |
| `ocr-models-v2` | superseded | Paddle-Lite v2.14-rc | PP-OCRv5 mobile (`ppocrv5_*.nb`) | `arm64-v8a`, `armeabi-v7a` |
| `ocr-models-v1` | superseded | Paddle-Lite v2.14-rc | PP-OCRv4 (`ppocrv4_*.nb`) | `arm64-v8a`, `armeabi-v7a` |

`v3` is a backend swap **and** a model bump: ONNX Runtime replaces Paddle-Lite (so all three ABIs —
including `x86_64` — are first-class), and PP-OCRv6 small replaces PP-OCRv5 mobile.

### Superseded releases (`ocr-models-v1`, `ocr-models-v2`)

Both are **superseded** by `ocr-models-v3` and are kept only for history: the releases and their
tags stay intact, and the app deletes the old model files on upgrade
(`ModelDownloader.cleanupStaleModels()`). They ran on **Paddle-Lite v2.14-rc** (the `opt`
`naive_buffer` converter + prebuilt Android inference libraries) and were **ARM-only** —
Paddle-Lite publishes no Android `x86_64` prebuilt and its converted models are target-specific, so
there was no `x86_64` entry:

* `ocr-models-v1` — PP-OCRv4: `ppocrv4_det.nb`, `ppocrv4_japan_rec.nb`, `ppocrv4_japan_dict.txt`.
* `ocr-models-v2` — PP-OCRv5 mobile: `ppocrv5_det.nb`, `ppocrv5_rec.nb`, `ppocrv5_dict.txt`.

The Paddle-Lite build material for them — the former `build.sh`/`CMakeLists.txt`, the `opt`
conversion step, the x86_64 caveat and the `.nb` pipeline notes — is **not deleted, it is preserved
in those tags** and is the accurate description of those releases:

```bash
git show ocr-models-v1:README.md      # Paddle-Lite build, `.nb` models, ARM-only ABIs
git show ocr-models-v1:build.sh       # the `opt` conversion + Paddle-Lite prebuilt fetch
git show ocr-models-v1:NOTICE         # the component list those releases shipped
git diff ocr-models-v2..main          # the full Paddle-Lite → ONNX Runtime change
```

`main` (this README, `NOTICE`, `deps.lock.json`, `CMakeLists.txt`, `build.sh`) describes **only the
current `ocr-models-v3` build**. For `v1`/`v2` the authoritative documents are the files at their
tags, not this branch. The one Paddle-Lite caveat worth remembering is that `opt` wrote a
non-deterministic header into the converted `.nb`, so those releases were never byte-reproducible
(the current v3 build does not use `opt` at all — see [Reproducibility](#reproducibility)).

## Published artifacts (`ocr-models-v3`)

Release assets are a flat namespace (GitHub rewrites `/` to `.`), so the ABI is a name prefix:

```
manifest.json                                        app-paddle-manifest.json
arm64-v8a_libpaddle_ocr.so      arm64-v8a_libonnxruntime.so      arm64-v8a_libomp.so   arm64-v8a_libc++_shared.so
armeabi-v7a_libpaddle_ocr.so    armeabi-v7a_libonnxruntime.so    armeabi-v7a_libomp.so armeabi-v7a_libc++_shared.so
x86_64_libpaddle_ocr.so         x86_64_libonnxruntime.so         x86_64_libomp.so      x86_64_libc++_shared.so
ppocrv6_small_det.onnx
ppocrv6_small_rec.onnx
ppocrv6_small_dict.txt
```

Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v3`
Each entry's `path` in `manifest.json` / `app-paddle-manifest.json` is the asset name.

App-side wiring (the patches that make `:core:ocr` consume this): **[INTEGRATION.md](INTEGRATION.md)**.

## How ONNX Runtime is provisioned (the call, and why)

**Decision: this release ships `libonnxruntime.so` per ABI and the app consumes it through the
existing checksum-pinned build-time fetch (`:core:ocr:fetchPaddleNativeLibs` → `jniLibs/`). The app
does *not* take `com.microsoft.onnxruntime:onnxruntime-android` as a Gradle dependency.**

Weighed against the alternative (an ORT AAR on the app's classpath):

| | ship the `.so` from this release (chosen) | ORT AAR as a Gradle dependency |
|---|---|---|
| Reproducibility | every byte the app builds against is in `paddle-manifest.json` with a SHA-256 and is **verified on fetch**; the release is frozen even if Maven Central changes | the effective runtime is whatever Gradle resolves; the AAR's `.so` is outside this repository's manifest and its hash is not pinned by us |
| Simplicity | no new Gradle dependency, no version skew between the AAR's Java API (unused) and the C API the shim speaks | one dependency line, but drags in an unused Java/Kotlin wrapper and its transitive metadata |
| Consistency | identical to how the previous Paddle-Lite artifact worked (and how `:core:dictionary`'s libs work) | new mechanism, second source of native libs |
| Cost | the AAR is unpacked by this build and republished byte-for-byte; APK size is the same either way | — |

Being able to state "the app runs ORT 1.31.0 whose `arm64-v8a` `libonnxruntime.so` is
`8005a741…`, re-verified on every fetch" is worth more than saving one dependency line, and it keeps
the entire native surface of an opt-in feature inside one auditable manifest.

### Runtime dependencies of the shim

* **`libonnxruntime.so`** — not *linked*, **loaded at runtime** (`dlopen` + `OrtGetApiBase()`, see
  `src/ort_runtime.h`). ONNX Runtime's C API is ABI-stable and explicitly designed for this, so the
  shim has no `DT_NEEDED` on ORT: when the runtime is missing, `dlopen` fails, `Engine::Init()`
  returns `false`, and OCR degrades gracefully exactly as before. Search order: `$KOMIKUN_ORT_LIB`
  → `libonnxruntime.so` next to the shim (the `filesDir/…/lib/<abi>/` layout) → the bare soname (the
  APK `jniLibs` layout).
* **`libc++_shared.so`** — the shim is built with `-DANDROID_STL=c++_shared`. ORT's own Android
  runtime is *not* (it links `libc/libdl/liblog/libandroid/libm` only, with libc++ statically
  linked), so ORT adds no STL dependency of its own. As before, the app's AGP already packages the
  NDK copy for `:core:dictionary`, which is why the **app** manifest omits this entry (a second copy
  fails the app merge); the release still publishes it for completeness.
* **`libomp.so`** — still required, but no longer by ORT: opencv-mobile's `parallel_for_` calls
  `omp_get_thread_num` / `omp_get_max_threads`. ORT's own threading is its intra-op pool
  (`SetIntraOpNumThreads`), not OpenMP.

## What `libpaddle_ocr.so` implements

The exact JNI surface declared by
`core/ocr/src/main/java/mihon/core/ocr/paddle/PaddleOcrEngine.kt` — **unchanged** from v1/v2:

| Symbol | Behaviour |
|---|---|
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeInit(detParam, detBin, recParam, recBin, dict, threads)` | Loads the detection + recognition ONNX graphs and the dictionary. `detBin`/`recBin` are accepted for interface compatibility and used only as a fallback path. |
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeProcess(bitmap, thresh, boxThresh, unclipRatio, filterFurigana, threads)` | Runs DB detection + CRNN recognition over an `ARGB_8888` `Bitmap` and returns the `TIMING|`/`LINE|` protocol string. |
| `Java_mihon_core_ocr_paddle_PaddleOcrEngine_nativeRelease()` | Frees the sessions. |

Output protocol (parsed by `PaddleOcrEngine.parseOutput`) — also byte-for-byte unchanged:

```
TIMING|<total>|<prep>|<det>|<post>|<crop>|<rec>|<rawBoxCount>
LINE|<order>|<score>|<L>,<T>,<R>,<B>|<tlx>,<tly>,<trx>,<try>,<brx>,<bry>,<blx>,<bly>|<isVertical>|<isFurigana>|<text>
```

All coordinates are normalised to `[0,1]` over the input image; the quad is `tl,tr,br,bl` and drives
`PaddleOcrEngine.quadRotation`. `isFurigana` is always `0` — ruby filtering is the merger's job.
Because the JNI names, parameter order, return types and the protocol text are identical, **the
Kotlin layer needs no logic change**: only the model file names and the release URL move.

## Pipeline

```
page (BGR)
  → DetResize: longest side ≤ 960, both sides rounded to a multiple of 32
  → NormalizeImage: /255, then (x - mean)/std, mean [0.485,0.456,0.406], std [0.229,0.224,0.225]
  → PP-OCRv6-small det → probability map → threshold/dilate → DB box extraction (PaddleOCR
    db_post_process, clipped by Clipper) → FilterTagDetRes
  → reading order (manga: right-to-left columns for vertical text)
  → per-box perspective crop (GetRotateCropImage, which transposes tall crops 90° CCW so vertical
    Japanese columns are read horizontally — PaddleOCR's own convention)
  → per-crop resize to rec_image_height 48, width ceil(48·w/h) capped at 3200
  → NormalizeImage: (x/255 - 0.5) / 0.5
  → PP-OCRv6-small rec → greedy CTC decode (dict index 0 = blank, trailing space class)
```

Preprocessing is byte-for-byte the validated benchmark's (`/tmp/opencode/ocr-bench/common.py`) — see
[Validation](#validation). One deliberate difference from `ocr-models-v1`/`v2`:

* **No shared recognition width.** Paddle-Lite's mobile predictor returned wrong results when a
  `Run()` used a different input shape than the previous one, so the v1/v2 engine forced one
  recognition width per page and rebuilt predictors on shape changes. ONNX Runtime handles dynamic
  shapes correctly, so every crop is now recognised at its own natural width — which is also exactly
  what the benchmark measured. The `rec_image_height` decoupling/session-rebuild machinery is gone.

## Model-agnostic by construction

The ONNX graphs are architecture-independent, so **one model set serves all three ABIs** — the
per-ABI `valid_targets` problem that forced v1/v2 to be ARM-only does not exist here.

## Models

`ocr-models-v3` ships the official **PP-OCRv6 small** ONNX exports from the `PaddlePaddle` HF org,
pinned to a revision in `deps.lock.json`:

| File | Source | Notes |
|---|---|---|
| `ppocrv6_small_det.onnx` | `PaddlePaddle/PP-OCRv6_small_det_onnx@28fe5895` | PP-OCRv6 small DB text detector. `sha256 d73e0058…`, 9.4 MiB. |
| `ppocrv6_small_rec.onnx` | `PaddlePaddle/PP-OCRv6_small_rec_onnx@b8f84f0b` | PP-OCRv6 small recogniser, `rec_image_height = 48`, 18 710 output classes. `sha256 5435fd74…`, 20.2 MiB. |
| `ppocrv6_small_dict.txt` | `PostProcess.character_dict` of the same `inference.yml`, extracted by `tools/extract_dict.py` | 18 708 entries (**180 kana**, 15 565 han). The engine prepends `#` (CTC blank) and appends a space class. `sha256 b5f2bfe2…`. |

**Why `small` and not `tiny`:** PP-OCRv6 tiny's recogniser has no kana at all, so it cannot read
Japanese (the benchmark scored it at 3.6 % exact match / 78 % CER on the manga gold set). Small is
the cheapest v6 tier that does (86 hiragana + 94 katakana).

## Build

```bash
./build.sh                 # Android .so for every supported ABI + models + manifests
./build.sh --host          # also build the off-device conformance harness
./build.sh --abi arm64-v8a
./build.sh --manifest-only # regenerate the manifests from an existing out/
./build.sh --clean
```

Requirements: `cmake >= 3.22`, `ninja`/`make`, `curl`, `unzip`, `python3` (+ `pyyaml`), and the
**Android NDK `29.0.14206865`** (`ANDROID_NDK_HOME`, or `$ANDROID_SDK_ROOT/ndk/...`). Everything else
is downloaded and pinned by `deps.lock.json`.

`build.sh` writes:

* `out/lib/<abi>/{libpaddle_ocr.so,libonnxruntime.so,libomp.so,libc++_shared.so}`
* `out/models/{ppocrv6_small_det.onnx,ppocrv6_small_rec.onnx,ppocrv6_small_dict.txt}`
* `out/manifest.json` — the release manifest, consumed at runtime by `ModelDownloader`
* `out/app-paddle-manifest.json` — the copy that belongs at `core/ocr/paddle-manifest.json`
* `out/paddle_ocr_host` — the host conformance harness (`--host`)

## Validation

Validation was done on the host with the **shipped engine sources**, built for Linux x86_64 against
the **official Linux ONNX Runtime 1.31.0** build and the **same ONNX model bytes** the release
publishes. This is possible precisely because ORT is portable and the graphs are
architecture-independent — the one thing the Paddle-Lite artifact could never do (its `.nb` models
were `valid_targets=arm` and aborted on x86_64).

`tools/parity_check.py` drives `paddle_ocr_host` over the PP-OCRv6 benchmark's real manga crops and
pages (`/tmp/opencode/ocr-bench`, 5 Japanese titles, 30 pages, 752 saved crops per condition) and
diffs against the recorded PP-OCRv6-small results:

```bash
./build.sh --host
python3 tools/parity_check.py --host out/paddle_ocr_host --models out/models \
    --bench /tmp/opencode/ocr-bench --cond orig   --mode crops
python3 tools/parity_check.py --host out/paddle_ocr_host --models out/models \
    --bench /tmp/opencode/ocr-bench --cond orig   --mode detect
python3 tools/parity_check.py --host out/paddle_ocr_host --models out/models \
    --bench /tmp/opencode/ocr-bench --cond orig   --mode pages
```

| Check | Result |
|---|---|
| **Recognition parity, `orig` crops** (`paddle_ocr_host parity`, exactly the same crop images the benchmark scored) | **752 / 752 exact byte match (100 %)**, CER 0.00 % |
| Recognition parity, `ai` (AI-upscaled) crops | 752 / 752 exact byte match (100 %), CER 0.00 % |
| Recognition parity, `classical` (Lanczos) crops | 752 / 752 exact byte match (100 %), CER 0.00 % |
| **Detection parity** vs the benchmark's own PP-OCRv6-small detector (`det_stats`) | native 538 boxes vs reference 542; **23 / 30 pages identical**, max abs diff 2, mean abs diff 0.33 |
| End-to-end on 30 pages (`paddle_ocr_host full`) | 530 lines, sane Japanese (e.g. `先輩たちに`, `何が始まるんだろう`, `一年組の十一月`) |
| End-to-end latency (Linux x86_64, 4 threads, 30 pages) | det ≈ 58–70 ms, rec ≈ 8–151 ms, ~77–234 ms per page |

The 752-crop set is the benchmark's **v5** detector crop set; the IoU-matched text agreement against
it (81.4 % exact, 5.7 % CER) therefore measures the v5 → v6 detector change, not an engine
difference — which is why the native-vs-reference-v6-small box comparison above is the meaningful
detection parity number.

**Not validated here:** the Android `.so`s were cross-compiled and verified to exist, be
`ELF … ARM aarch64`/`ARM`/`x86-64`, and match `sha256` — but they were **not executed on an Android
device or emulator** (none is available). The engine logic and the model bytes *are* host-validated
above; only the NDK toolchain output and the JNI glue are device-untested.

## Reproducibility

The build is fully scripted and every input is pinned in `deps.lock.json`. `libonnxruntime.so` and
the three model files are republished **byte-for-byte** (they are upstream builds/downloads, verified
by SHA-256 at fetch time), so those are deterministic. Our own `libpaddle_ocr.so` and the NDK
`libomp.so`/`libc++_shared.so` are **not** byte-identical across runs (NDK artifacts carry build
ids). The manifest pins the bytes that were actually published, and both consumers — the Gradle
`fetchPaddleNativeLibs` task and the app's `ModelDownloader` — verify SHA-256 on every fetch. After
republishing, regenerate both manifests with `./build.sh --manifest-only` and update the app's
`core/ocr/paddle-manifest.json`.

## Licensing

* This repository's own code (`src/`, `tools/`, `build.sh`, `CMakeLists.txt`): **Apache-2.0**.
* Vendored `third_party/paddleocr_lite/`: PaddleOCR `deploy/lite`, Apache-2.0.
* `third_party/clipper/`: Clipper 6.4.2, Boost Software License 1.0.
* ONNX Runtime 1.31.0 (republished `libonnxruntime.so`): **MIT**.
* PP-OCRv6 small models + dictionary: Apache-2.0.
* opencv-mobile/OpenCV: Apache-2.0.
* Android NDK runtime libraries: Apache-2.0 WITH LLVM-exception.

See `LICENSE` and `NOTICE`.
