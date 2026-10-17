# Drop-in integration for `komi-kun`'s `:core:ocr`

Everything below is a change in the **app repository**, not here. `ocr-models-v3` swaps the
inference backend (Paddle-Lite → **ONNX Runtime**) and the models (PP-OCRv5 mobile → **PP-OCRv6
small**), and adds `x86_64` — but the **JNI signatures and the `TIMING|`/`LINE|` protocol are
byte-for-byte unchanged**, so the Kotlin *logic* needs no change: only the model file names, the
release URL, one manifest file, and some comments.

Published release:
`https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v3`
Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v3`

---

## 1. `core/ocr/paddle-manifest.json` — replace the whole file with

(also published as `app-paddle-manifest.json` on the release)

```json
{
  "_comment": [
    "Consumed by :core:ocr:fetchPaddleNativeLibs. sha256 must match the file published at",
    "<baseUrl>/<path>. Regenerate with build.sh --manifest-only if the release is republished.",
    "libc++_shared.so is deliberately NOT listed: AGP already packages the NDK copy for any",
    "module built with -DANDROID_STL=c++_shared (:core:dictionary), and a second copy in",
    "jniLibs fails the app merge with a duplicate-path error. Re-add the release's",
    "libc++_shared.so entries only if no other module provides it."
  ],
  "release": "ocr-models-v3",
  "baseUrl": "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v3",
  "abi": ["arm64-v8a", "armeabi-v7a", "x86_64"],
  "files": [
    { "abi": "arm64-v8a", "name": "libpaddle_ocr.so", "path": "arm64-v8a_libpaddle_ocr.so", "sha256": "164d82b944b6b0c45c9a18fa963290d6ae5ee4726410ae09c6a8a9245c8bee20", "size": 2950104, "spdx": "Apache-2.0" },
    { "abi": "arm64-v8a", "name": "libonnxruntime.so", "path": "arm64-v8a_libonnxruntime.so", "sha256": "8005a741e57117b4d7b476417706c0e9af13cf1b89db47e60f0cbc56e1af91e0", "size": 34208680, "spdx": "MIT" },
    { "abi": "arm64-v8a", "name": "libomp.so", "path": "arm64-v8a_libomp.so", "sha256": "b0b507bb9e89360304272ba335d5fb2f6661f88842c2b3c749474cb7b8acda76", "size": 1234864, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "abi": "armeabi-v7a", "name": "libpaddle_ocr.so", "path": "armeabi-v7a_libpaddle_ocr.so", "sha256": "b3820f02194cd681d1ca8a8762ff1b243f7bc804299d5b8dd69402d9a932851c", "size": 1316384, "spdx": "Apache-2.0" },
    { "abi": "armeabi-v7a", "name": "libonnxruntime.so", "path": "armeabi-v7a_libonnxruntime.so", "sha256": "2a63f9a56b211511209dbf4b5bc143f18d0b365977dd7a0afd57d95c85479af6", "size": 24182012, "spdx": "MIT" },
    { "abi": "armeabi-v7a", "name": "libomp.so", "path": "armeabi-v7a_libomp.so", "sha256": "c533187575eedfa92b56f8b462ba66bb8ab809b03a04d951d07bff7720ef9c5f", "size": 994812, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "abi": "x86_64", "name": "libpaddle_ocr.so", "path": "x86_64_libpaddle_ocr.so", "sha256": "23befb4c78f142bef52f071abe3553810b456284f5e9bf866f9e997f95303634", "size": 3730648, "spdx": "Apache-2.0" },
    { "abi": "x86_64", "name": "libonnxruntime.so", "path": "x86_64_libonnxruntime.so", "sha256": "068896846a7fe679cf780c1c532745286d9ef2e469f3dbde5bb9cf2c02095b18", "size": 40827944, "spdx": "MIT" },
    { "abi": "x86_64", "name": "libomp.so", "path": "x86_64_libomp.so", "sha256": "dbd0e8727752ff227ebb0c2a4dcb23a76a9c66d90bb86d22c30f47bb719022bf", "size": 1164912, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "name": "ppocrv6_small_det.onnx", "path": "ppocrv6_small_det.onnx", "sha256": "d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e", "size": 9880512, "spdx": "Apache-2.0" },
    { "name": "ppocrv6_small_rec.onnx", "path": "ppocrv6_small_rec.onnx", "sha256": "5435fd747c9e0efe15a96d0b378d5bd157e9492ed8fd80edf08f30d02fa24634", "size": 21159378, "spdx": "Apache-2.0" },
    { "name": "ppocrv6_small_dict.txt", "path": "ppocrv6_small_dict.txt", "sha256": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d", "size": 74947, "spdx": "Apache-2.0" }
  ],
  "models": {
    "det": ["ppocrv6_small_det.onnx"],
    "rec": ["ppocrv6_small_rec.onnx"],
    "dict": ["ppocrv6_small_dict.txt"]
  }
}
```

`fetchPaddleNativeLibs` needs **no change**: it already honours `entry["path"]`, verifies
non-empty `sha256` values, and hard-fails on mismatch. It fetches every entry with an `abi` (now
`libpaddle_ocr.so`, **`libonnxruntime.so`** and `libomp.so`, for three ABIs) into
`src/main/jniLibs/<abi>/`:

```bash
./gradlew :core:ocr:fetchPaddleNativeLibs
```

Then verify the packaged APK contains, per ABI, `lib/<abi>/libpaddle_ocr.so`,
`lib/<abi>/libonnxruntime.so`, `lib/<abi>/libomp.so` (and `libc++_shared.so` from the NDK side).

## 2. `core/ocr/src/main/java/mihon/core/ocr/paddle/PaddleOcrEngine.kt`

Model file names (the ONNX graphs replace the Paddle-Lite `.nb` files):

```diff
     companion object {
         private const val MODEL_DIR = "paddle_ocr"
-        // PP-OCRv5 (ocr-models-v2): unified multi-language recogniser. Still CTC with
-        // rec_image_height = 48, so the JNI input shape and the dict convention are unchanged.
-        private const val DET_MODEL = "ppocrv5_det.nb"
-        private const val REC_MODEL = "ppocrv5_rec.nb"
-        private const val DICT_TXT = "ppocrv5_dict.txt"
+        // PP-OCRv6 small (ocr-models-v3), run through ONNX Runtime. Still CTC with
+        // rec_image_height = 48, so the JNI input shape and the dict convention are unchanged;
+        // the `.nb` (Paddle-Lite) files are replaced by `.onnx`.
+        private const val DET_MODEL = "ppocrv6_small_det.onnx"
+        private const val REC_MODEL = "ppocrv6_small_rec.onnx"
+        private const val DICT_TXT = "ppocrv6_small_dict.txt"
```

`x86_64` is now shipped, so `SUPPORTED_ABIS` and its comment change from "attempt/degrade" to
"supported":

```diff
-        /**
-         * ABIs the Kotlin engine will *attempt*. Only `arm64-v8a` / `armeabi-v7a` are shipped by
-         * `komi-kun-ocr-models`; `x86_64` is kept here for graceful degradation (see [supportedAbi]).
-         */
+        /**
+         * ABIs the Kotlin engine will *attempt*. `komi-kun-ocr-models` ships all three
+         * (`ocr-models-v3`, ONNX Runtime + architecture-independent ONNX models); an ABI that is
+         * somehow absent still degrades gracefully (see [supportedAbi]).
+         */
         private val SUPPORTED_ABIS = listOf("arm64-v8a", "armeabi-v7a", "x86_64")
```

```diff
     /**
      * First supported ABI of the device, or null.
      *
-     * CAVEAT (x86_64): Paddle-Lite publishes **no Android x86_64 prebuilt** and its `opt`-converted
-     * models are target-specific, so `komi-kun-ocr-models` ships only `arm64-v8a`/`armeabi-v7a`.
-     * `x86_64` is still accepted here on purpose: on an x86_64 emulator `System.loadLibrary`
-     * throws `UnsatisfiedLinkError`, [loadNativeLibraries] catches it, and `init()` returns false —
-     * the intended graceful degradation (OCR simply unavailable), not a crash. Do **not** add an
-     * `x86_64` manifest entry or build an ARM `.so` under an x86_64 name.
+     * `ocr-models-v3` ships `arm64-v8a`, `armeabi-v7a` **and `x86_64`**: ONNX Runtime publishes all
+     * three, and the ONNX models are architecture-independent, so x86_64 emulators are supported
+     * rather than merely degrading. The `UnsatisfiedLinkError` catch below is kept as a safety net
+     * for a device whose ABI has no published artifact.
      */
```

The `INTEGRATION POINT` header comment (top of the file) should mention the runtime library as well:

```diff
-// INTEGRATION POINT — the native `libpaddle_ocr.so` is a SEPARATE artifact (contract §1.7).
+// INTEGRATION POINT — `libpaddle_ocr.so` *and* `libonnxruntime.so` are SEPARATE artifacts
+// (contract §1.7).
 // This class declares the JNI surface (`nativeInit`/`nativeProcess`/`nativeRelease`) and the Kotlin
-// overload that consumes it. The `.so` is our own JNI wrapper over Paddle-Lite (Apache-2.0),
-// published to `komi-kun/komi-kun-ocr-models` alongside `libomp.so`/`libc++_shared.so` and fetched
+// overload that consumes it. The `.so` is our own JNI wrapper over ONNX Runtime (MIT, loaded with
+// `dlopen`), published to `komi-kun/komi-kun-ocr-models` alongside `libonnxruntime.so`,
+// `libomp.so`/`libc++_shared.so` and fetched
 // by `:core:ocr:fetchPaddleNativeLibs` from `core/ocr/paddle-manifest.json`. Nothing in this module
-// builds Paddle-Lite. Until the artifact lands, `init()` simply returns false and `recognize()`
+// builds ONNX Runtime. Until the artifact lands, `init()` simply returns false and `recognize()`
 // yields an empty list — the engine degrades gracefully.
```

**Optional but recommended** — `loadNativeLibraries`'s app-storage branch loads its libraries in
dependency order; add the runtime there too (currently unreachable, because nothing writes the
`.so`s into `filesDir`, but it would silently break if a runtime download path is reintroduced):

```diff
                 System.load(File(libDir, "libc++_shared.so").absolutePath)
                 System.load(File(libDir, "libomp.so").absolutePath)
+                System.load(File(libDir, "libonnxruntime.so").absolutePath)
                 System.load(engine.absolutePath)
```

## 3. `core/ocr/src/main/java/mihon/core/ocr/paddle/ModelDownloader.kt`

```diff
 private const val DEFAULT_BASE_URL =
-    "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2"
+    "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v3"
```

```diff
- * Downloads and verifies the PaddleOCR det/rec models and `ppocrv5_dict.txt` into
+ * Downloads and verifies the PaddleOCR det/rec models and `ppocrv6_small_dict.txt` into
```

```diff
         // PP-OCRv5 (ocr-models-v2): unified multi-language recogniser.
-        private const val DET_MODEL = "ppocrv5_det.nb"
-        private const val REC_MODEL = "ppocrv5_rec.nb"
-        private const val DICT_TXT = "ppocrv5_dict.txt"
+        // PP-OCRv6 small (ocr-models-v3): ONNX graphs (the shipped backend is ONNX Runtime).
+        private const val DET_MODEL = "ppocrv6_small_det.onnx"
+        private const val REC_MODEL = "ppocrv6_small_rec.onnx"
+        private const val DICT_TXT = "ppocrv6_small_dict.txt"
```

```diff
-        /**
-         * Model files shipped by `ocr-models-v1` (PP-OCRv4) that the engine no longer loads. Listed
-         * by name so [cleanupStaleModels] can prune them on upgrade without ever touching a file
-         * this app did not publish.
-         */
+        /**
+         * Model files shipped by `ocr-models-v1` (PP-OCRv4) and `ocr-models-v2` (PP-OCRv5) that the
+         * engine no longer loads. Listed by name so [cleanupStaleModels] can prune them on upgrade
+         * without ever touching a file this app did not publish.
+         */
         private val LEGACY_MODELS = listOf(
             "ppocrv4_det.nb",
             "ppocrv4_japan_rec.nb",
             "ppocrv4_japan_dict.txt",
+            "ppocrv5_det.nb",
+            "ppocrv5_rec.nb",
+            "ppocrv5_dict.txt",
         )
```

## 4. Docs to update (no behaviour change)

* `core/ocr/README.md` — the "PaddleOCR artifact (integration point)" section still says
  `ocr-models-v2`, "our own JNI over Paddle-Lite", `libomp.so` "is required because only this
  artifact uses OpenMP", "ABIs shipped: arm64-v8a, armeabi-v7a" and "**`x86_64` is not supported**".
  Rewrite it for `ocr-models-v3` / ONNX Runtime / three ABIs (`libomp.so` is still required, but by
  opencv-mobile's `parallel_for_`, not by ONNX Runtime).
* `core/ocr/build.gradle.kts` — the comment above `fetchPaddleNativeLibs` says "our own JNI over
  Paddle-Lite (Apache-2.0) … together with `libomp.so`". Update to "our own JNI over ONNX Runtime
  (MIT) … together with `libonnxruntime.so` and `libomp.so`".

## 5. `app/build.gradle.kts` — nothing to do

* `splits.abi.include("armeabi-v7a", "arm64-v8a", "x86_64")` already matches the published ABIs.
* `keepDebugSymbols += listOf("libhoshidicts_jni", "libpaddle_ocr")` needs no change:
  `libpaddle_ocr.so` is stripped by this artifact before publishing, and `libonnxruntime.so` is
  already stripped upstream. (Adding `"libonnxruntime"` there would be harmless.)
* No Gradle dependency is added: ONNX Runtime is provisioned as a checksum-verified `.so` from this
  release, exactly like the old Paddle-Lite libraries — see the README's provisioning decision.

---

## Caveats to record in the app

* **`x86_64` is now supported.** `supportedAbi()` already returned `x86_64`; it now resolves to a
  real artifact, so OCR works on x86_64 emulators instead of silently failing. Nothing else changes.
* **APK size.** `libonnxruntime.so` is ~34 MB (`arm64-v8a`), ~24 MB (`armeabi-v7a`), ~41 MB
  (`x86_64`) uncompressed. With ABI splits each per-ABI APK grows by its own copy; the **universal**
  APK carries all three. That is the cost of the ONNX Runtime backend. If that is unacceptable, the
  alternative (not taken here) is to move `libonnxruntime.so` from build-time `jniLibs` to a
  runtime download like the models — which reintroduces a download step in
  `PaddleOcrEngine.loadNativeLibraries` and loses the "shipped in the APK" property.
* **`libc++_shared.so`** is still intentionally absent from the app manifest (see the `_comment`):
  AGP already packages the NDK copy for `-DANDROID_STL=c++_shared` modules (`:core:dictionary`),
  and a second copy in `jniLibs` fails the app merge.
* **`libomp.so` is still required** — not by ONNX Runtime (which owns its own thread pool) but by
  opencv-mobile's `parallel_for_` (`omp_get_thread_num` / `omp_get_max_threads`), which is
  statically linked into `libpaddle_ocr.so`.
* **`nativeInit`'s `detBin`/`recBin` stay empty.** The engine loads one ONNX graph per stage; the
  two "bin" arguments remain accepted for JNI-signature compatibility and are only a fallback path.
* **Model size grows slightly.** `rec` is 20.2 MB (`ppocrv6_small_rec.onnx`) vs 16.7 MB for the
  PP-OCRv5 recogniser, plus a new 9.4 MB detector — a one-time opt-in runtime download, not packaged
  in the APK. `cleanupStaleModels()` prunes the superseded v5/v4 files on upgrade.
* **Reproducibility.** The models and `libonnxruntime.so` are the fixed upstream bytes (pinned by
  SHA-256); our `libpaddle_ocr.so` is rebuilt from source and is not byte-identical across runs.
  Both consumers verify SHA-256 on every fetch, so republishing requires regenerating both
  manifests (`./build.sh --manifest-only`) and updating the app's `paddle-manifest.json`.
