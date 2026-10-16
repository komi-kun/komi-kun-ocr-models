# Drop-in integration for `komi-kun`'s `:core:ocr`

Everything below is a change in the **app repository**, not here. Four small patches; the
`paddle-manifest.json`, the `fetchPaddleNativeLibs` task and `keepDebugSymbols` were already
scaffolded by W1, so only the URL shape and the model file names change.

Published release:
`https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v1`
Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v1`

---

## 1. `core/ocr/paddle-manifest.json` — replace with

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
  "release": "ocr-models-v1",
  "baseUrl": "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v1",
  "abi": ["arm64-v8a", "armeabi-v7a"],
  "files": [
    { "abi": "arm64-v8a", "name": "libpaddle_ocr.so", "path": "arm64-v8a_libpaddle_ocr.so", "sha256": "5dbcd1a20cabd957fe7d5a6ae80df11e4f16b0e76341df7466c1be33fce465f6", "size": 3130624, "spdx": "Apache-2.0" },
    { "abi": "arm64-v8a", "name": "libomp.so", "path": "arm64-v8a_libomp.so", "sha256": "b0b507bb9e89360304272ba335d5fb2f6661f88842c2b3c749474cb7b8acda76", "size": 1234864, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "abi": "armeabi-v7a", "name": "libpaddle_ocr.so", "path": "armeabi-v7a_libpaddle_ocr.so", "sha256": "0cabcfe1f02603dc13f6ed8f50752b946ef704de384432a0dd24d8a14f0b0e89", "size": 1445072, "spdx": "Apache-2.0" },
    { "abi": "armeabi-v7a", "name": "libomp.so", "path": "armeabi-v7a_libomp.so", "sha256": "c533187575eedfa92b56f8b462ba66bb8ab809b03a04d951d07bff7720ef9c5f", "size": 994812, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "name": "ppocrv4_det.nb", "path": "ppocrv4_det.nb", "sha256": "237749af5d5a699b5cc4b3662be9c4dfbccbac2d15ec66573ae318cfc10b6ade", "size": 4940078, "spdx": "Apache-2.0" },
    { "name": "ppocrv4_japan_rec.nb", "path": "ppocrv4_japan_rec.nb", "sha256": "f6e7de1a3f2a51298e0b44a221763127668ec8c01dd4ea4294bba52167412c61", "size": 9915998, "spdx": "Apache-2.0" },
    { "name": "ppocrv4_japan_dict.txt", "path": "ppocrv4_japan_dict.txt", "sha256": "1dcfcb41eec90576a945b3084f22ade11ced506e24f14879245b071698f308e8", "size": 17332, "spdx": "Apache-2.0" }
  ],
  "models": {
    "det": ["ppocrv4_det.nb"],
    "rec": ["ppocrv4_japan_rec.nb"],
    "dict": ["ppocrv4_japan_dict.txt"]
  }
}
```

`fetchPaddleNativeLibs` already skips empty `sha256` values, verifies non-empty ones and hard-fails
on mismatch, so this file alone is safe to land before the release exists (it is fetched by name, not
by ABI folder).

## 2. `core/ocr/build.gradle.kts` — make the fetch task honour `path`

GitHub release assets are a flat namespace (a `/` in an asset name is rewritten to `.`), so the URL
can no longer be assembled as `"$baseUrl/$abi/$name"`. One line, in `fetchPaddleNativeLibs`:

```diff
-            val url = "$baseUrl/$abi/$name"
+            // Release assets are flat; `path` is the published asset name (see paddle-manifest.json).
+            val assetPath = (entry["path"] as? String)?.takeIf { it.isNotBlank() } ?: "$abi/$name"
+            val url = "$baseUrl/$assetPath"
```

Run it once to drop the libraries into `core/ocr/src/main/jniLibs/<abi>/` (git-ignored):

```bash
./gradlew :core:ocr:fetchPaddleNativeLibs
```

Then verify the packaged APK contains `lib/arm64-v8a/libpaddle_ocr.so` and
`lib/arm64-v8a/libomp.so` (and the same for `armeabi-v7a`).

## 3. `PaddleOcrEngine.kt` — model file names

The engine now loads a single Paddle-Lite `naive_buffer` model per stage (the `.param`/`.bin` pair is
a two-file format Paddle-Lite does not use for its supported mobile models). Replace the constants
and the `nativeInit` call; keep the JNI signature exactly as-is.

```diff
     companion object {
         private const val MODEL_DIR = "paddle_ocr"
-        private const val DET_PARAM = "manga_det_fp16.param"
-        private const val DET_BIN = "manga_det_fp16.bin"
-        private const val REC_PARAM = "manga_rec_fp16.param"
-        private const val REC_BIN = "manga_rec_fp16.bin"
-        private const val DICT_TXT = "ppocrv6_dict.txt"
+        private const val DET_MODEL = "ppocrv4_det.nb"
+        private const val REC_MODEL = "ppocrv4_japan_rec.nb"
+        private const val DICT_TXT = "ppocrv4_japan_dict.txt"
```

```diff
         val success = nativeInit(
-            detParamAbsPath = File(modelDir, DET_PARAM).absolutePath,
-            detBinAbsPath = File(modelDir, DET_BIN).absolutePath,
-            recParamAbsPath = File(modelDir, REC_PARAM).absolutePath,
-            recBinAbsPath = File(modelDir, REC_BIN).absolutePath,
+            detParamAbsPath = File(modelDir, DET_MODEL).absolutePath,
+            detBinAbsPath = "",
+            recParamAbsPath = File(modelDir, REC_MODEL).absolutePath,
+            recBinAbsPath = "",
             dictAbsPath = File(modelDir, DICT_TXT).absolutePath,
             numThreads = options.numThreads,
         )
```

`nativeInit` ignores the empty `detBin`/`recBin` arguments (they exist for interface compatibility
and are only used as a fallback path).

## 4. `ModelDownloader.kt` — required models

```diff
     companion object {
         private const val MANIFEST_NAME = "manifest.json"
         private const val PADDLE_DIR = "paddle_ocr"

-        private const val DET_PARAM = "manga_det_fp16.param"
-        private const val DET_BIN = "manga_det_fp16.bin"
-        private const val REC_PARAM = "manga_rec_fp16.param"
-        private const val REC_BIN = "manga_rec_fp16.bin"
-        private const val DICT_TXT = "ppocrv6_dict.txt"
+        private const val DET_MODEL = "ppocrv4_det.nb"
+        private const val REC_MODEL = "ppocrv4_japan_rec.nb"
+        private const val DICT_TXT = "ppocrv4_japan_dict.txt"

         /** Must stay in sync with [PaddleOcrEngine]'s model constants. */
-        val REQUIRED_MODELS = listOf(DET_PARAM, DET_BIN, REC_PARAM, REC_BIN, DICT_TXT)
+        val REQUIRED_MODELS = listOf(DET_MODEL, REC_MODEL, DICT_TXT)
     }
```

`DEFAULT_BASE_URL` in `ModelDownloader` already equals the release base URL, and
`downloadAndVerify` already uses `entry.path`, so runtime model download needs no other change.
If `importFromUri` is used, the user-supplied zip must contain the three files under a `paddle_ocr/`
prefix (that is where `verifyInstalled` looks).

## 5. `app/build.gradle.kts` — nothing to do

`keepDebugSymbols += listOf("libhoshidicts_jni", "libpaddle_ocr")` (with `.map { "**/$it.so" }`) is
already present, and `splits.abi.include("armeabi-v7a", "arm64-v8a", "x86_64")` stays as-is.

---

## Caveats to record in the app

* **`x86_64` is not published.** Paddle-Lite ships no Android x86_64 prebuilt and `opt`-converted
  models are target-specific, so an ARM `.so`/model pair on x86_64 would abort, not degrade.
  `PaddleOcrEngine.supportedAbi()` may keep accepting `x86_64`: `System.loadLibrary("paddle_ocr")`
  then throws `UnsatisfiedLinkError`, `loadNativeLibraries` catches it and `init()` returns `false`,
  which is the intended graceful degradation. Do not add an `x86_64` entry to
  `paddle-manifest.json` or `splits.abi` until an x86_64 Paddle-Lite build + x86 model set exist.
* **`libc++_shared.so`** is intentionally absent from the manifest (see the `_comment`): AGP already
  packages the NDK copy for `-DANDROID_STL=c++_shared` modules. Shipping a second copy through
  `jniLibs` fails the app merge.
* **`libomp.so`** *is* required, because only this artifact uses OpenMP.
* **Reproducibility.** The build is scripted and pinned (`deps.lock.json`), but the outputs are not
  byte-identical across runs: Paddle-Lite's `opt` writes a non-deterministic naive-buffer header, and
  NDK artifacts carry build ids. The manifest therefore pins the *published* bytes and both consumers
  (the Gradle fetch task and `ModelDownloader`) verify SHA-256 on every fetch. Republishing the
  release requires regenerating both manifests (`./build.sh --manifest-only`) and updating the app's
  `paddle-manifest.json`.
