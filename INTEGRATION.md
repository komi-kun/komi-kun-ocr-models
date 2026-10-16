# Drop-in integration for `komi-kun`'s `:core:ocr`

Everything below is a change in the **app repository**, not here. The `paddle-manifest.json`, the
`fetchPaddleNativeLibs` task and `keepDebugSymbols` were already scaffolded for `ocr-models-v1`
(and the build task already honours `path`), so moving to **`ocr-models-v2` / PP-OCRv5** is only:

1. replace `core/ocr/paddle-manifest.json`,
2. rename the three model constants in `PaddleOcrEngine.kt` and `ModelDownloader.kt`, and
3. bump `ModelDownloader.DEFAULT_BASE_URL`.

There are **no engine, JNI-signature, input-shape or dictionary-format changes**: PP-OCRv5
recognition is still CTC at `rec_image_height = 48` (`C = 18385`), DB detection is unchanged, and the
engine's `['#' (blank)] + dict + [' ']` convention already matches PP-OCRv5's `ppocr_keys_ocrv5.txt`.

Published release:
`https://github.com/komi-kun/komi-kun-ocr-models/releases/tag/ocr-models-v2`
Base URL: `https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2`

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
  "release": "ocr-models-v2",
  "baseUrl": "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2",
  "abi": ["arm64-v8a", "armeabi-v7a"],
  "files": [
    { "abi": "arm64-v8a", "name": "libpaddle_ocr.so", "path": "arm64-v8a_libpaddle_ocr.so", "sha256": "5882054ecd84625ae967b69ab0ffa94b66c78105a08672c49bef4598976f66f8", "size": 3130624, "spdx": "Apache-2.0" },
    { "abi": "arm64-v8a", "name": "libomp.so", "path": "arm64-v8a_libomp.so", "sha256": "b0b507bb9e89360304272ba335d5fb2f6661f88842c2b3c749474cb7b8acda76", "size": 1234864, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "abi": "armeabi-v7a", "name": "libpaddle_ocr.so", "path": "armeabi-v7a_libpaddle_ocr.so", "sha256": "4fc90597ef3da08ff1cf247b6f47ea388a134403605ddf99f4c9ba3974710781", "size": 1445072, "spdx": "Apache-2.0" },
    { "abi": "armeabi-v7a", "name": "libomp.so", "path": "armeabi-v7a_libomp.so", "sha256": "c533187575eedfa92b56f8b462ba66bb8ab809b03a04d951d07bff7720ef9c5f", "size": 994812, "spdx": "Apache-2.0 WITH LLVM-exception" },
    { "name": "ppocrv5_det.nb", "path": "ppocrv5_det.nb", "sha256": "bee84fdb3d7c5f312a797da81f2dd44c89088c08e1f58a51044a823cddfbd90c", "size": 5001214, "spdx": "Apache-2.0" },
    { "name": "ppocrv5_rec.nb", "path": "ppocrv5_rec.nb", "sha256": "9d073b3ee01deee358bf929dd8952d4d355c9545f4a93d8070605581b4c21c0c", "size": 16718470, "spdx": "Apache-2.0" },
    { "name": "ppocrv5_dict.txt", "path": "ppocrv5_dict.txt", "sha256": "17665d27ed39f0deb82007859992d626d3105d0ee4578c120b7c72138dc04d05", "size": 74011, "spdx": "Apache-2.0" }
  ],
  "models": {
    "det": ["ppocrv5_det.nb"],
    "rec": ["ppocrv5_rec.nb"],
    "dict": ["ppocrv5_dict.txt"]
  }
}
```

`fetchPaddleNativeLibs` skips empty `sha256` values, verifies non-empty ones and hard-fails on
mismatch, so this file is safe to land before the release exists (it is fetched by name, not by ABI
folder). Then run it once to drop the libraries into `core/ocr/src/main/jniLibs/<abi>/` (git-ignored):

```bash
./gradlew :core:ocr:fetchPaddleNativeLibs
```

Verify the packaged APK contains `lib/arm64-v8a/libpaddle_ocr.so` and `lib/arm64-v8a/libomp.so`
(and the same for `armeabi-v7a`).

## 2. `core/ocr/src/main/java/mihon/core/ocr/paddle/PaddleOcrEngine.kt` — model file names

```diff
     companion object {
         private const val MODEL_DIR = "paddle_ocr"
-        private const val DET_MODEL = "ppocrv4_det.nb"
-        private const val REC_MODEL = "ppocrv4_japan_rec.nb"
-        private const val DICT_TXT = "ppocrv4_japan_dict.txt"
+        private const val DET_MODEL = "ppocrv5_det.nb"
+        private const val REC_MODEL = "ppocrv5_rec.nb"
+        private const val DICT_TXT = "ppocrv5_dict.txt"
```

The `nativeInit` call, the `TIMING|`/`LINE|` parsing and every threshold stay exactly as they are.

## 3. `core/ocr/src/main/java/mihon/core/ocr/paddle/ModelDownloader.kt` — required models + base URL

```diff
 private const val DEFAULT_BASE_URL =
-    "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v1"
+    "https://github.com/komi-kun/komi-kun-ocr-models/releases/download/ocr-models-v2"
```

```diff
- * Downloads and verifies the PaddleOCR det/rec models and `ppocrv4_japan_dict.txt` into
+ * Downloads and verifies the PaddleOCR det/rec models and `ppocrv5_dict.txt` into
```

```diff
     companion object {
         private const val MANIFEST_NAME = "manifest.json"
         private const val PADDLE_DIR = "paddle_ocr"

-        private const val DET_MODEL = "ppocrv4_det.nb"
-        private const val REC_MODEL = "ppocrv4_japan_rec.nb"
-        private const val DICT_TXT = "ppocrv4_japan_dict.txt"
+        private const val DET_MODEL = "ppocrv5_det.nb"
+        private const val REC_MODEL = "ppocrv5_rec.nb"
+        private const val DICT_TXT = "ppocrv5_dict.txt"

         /** Must stay in sync with [PaddleOcrEngine]'s model constants. */
         val REQUIRED_MODELS = listOf(DET_MODEL, REC_MODEL, DICT_TXT)
     }
```

`downloadAndVerify` already uses `entry.path`, so runtime model download needs no other change.
If `importFromUri` is used, the user-supplied zip must contain the three files under a `paddle_ocr/`
prefix (that is where `verifyInstalled` looks).

## 4. Docs to update (optional)

`core/ocr/README.md` mentions `ocr-models-v1` and the `ppocrv4_*` names — update those strings.

## 5. `app/build.gradle.kts` — nothing to do

`keepDebugSymbols += listOf("libhoshidicts_jni", "libpaddle_ocr")` is already present, and
`splits.abi.include("armeabi-v7a", "arm64-v8a", "x86_64")` stays as-is.

---

## Caveats to record in the app

* **`x86_64` is not published.** Paddle-Lite ships no Android x86_64 prebuilt and `opt`-converted
  models are target-specific, so an ARM `.so`/model pair on x86_64 would abort, not degrade.
  `PaddleOcrEngine.supportedAbi()` may keep accepting `x86_64` for graceful degradation
  (`System.loadLibrary` throws, `init()` returns `false`). Do not add an `x86_64` entry to
  `paddle-manifest.json` or `splits.abi` until an x86_64 Paddle-Lite build + x86 model set exist.
* **`libc++_shared.so`** is intentionally absent from the manifest (see the `_comment`): AGP already
  packages the NDK copy for `-DANDROID_STL=c++_shared` modules. Shipping a second copy through
  `jniLibs` fails the app merge.
* **`libomp.so`** *is* required, because only this artifact uses OpenMP.
* **Model size grows.** `ppocrv5_rec.nb` is ~16.7 MB (vs ~9.9 MB for the v4 Japanese model) because
  PP-OCRv5's recogniser is a single unified multi-language model. That is a one-time,
  opt-in runtime download; it is not packaged in the APK.
* **Reproducibility.** The models in `ocr-models-v2` are the fixed official PaddlePaddle `.nb` bytes
  (pinned by SHA-256); the `.so` files are rebuilt from source and are not byte-identical across
  runs. Both consumers verify SHA-256 on every fetch, so republishing requires regenerating both
  manifests (`./build.sh --manifest-only`) and updating the app's `paddle-manifest.json`.
