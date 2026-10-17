#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
"""Host-parity check: the native engine vs the recorded PP-OCRv6-small benchmark.

Runs the *shipped* engine sources (built for Linux x86_64 against the Linux ONNX Runtime) over the
exact crop images the benchmark scored, and diffs the recognised text against
`raw_results.json`'s `rec.v6_small.text` for the same crop.

Two checks:

* ``crops``  — recognition-only parity (`paddle_ocr_host parity`) on every saved crop of one
  condition. Expected: near-100 % exact match, because both sides now use the same ONNX model,
  the same `rec_image_height = 48`, the same `ceil(48*w/h)` width and the same greedy CTC decode.
* ``pages``  — full pipeline (`paddle_ocr_host full`) on the page images, reporting box counts and
  the recognised Japanese text, for a sanity check of detection + reading order end-to-end.

Usage:
    parity_check.py --host out/paddle_ocr_host --models out/models \
        --bench /tmp/opencode/ocr-bench --cond orig [--limit N]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unicodedata

DET = "ppocrv6_small_det.onnx"
REC = "ppocrv6_small_rec.onnx"
DICT = "ppocrv6_small_dict.txt"


def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text or "").split())


def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def to_ppm(src: str, dst: str) -> None:
    try:
        import cv2  # type: ignore
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2 could not read {src}")
        if not cv2.imwrite(dst, img):
            raise RuntimeError(f"cv2 could not write {dst}")
        return
    except ImportError:
        pass
    from PIL import Image  # type: ignore
    with Image.open(src) as img:
        img.convert("RGB").save(dst, format="PPM")


def check_crops(args) -> int:
    bench = args.bench
    results = json.load(open(os.path.join(bench, "raw_results.json"), encoding="utf-8"))
    records = [r for r in results["records"] if r["cond"] == args.cond]
    records = [r for r in records if "v6_small" in r["rec"]]
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"no records for condition {args.cond}", file=sys.stderr)
        return 1

    exact = 0
    same_norm = 0
    cer_num = 0
    cer_den = 0
    samples = []

    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for r in records:
            src = os.path.join(bench, "crops", args.cond, r["title"], f"{r['stem']}_{r['box_id']}.png")
            if not os.path.isfile(src):
                continue
            dst = os.path.join(tmp, f"{r['title']}_{r['stem']}_{r['box_id']}.ppm")
            to_ppm(src, dst)
            rows.append((dst, "1" if r["vertical"] else "0", r["rec"]["v6_small"]["text"]))
        if not rows:
            print("no crops on disk (run the benchmark's run_bench.py first)", file=sys.stderr)
            return 1

        list_path = os.path.join(tmp, "crops.txt")
        with open(list_path, "w", encoding="utf-8") as handle:
            for dst, vertical, _ in rows:
                handle.write(f"{dst}\t{vertical}\n")

        cmd = [args.host, "parity",
               os.path.join(args.models, DET), os.path.join(args.models, REC),
               os.path.join(args.models, DICT), list_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            return proc.returncode

        native = {}
        for line in proc.stdout.splitlines():
            if not line.startswith("TEXT\t"):
                continue
            _, path, text = line.split("\t", 2)
            native[path] = text

        for dst, _, expected in rows:
            got = native.get(dst)
            if got is None:
                continue
            if got == expected:
                exact += 1
            if norm(got) == norm(expected):
                same_norm += 1
            cer_num += lev(norm(expected), norm(got))
            cer_den += max(len(norm(expected)), 1)
            if got != expected and len(samples) < args.max_samples:
                samples.append((expected, got))

    total = len(rows)
    print(f"crops condition={args.cond} n={total}")
    print(f"  exact byte match : {exact}/{total} = {100.0 * exact / total:.2f}%")
    print(f"  NFKC+ws match    : {same_norm}/{total} = {100.0 * same_norm / total:.2f}%")
    print(f"  CER (micro)      : {100.0 * cer_num / cer_den:.2f}%")
    if samples:
        print("  differing samples (benchmark -> native):")
        for expected, got in samples:
            print(f"    {expected!r} -> {got!r}")
    return 0


def _run_full(args, pages, tmp):
    """Runs the full pipeline over already-converted PPM `pages`; returns {basename: [lines]}."""
    cmd = [args.host, "full",
           os.path.join(args.models, DET), os.path.join(args.models, REC),
           os.path.join(args.models, DICT)] + [p[0] for p in pages]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    out = {}
    current = None
    for line in proc.stdout.splitlines():
        if line.startswith("### "):
            current = os.path.basename(line[4:])
            out[current] = []
        elif line.startswith("TIMING|") and current is not None:
            out[current].append(("TIMING", line.split("|")))
        elif line.startswith("LINE|") and current is not None:
            out[current].append(("LINE", line.split("|", 7)))
    return out


def _page_ppm(args, bench, meta, tmp):
    pages = []
    for page in meta:
        src = os.path.join(bench, "pages", args.cond, page["title"], f"{page['stem']}.png")
        if not os.path.isfile(src):
            continue
        dst = os.path.join(tmp, f"{page['title']}_{page['stem']}.ppm")
        to_ppm(src, dst)
        pages.append((dst, page))
    return pages


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_detect(args) -> int:
    """Detection comparison against the benchmark's reference runs.

    Two things are reported:

    * **native vs reference PP-OCRv6-small** — the benchmark's `det_stats` recorded the box count
      of its own Python DB post-process for every model, so the native engine's box count can be
      compared page-by-page with the *same* model. This is the meaningful parity number.
    * **native vs the v5 crop set** (informational) — the benchmark's saved crops come from the
      shipped v5 detector, so region/text agreement with them measures the v5 -> v6 detector
      change, not an engine difference.
    """
    bench = args.bench
    if args.cond != "orig":
        # `raw_results.json` stores the crop bbox in ORIGINAL page coordinates for every
        # condition, so only `orig` is directly comparable.
        print("detect mode only supports --cond orig", file=sys.stderr)
        return 2
    meta = json.load(open(os.path.join(bench, "pages_meta.json"), encoding="utf-8"))
    results = json.load(open(os.path.join(bench, "raw_results.json"), encoding="utf-8"))
    det_stats = results["det_stats"]
    by_page = {}
    for r in results["records"]:
        if r["cond"] != args.cond or "v6_small" not in r["rec"]:
            continue
        by_page.setdefault(f"{r['title']}/{r['stem']}", []).append(r)
    if args.limit:
        meta = meta[: args.limit]

    matched = 0
    exact = 0
    cer_num = 0
    cer_den = 0
    bench_total = 0
    native_total = 0
    ref_total = 0
    misses = 0
    ref_deltas = []
    with tempfile.TemporaryDirectory() as tmp:
        pages = _page_ppm(args, bench, meta, tmp)
        out = _run_full(args, pages, tmp)

        for dst, page in pages:
            key = f"{page['title']}/{page['stem']}"
            base = by_page.get(key, [])
            entries = out.get(os.path.basename(dst), [])
            timing = next((e for e in entries if e[0] == "TIMING"), None)
            native = [([float(v) for v in parts[3].split(",")], parts[7])
                      for kind, parts in entries if kind == "LINE"]

            native_boxes_count = int(timing[1][7]) if timing is not None else 0
            native_total += native_boxes_count
            bench_total += len(base)
            reference = det_stats.get(f"{key}|{args.cond}|v6_small")
            if reference is not None:
                ref_total += reference["nboxes"]
                ref_deltas.append(native_boxes_count - reference["nboxes"])

            # Resolve the page size from the PPM header so normalised coords can be compared.
            with open(dst, "rb") as handle:
                header = handle.read(64).split()
            width, height = int(header[1]), int(header[2])

            native_boxes = [([box[0] * width, box[1] * height, box[2] * width, box[3] * height], text)
                            for box, text in native]
            used = set()
            for r in base:
                rb = r["bbox"]
                best = None
                best_iou = 0.5
                for i, (nb, text) in enumerate(native_boxes):
                    if i in used:
                        continue
                    score = _iou(rb, nb)
                    if score > best_iou:
                        best_iou = score
                        best = (i, text)
                if best is None:
                    misses += 1
                    continue
                used.add(best[0])
                matched += 1
                expected = r["rec"]["v6_small"]["text"]
                if best[1] == expected:
                    exact += 1
                cer_num += lev(norm(expected), norm(best[1]))
                cer_den += max(len(norm(expected)), 1)

    print(f"detect condition={args.cond} pages={len(pages)}")
    print(f"  box counts: native(v6-small)={native_total} reference-v6-small={ref_total} "
          f"benchmark-v5-cropset={bench_total}")
    if ref_deltas:
        exact_pages = sum(1 for d in ref_deltas if d == 0)
        print(f"  native vs reference v6-small per page: identical={exact_pages}/{len(ref_deltas)} "
              f"max|diff|={max(abs(d) for d in ref_deltas)} "
              f"mean|diff|={sum(abs(d) for d in ref_deltas) / len(ref_deltas):.2f}")
    print(f"  vs v5 cropset: IoU>=0.5 matched={matched} unmatched={misses}")
    if matched:
        print(f"  matched text exact : {exact}/{matched} = {100.0 * exact / matched:.2f}%")
        print(f"  matched text CER   : {100.0 * cer_num / cer_den:.2f}%")
    return 0


def check_pages(args) -> int:
    bench = args.bench
    meta = json.load(open(os.path.join(bench, "pages_meta.json"), encoding="utf-8"))
    if args.limit:
        meta = meta[: args.limit]
    if not meta:
        return 1
    subprocess.run([args.host, "info", os.path.join(args.models, DET),
                    os.path.join(args.models, REC), os.path.join(args.models, DICT)],
                   check=True, capture_output=True, text=True)

    total_lines = 0
    with tempfile.TemporaryDirectory() as tmp:
        pages = _page_ppm(args, bench, meta, tmp)
        out = _run_full(args, pages, tmp)
        for dst, _ in pages:
            entries = out.get(os.path.basename(dst), [])
            timing = next((e for e in entries if e[0] == "TIMING"), None)
            if timing is not None:
                parts = timing[1]
                print(f"  {os.path.basename(dst)}: boxes={parts[7]} "
                      f"total={parts[1]}ms (det={parts[3]} post={parts[4]} rec={parts[6]})")
            for kind, parts in entries:
                if kind != "LINE":
                    continue
                total_lines += 1
                if args.show_text and total_lines <= args.max_samples:
                    print(f"    {parts[7]}")
    print(f"pages n={len(pages)} recognised lines={total_lines}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="out/paddle_ocr_host")
    ap.add_argument("--models", default="out/models")
    ap.add_argument("--bench", required=True,
                    help="benchmark directory containing raw_results.json, crops/, pages/ "
                         "(i.e. /tmp/opencode/ocr-bench from the PP-OCRv6 evaluation; not in this repo)")
    ap.add_argument("--cond", default="orig", choices=["orig", "ai", "classical"])
    ap.add_argument("--mode", default="crops", choices=["crops", "pages", "detect", "both"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--show-text", action="store_true")
    args = ap.parse_args()

    rc = 0
    if args.mode in ("crops", "both"):
        rc |= check_crops(args)
    if args.mode in ("pages", "both"):
        rc |= check_pages(args)
    if args.mode == "detect":
        rc |= check_detect(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
