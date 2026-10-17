#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 komi-kun contributors
"""Extract the CTC character dictionary from a PaddleOCR `inference.yml`.

PaddleOCR's `CTCLabelDecode` convention: index 0 is the CTC blank, then the model's
`PostProcess.character_dict`, then a trailing ASCII space class. The recogniser's output class
count is therefore `len(character_dict) + 2` (PP-OCRv6 small: 18 708 + 2 = 18 710).

The produced file is exactly what the engine reads: one character per line, in model order. The
engine prepends `#` (blank) and appends `' '` (space) itself.
"""
import sys

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: extract_dict.py <inference.yml> <out.txt>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    cfg = yaml.safe_load(open(src, encoding="utf-8"))
    chars = list(cfg["PostProcess"]["character_dict"])
    with open(dst, "w", encoding="utf-8") as handle:
        for ch in chars:
            handle.write(ch + "\n")
    kana = sum(1 for c in chars if "\u3040" <= c <= "\u30ff")
    han = sum(1 for c in chars if "\u4e00" <= c <= "\u9fff")
    print(f"{dst}: n={len(chars)} kana={kana} han={han} blank+space => C={len(chars) + 2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
