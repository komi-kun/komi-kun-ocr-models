#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Renders the synthetic Japanese test pages used by the off-device conformance check.
# Requires Pillow and a CJK font:
#   ./make_pages.py [outdir]
import os
import sys

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]

HORIZONTAL = [
    "吾輩は猫である。名前はまだ無い。",
    "どこで生れたかとんと見当がつかぬ。",
    "何でも薄暗いじめじめした所で",
    "ニャーニャー泣いていた事だけは記憶している。",
    "これは横書きのテストです。ABC 123",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    raise SystemExit("no CJK font found; install fonts-noto-cjk")


def horizontal(path: str, font_path_size: int = 40) -> None:
    width, height = 1000, 700
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = load_font(font_path_size)
    y = 60
    for line in HORIZONTAL:
        draw.text((60, y), line, font=font, fill=(0, 0, 0))
        y += 110
    img.save(path)


def vertical(path: str, font_size: int = 44) -> None:
    width, height = 900, 1300
    img = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)
    columns = ["吾輩は猫である", "名前はまだ無い", "どこで生れたか", "とんと見当が", "つかぬ", "何でも薄暗い"]
    x = width - 90
    for column in columns:
        y = 60
        for ch in column:
            draw.text((x, y), ch, font=font, fill=(20, 20, 20))
            y += 52
        x -= 62
    img.save(path)


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    horizontal(os.path.join(outdir, "page_horizontal.ppm"))
    vertical(os.path.join(outdir, "page_vertical.ppm"))
    print("wrote page_horizontal.ppm + page_vertical.ppm to", outdir)


if __name__ == "__main__":
    main()
