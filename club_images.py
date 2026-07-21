"""
club_images.py — Combine multiple frame_NNNN.png images into grid montages.

Lays out several encoded frame images on one canvas, each cell separated by a
black border (which also acts as spacing so a later sloppy per-cell crop
doesn't pick up a neighboring cell's finder markers), with the original
filename printed below each image.

This does not change the underlying frame format at all — it only affects
how images are packaged for a single screenshot. The companion program,
split_images.py, reverses this: it takes a screenshot of a montage and crops
it back into individual images. Each crop just needs to be roughly centered
on its cell; decoder.py's own finder-marker detection handles any imprecision
in the crop boundary, the same way it already handles a sloppy screenshot
crop of a single image.

Usage:
    python3 club_images.py --input encoded --output montages --per-montage 9 --cols 3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

DEFAULT_BORDER = 24        # black border/spacing around each image, in px
DEFAULT_LABEL_HEIGHT = 36  # space reserved below each image for its filename
DEFAULT_PER_MONTAGE = 9
DEFAULT_COLS = 3
BG_COLOR = (0, 0, 0)
LABEL_BG = (0, 0, 0)
LABEL_COLOR = (255, 255, 255)


def load_font(size: int) -> ImageFont.ImageFont:
    """Use a real TrueType font if one is findable, else fall back to PIL's default bitmap font."""
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_montage(
    images: List[Path],
    cols: int,
    rows: int,
    border: int,
    label_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    """
    Lay out `images` in a `cols`-wide, `rows`-tall grid on one canvas.

    `rows` is always based on the configured --per-montage/--cols, not the
    actual number of images in this group, so every montage file has the
    identical canvas size and grid shape (the last montage may have empty
    trailing cells). This lets split_images.py divide any montage screenshot
    into the same fixed grid without needing to know how many images were
    actually in it.
    """
    first = Image.open(images[0]).convert("RGB")
    img_w, img_h = first.size

    cell_w = img_w + 2 * border
    cell_h = img_h + 2 * border + label_height

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(images):
        col = idx % cols
        row = idx // cols
        cell_x0 = col * cell_w
        cell_y0 = row * cell_h

        img = Image.open(path).convert("RGB")
        if img.size != (img_w, img_h):
            raise ValueError(
                f"{path.name} is {img.size}, expected {(img_w, img_h)} to match the first image. "
                "All images in one montage must be the same size."
            )
        canvas.paste(img, (cell_x0 + border, cell_y0 + border))

        label = path.name
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = cell_x0 + (cell_w - text_w) // 2
        text_y = cell_y0 + border + img_h + (label_height - (text_bbox[3] - text_bbox[1])) // 2
        draw.text((text_x, text_y), label, fill=LABEL_COLOR, font=font)

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine frame images into grid montages for one screenshot each.")
    parser.add_argument("--input", required=True, help="Directory containing frame_NNNN.png files.")
    parser.add_argument("--output", required=True, help="Directory to write montage_NNNN.png files.")
    parser.add_argument("--per-montage", type=int, default=DEFAULT_PER_MONTAGE,
                         help="How many frame images to place in each montage (default: 9).")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Grid columns per montage (default: 3).")
    parser.add_argument("--border", type=int, default=DEFAULT_BORDER, help="Black border/spacing per cell, px.")
    parser.add_argument("--label-height", type=int, default=DEFAULT_LABEL_HEIGHT,
                         help="Space reserved below each image for its filename, px.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(input_dir.glob("*.png"))
    if not frame_paths:
        print(f"No PNG files found in {input_dir}")
        raise SystemExit(1)

    font = load_font(max(14, args.label_height - 10))

    num_montages = math.ceil(len(frame_paths) / args.per_montage)
    fixed_rows = math.ceil(args.per_montage / args.cols)
    print(f"{len(frame_paths)} frame images -> {num_montages} montage(s) of up to {args.per_montage} each "
          f"({args.cols} columns x {fixed_rows} rows, fixed grid shape for every montage)")

    for m in range(num_montages):
        group = frame_paths[m * args.per_montage:(m + 1) * args.per_montage]
        montage = build_montage(group, args.cols, fixed_rows, args.border, args.label_height, font)
        out_path = output_dir / f"montage_{m + 1:04d}.png"
        montage.save(out_path, format="PNG")
        print(f"Wrote {out_path.name}: {len(group)} images ({group[0].name} .. {group[-1].name}), "
              f"canvas {montage.size}")

    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()
