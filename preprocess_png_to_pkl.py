
# -*- coding: utf-8 -*-
"""Convert PNG samples under /app/seed/<mission_id> into FAWA pkl files.

Default input layout:
    /app/seed/<mission_id>/png_dir/*.png
    /app/seed/<mission_id>/gt.txt
    /app/seed/<mission_id>/target.txt

Default outputs:
    /app/img_data/<mission_id>-<font_name>.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: str) -> str:
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError("mission_id only supports letters, digits, underscore, dot and hyphen")
    return mission_id


def normalize_image(img: Image.Image, target_height: int):
    """Convert a PIL image to a normalized grayscale array of shape (width, height)."""
    if img.mode != "L":
        img = img.convert("L")

    w, h = img.size
    if h != target_height:
        new_w = int(round(w * target_height / float(h)))
        img = img.resize((new_w, target_height), Image.LANCZOS)
        w, h = img.size

    arr = np.asarray(img, dtype=np.float32) / 255.0
    # Repository format: text=1.0, background=0.0.
    # Dark text on light background should be inverted.
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.T, w


def load_text_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_img_data(png_paths, target_height: int, pad_width=None):
    data = []
    lengths = []

    for path in png_paths:
        img = Image.open(path)
        arr, width = normalize_image(img, target_height)
        data.append(arr)
        lengths.append(width)

    if pad_width is None:
        pad_width = max(lengths)

    padded_data = []
    for arr in data:
        w, h = arr.shape
        if w > pad_width:
            raise ValueError(f"Image width {w} is larger than pad_width {pad_width}")
        padded = np.zeros((pad_width, h), dtype=np.float32)
        padded[:w, :] = arr
        padded_data.append(padded)

    input_img = np.stack(padded_data, axis=0)
    return input_img, lengths


def main():
    parser = argparse.ArgumentParser(description="Preprocess PNGs into mission-tagged FAWA pickle files")
    parser.add_argument("--mission_id", required=True, type=str,
                        help="Mission id. Data is read from /app/seed/<mission_id> by default.")
    parser.add_argument("--app_root", type=str, default="/app", help="Application root. Default: /app")
    parser.add_argument("--seed_root", type=str, default=None,
                        help="Seed root. Default: <app_root>/seed")
    parser.add_argument("--png_dir", type=str, default=None,
                        help="Directory containing input PNG files. Default: <seed_root>/<mission_id>/png_dir")
    parser.add_argument("--gt_txt", type=str, default=None,
                        help="Ground-truth text file. Default: <seed_root>/<mission_id>/gt.txt")
    parser.add_argument("--target_txt", type=str, default=None,
                        help="Target text file. Default: <seed_root>/<mission_id>/target.txt")
    parser.add_argument("--font_name", type=str, required=True,
                        help="Font name used for img_data/<mission_id>-<font_name>.pkl")
    parser.add_argument("--case", type=str, required=True,
                        help="Case name used for attack_pair/<mission_id>-<font_name>-<case>.pkl")
    parser.add_argument("--height", type=int, default=48,
                        help="Target image height after resize")
    parser.add_argument("--pad_width", type=int, default=None,
                        help="Optional width to pad all images to. Default=max image width")
    parser.add_argument("--output_img_data", type=str, default=None,
                        help="Output directory for img_data pickle. Default: <app_root>/img_data")
    parser.add_argument("--output_attack_pair", type=str, default=None,
                        help="Output directory for attack_pair pickle. Default: <app_root>/attack_pair")
    parser.add_argument("--ext", type=str, default="png",
                        help="Image file extension to load from png_dir")
    args = parser.parse_args()

    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(args.app_root)
    seed_root = Path(args.seed_root) if args.seed_root else app_root / "seed"
    # mission_seed_dir = seed_root / mission_id
    mission_seed_dir = seed_root / mission_id / "user_dataset"

    png_dir = Path(args.png_dir) if args.png_dir else mission_seed_dir / "png_dir"
    gt_txt_path = Path(args.gt_txt) if args.gt_txt else mission_seed_dir / "gt.txt"
    target_txt_path = Path(args.target_txt) if args.target_txt else mission_seed_dir / "target.txt"

    output_img_data = Path(args.output_img_data) if args.output_img_data else app_root / "img_data"
    output_attack_pair = Path(args.output_attack_pair) if args.output_attack_pair else app_root / "attack_pair"

    png_paths = sorted(glob(str(png_dir / f"*.{args.ext}")))
    if len(png_paths) == 0:
        raise FileNotFoundError(f"No .{args.ext} files found in {png_dir}")
    if not gt_txt_path.exists():
        raise FileNotFoundError(f"Ground-truth text file not found: {gt_txt_path}")
    if not target_txt_path.exists():
        raise FileNotFoundError(f"Target text file not found: {target_txt_path}")

    gt_txt = load_text_lines(gt_txt_path)
    target_txt = load_text_lines(target_txt_path)

    if len(gt_txt) != len(png_paths) or len(target_txt) != len(png_paths):
        raise ValueError(
            "The number of PNG files and the number of text lines must match. "
            f"png={len(png_paths)}, gt={len(gt_txt)}, target={len(target_txt)}"
        )

    input_img, len_x = build_img_data(png_paths, args.height, args.pad_width)

    output_img_data.mkdir(parents=True, exist_ok=True)
    output_attack_pair.mkdir(parents=True, exist_ok=True)

    img_data_path = output_img_data / f"{mission_id}-{args.font_name}.pkl"
    attack_pair_path = output_attack_pair / f"{mission_id}-{args.font_name}-{args.case}.pkl"

    with img_data_path.open("wb") as f:
        pickle.dump((input_img, len_x, gt_txt), f)
    with attack_pair_path.open("wb") as f:
        pickle.dump((gt_txt, target_txt), f)

    print(f"Wrote {img_data_path} shape={input_img.shape} count={len(input_img)}")
    print(f"Wrote {attack_pair_path} count={len(gt_txt)}")


if __name__ == "__main__":
    main()
