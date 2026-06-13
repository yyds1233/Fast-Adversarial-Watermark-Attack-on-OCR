# -*- coding: utf-8 -*-
"""Convert PNG samples under /app/seed/<mission_id>/user_dataset into FAWA pkl files.

Default input layout:
    /app/seed/<mission_id>/user_dataset/png_dir/*.png
    /app/seed/<mission_id>/user_dataset/value.txt
    /app/seed/<mission_id>/user_dataset/target.txt

Backward compatibility:
    If value.txt does not exist, this script will try gt.txt.

Supported value.txt / target.txt formats:

1. New named format:
    img_0000.png toy
    img_0001.png states

2. Old line-only format:
    toy
    states

Default outputs:
    /app/img_data/<mission_id>-<font_name>.pkl
    /app/img_data/<mission_id>-<font_name>.meta.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.meta.pkl
"""

from __future__ import annotations

import argparse
import pickle
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: str) -> str:
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError(
            "mission_id only supports letters, digits, underscore, dot and hyphen"
        )
    return mission_id


def normalize_image(img: Image.Image, target_height: int) -> Tuple[np.ndarray, int]:
    """Convert PIL image to normalized grayscale array with shape (width, height)."""
    if img.mode != "L":
        img = img.convert("L")

    w, h = img.size

    if h != target_height:
        new_w = int(round(w * target_height / float(h)))
        resampling = getattr(Image, "Resampling", Image)
        img = img.resize((new_w, target_height), resampling.LANCZOS)

    w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0

    # FAWA 原始逻辑中通常是文字为 1、背景为 0。
    # 常见黑字白底图片需要反色。
    if arr.mean() > 0.5:
        arr = 1.0 - arr

    return arr.T, w


def load_text_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_named_or_ordered_labels(
    txt_path: Path,
    sample_names: List[str],
    field_name: str,
) -> List[str]:
    """Parse label text file.

    New format:
        img_0000.png toy

    Old format:
        toy

    Return labels ordered by sample_names.
    """
    lines = load_text_lines(txt_path)

    if len(lines) != len(sample_names):
        raise ValueError(
            f"{field_name} line count does not match PNG count. "
            f"png={len(sample_names)}, {field_name}={len(lines)}, file={txt_path}"
        )

    sample_name_set = set(sample_names)
    parsed_named: Dict[str, str] = {}
    named_count = 0

    for raw_line in lines:
        parts = raw_line.split(maxsplit=1)

        if len(parts) == 2:
            name = Path(parts[0]).name
            label = parts[1]

            if name in sample_name_set:
                parsed_named[name] = label
                named_count += 1

    if named_count == len(lines):
        missing = [name for name in sample_names if name not in parsed_named]
        if missing:
            raise ValueError(
                f"{field_name} is missing labels for samples: {missing[:10]}"
            )

        return [parsed_named[name] for name in sample_names]

    if named_count == 0:
        # Backward compatible: one label per line, ordered by sorted PNG names.
        return lines

    raise ValueError(
        f"{field_name} mixes named and line-only formats. "
        f"Please use either all '<filename> <label>' lines or all label-only lines. "
        f"file={txt_path}"
    )


def build_img_data(
    png_paths: List[str],
    target_height: int,
    pad_width=None,
):
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

    for path, arr in zip(png_paths, data):
        w, h = arr.shape

        if w > pad_width:
            raise ValueError(
                f"Image width {w} is larger than pad_width {pad_width}: {path}"
            )

        padded = np.zeros((pad_width, h), dtype=np.float32)
        padded[:w, :] = arr
        padded_data.append(padded)

    input_img = np.stack(padded_data, axis=0)

    return input_img, lengths


def resolve_value_txt_path(args, mission_seed_dir: Path) -> Path:
    if args.value_txt:
        return Path(args.value_txt)

    if args.gt_txt:
        return Path(args.gt_txt)

    value_txt_path = mission_seed_dir / "value.txt"
    if value_txt_path.exists():
        return value_txt_path

    legacy_gt_path = mission_seed_dir / "gt.txt"
    if legacy_gt_path.exists():
        return legacy_gt_path

    return value_txt_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PNGs into mission-tagged FAWA pickle files"
    )

    parser.add_argument(
        "--mission_id",
        required=True,
        type=str,
        help="Mission id. Data is read from /app/seed/<mission_id>/user_dataset by default.",
    )
    parser.add_argument(
        "--app_root",
        type=str,
        default="/app",
        help="Application root. Default: /app",
    )
    parser.add_argument(
        "--seed_root",
        type=str,
        default=None,
        help="Seed root. Default: /app/seed",
    )
    parser.add_argument(
        "--png_dir",
        type=str,
        default=None,
        help="Directory containing input PNG files. Default: /app/seed/<mission_id>/user_dataset/png_dir",
    )
    parser.add_argument(
        "--value_txt",
        type=str,
        default=None,
        help="Value/ground-truth text file. Default: /app/seed/<mission_id>/user_dataset/value.txt",
    )
    parser.add_argument(
        "--gt_txt",
        type=str,
        default=None,
        help="Deprecated alias for --value_txt. Kept for compatibility.",
    )
    parser.add_argument(
        "--target_txt",
        type=str,
        default=None,
        help="Target text file. Default: /app/seed/<mission_id>/user_dataset/target.txt",
    )
    parser.add_argument(
        "--font_name",
        type=str,
        required=True,
        help="Font name used for img_data/<mission_id>-<font_name>.pkl",
    )
    parser.add_argument(
        "--case",
        type=str,
        required=True,
        help="Case name used for attack_pair/<mission_id>-<font_name>-<case>.pkl",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=48,
        help="Target image height after resize. Default: 48",
    )
    parser.add_argument(
        "--pad_width",
        type=int,
        default=None,
        help="Optional width to pad all images to. Default: max image width",
    )
    parser.add_argument(
        "--output_img_data",
        type=str,
        default=None,
        help="Output directory for img_data pickle. Default: /app/img_data",
    )
    parser.add_argument(
        "--output_attack_pair",
        type=str,
        default=None,
        help="Output directory for attack_pair pickle. Default: /app/attack_pair",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="png",
        help="Image file extension to load from png_dir. Default: png",
    )

    args = parser.parse_args()

    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(args.app_root)
    seed_root = Path(args.seed_root) if args.seed_root else app_root / "seed"

    mission_seed_dir = seed_root / mission_id / "user_dataset"

    png_dir = Path(args.png_dir) if args.png_dir else mission_seed_dir / "png_dir"
    value_txt_path = resolve_value_txt_path(args, mission_seed_dir)
    target_txt_path = (
        Path(args.target_txt) if args.target_txt else mission_seed_dir / "target.txt"
    )

    output_img_data = (
        Path(args.output_img_data) if args.output_img_data else app_root / "img_data"
    )
    output_attack_pair = (
        Path(args.output_attack_pair)
        if args.output_attack_pair
        else app_root / "attack_pair"
    )

    png_paths = sorted(glob(str(png_dir / f"*.{args.ext}")))

    if len(png_paths) == 0:
        raise FileNotFoundError(f"No .{args.ext} files found in {png_dir}")

    if not value_txt_path.exists():
        raise FileNotFoundError(
            f"Value text file not found: {value_txt_path}. "
            f"Expected value.txt under {mission_seed_dir}, or pass --value_txt."
        )

    if not target_txt_path.exists():
        raise FileNotFoundError(f"Target text file not found: {target_txt_path}")

    sample_names = [Path(p).name for p in png_paths]

    value_txt = parse_named_or_ordered_labels(
        value_txt_path,
        sample_names,
        "value_txt",
    )

    target_txt = parse_named_or_ordered_labels(
        target_txt_path,
        sample_names,
        "target_txt",
    )

    input_img, len_x = build_img_data(
        png_paths=png_paths,
        target_height=args.height,
        pad_width=args.pad_width,
    )

    output_img_data.mkdir(parents=True, exist_ok=True)
    output_attack_pair.mkdir(parents=True, exist_ok=True)

    img_data_path = output_img_data / f"{mission_id}-{args.font_name}.pkl"
    img_meta_path = output_img_data / f"{mission_id}-{args.font_name}.meta.pkl"

    attack_pair_path = output_attack_pair / (
        f"{mission_id}-{args.font_name}-{args.case}.pkl"
    )
    attack_pair_meta_path = output_attack_pair / (
        f"{mission_id}-{args.font_name}-{args.case}.meta.pkl"
    )

    # 保持原 tuple 格式不变，basic_grad.py / wm_grad.py 不需要改。
    # 第三个字段变量名虽然可能还叫 gt_txt，本质上现在就是 value_txt。
    with img_data_path.open("wb") as f:
        pickle.dump((input_img, len_x, value_txt), f)

    with attack_pair_path.open("wb") as f:
        pickle.dump((value_txt, target_txt), f)

    metadata = {
        "format_version": 3,
        "mission_id": mission_id,
        "font_name": args.font_name,
        "case": args.case,
        "png_dir": str(png_dir),
        "sample_names": sample_names,
        "png_paths": [str(Path(p)) for p in png_paths],
        "value_txt": value_txt,
        # 兼容旧代码字段。
        "gt_txt": value_txt,
        "target_txt": target_txt,
        "height": args.height,
        "pad_width": int(input_img.shape[1]),
        "len_x": len_x,
    }

    with img_meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    with attack_pair_meta_path.open("wb") as f:
        pickle.dump(metadata, f)

    print(f"Wrote {img_data_path} shape={input_img.shape} count={len(input_img)}")
    print(f"Wrote {img_meta_path}")
    print(f"Wrote {attack_pair_path} count={len(value_txt)}")
    print(f"Wrote {attack_pair_meta_path}")


if __name__ == "__main__":
    main()