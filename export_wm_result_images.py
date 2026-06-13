
# -*- coding: utf-8 -*-
"""Export PNG images from mission-tagged wm_result pickle files."""
from __future__ import annotations

import argparse
import pickle
import re
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: str | None) -> str | None:
    if mission_id is None:
        return None
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError("mission_id only supports letters, digits, underscore, dot and hyphen")
    return mission_id


def invert(data):
    if data.max() <= 1.0:
        return 1.0 - data
    return 255 - data


def transpose(data):
    if len(data.shape) == 2:
        return data.T
    return np.swapaxes(data, 1, 2)


def cvt2raw(data):
    return transpose(invert(data))


def array_to_image(array):
    if array.dtype != np.uint8:
        if array.max() <= 0.5:
            array = (array + 0.5) * 255.0
        elif array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")
    raise ValueError(f"Unsupported image array shape: {array.shape}")


def save_grayscale_images(arr, out_dir, prefix):
    for i in range(arr.shape[0]):
        img = array_to_image(cvt2raw(arr[i]))
        img.save(out_dir / f"{prefix}_{i:04d}.png")


def save_rgb_images(arr, out_dir, prefix):
    for i in range(arr.shape[0]):
        img = array_to_image(arr[i])
        img.save(out_dir / f"{prefix}_{i:04d}.png")


def process_pkl(path, out_root, save_adv, save_rgb, save_wm0):
    with open(path, "rb") as f:
        data = pickle.load(f)

    out_root.mkdir(parents=True, exist_ok=True)
    base = Path(path).stem

    if save_adv and len(data) > 6:
        adv_img = data[6]
        if isinstance(adv_img, np.ndarray) and adv_img.ndim == 3:
            save_dir = out_root / f"{base}_adv"
            save_dir.mkdir(exist_ok=True)
            save_grayscale_images(adv_img, save_dir, "adv")
            print(f"Saved {adv_img.shape[0]} adv images to {save_dir}")

    if save_wm0 and len(data) > 3:
        wm0_img = data[3]
        if isinstance(wm0_img, np.ndarray) and wm0_img.ndim == 3:
            save_dir = out_root / f"{base}_wm0"
            save_dir.mkdir(exist_ok=True)
            save_grayscale_images(wm0_img, save_dir, "wm0")
            print(f"Saved {wm0_img.shape[0]} wm0 images to {save_dir}")

    if save_rgb and len(data) > 10:
        rgb_img = data[10]
        if isinstance(rgb_img, np.ndarray) and rgb_img.ndim == 4 and rgb_img.shape[3] == 3:
            save_dir = out_root / f"{base}_rgb"
            save_dir.mkdir(exist_ok=True)
            save_rgb_images(rgb_img, save_dir, "rgb")
            print(f"Saved {rgb_img.shape[0]} rgb images to {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export images from mission-tagged wm_result pickle files.")
    parser.add_argument("--mission_id", type=str, default=None,
                        help="Mission id. If provided and --input is a directory, only <mission_id>-*.pkl is processed.")
    parser.add_argument("--app_root", type=str, default="/app", help="Application root. Default: /app")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to a wm_result pickle file or a directory. Default: <app_root>/wm_result")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory. Default: <app_root>/exported_wm_images[/<mission_id>]")
    parser.add_argument("--save_adv", action="store_true", help="Save adv_img images.")
    parser.add_argument("--save_rgb", action="store_true", help="Save rgb_img images.")
    parser.add_argument("--save_wm0", action="store_true", help="Save wm0_img images.")
    parser.add_argument("--pattern", type=str, default=None,
                        help="Glob pattern when input is a directory. Default: <mission_id>-*.pkl or *.pkl")
    args = parser.parse_args()

    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(args.app_root)
    input_path = Path(args.input) if args.input else app_root / "wm_result"

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = app_root / "exported_wm_images"
        if mission_id:
            output_path = output_path / mission_id

    if input_path.is_dir():
        pattern = args.pattern or (f"{mission_id}-*.pkl" if mission_id else "*.pkl")
        paths = sorted(input_path.glob(pattern))
    else:
        paths = [input_path]

    if not paths:
        raise FileNotFoundError(f"No pickle files found in {input_path}")

    for p in paths:
        print(f"Processing {p}")
        process_pkl(p, output_path / p.stem, args.save_adv, args.save_rgb, args.save_wm0)


if __name__ == "__main__":
    main()
