# -*- coding: utf-8 -*-
"""Export PNG images, per-folder value.txt, and per-folder SSIM metrics from wm_result pkl.

Default input:
    /app/wm_result/<mission_id>-*.pkl

Default output:
    /app/exported_wm_images/<mission_id>/<wm_result_stem>/
        adv_png_dir/
            img_0000.png
            img_0001.png
            ...
            value.txt
            ssim.txt
            adv_pred.txt
        rgb_png_dir/
            img_0000.png
            img_0001.png
            ...
            value.txt
            ssim.txt
            adv_pred.txt

Notes:
    1. target.txt is not exported.
    2. ssim_summary.txt is not exported.
    3. value.txt replaces gt.txt.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    try:
        from skimage.measure import compare_ssim as skimage_ssim
    except Exception:
        skimage_ssim = None


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: Optional[str]) -> Optional[str]:
    if mission_id is None:
        return None

    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError(
            "mission_id only supports letters, digits, underscore, dot and hyphen"
        )

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


def to_uint8_image_array(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)

    if array.dtype == np.uint8:
        return array

    array = array.astype(np.float32)

    if array.max() <= 0.5:
        array = (array + 0.5) * 255.0
    elif array.max() <= 1.0:
        array = array * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)

    return array


def array_to_image(array: np.ndarray) -> Image.Image:
    array = to_uint8_image_array(array)

    if array.ndim == 2:
        return Image.fromarray(array, mode="L")

    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")

    raise ValueError(f"Unsupported image array shape: {array.shape}")


def save_grayscale_images(
    arr: np.ndarray,
    out_dir: Path,
    names: List[str],
    fallback_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(arr.shape[0]):
        name = names[i] if i < len(names) else f"{fallback_prefix}_{i:04d}.png"
        img = array_to_image(cvt2raw(arr[i]))
        img.save(out_dir / name)


def save_rgb_images(
    arr: np.ndarray,
    out_dir: Path,
    names: List[str],
    fallback_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(arr.shape[0]):
        name = names[i] if i < len(names) else f"{fallback_prefix}_{i:04d}.png"
        img = array_to_image(arr[i])
        img.save(out_dir / name)


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def parse_named_or_ordered_labels(
    txt_path: Path,
    sample_names: List[str],
    field_name: str,
) -> List[str]:
    lines = read_lines(txt_path)

    if len(lines) < len(sample_names):
        raise ValueError(
            f"{field_name} line count is smaller than sample count. "
            f"samples={len(sample_names)}, {field_name}={len(lines)}, file={txt_path}"
        )

    lines = lines[: len(sample_names)]
    sample_name_set = set(sample_names)

    named_count = 0
    mapping: Dict[str, str] = {}

    for raw_line in lines:
        parts = raw_line.split(maxsplit=1)

        if len(parts) == 2:
            name = Path(parts[0]).name
            label = parts[1]

            if name in sample_name_set:
                named_count += 1
                mapping[name] = label

    if named_count == len(lines):
        missing = [name for name in sample_names if name not in mapping]
        if missing:
            raise ValueError(
                f"{field_name} is missing labels for samples: {missing[:10]}"
            )

        return [mapping[name] for name in sample_names]

    if named_count == 0:
        return lines

    raise ValueError(
        f"{field_name} mixes named and line-only formats. file={txt_path}"
    )


def write_label_file(path: Path, names: List[str], labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for name, label in zip(names, labels):
            f.write(f"{name} {label}\n")


def write_prediction_file(path: Path, names: List[str], preds: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for name, pred in zip(names, preds):
            f.write(f"{name} {pred}\n")


def parse_identity_from_result_name(
    result_path: Path,
    mission_id: Optional[str],
    font_name: Optional[str],
    case: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Infer font_name and case from result pkl name.

    Example:
        20260613-Arial-easy-linf-eps0.3-ieps0.05-iter10-positive.pkl

    Infer:
        font_name=Arial
        case=easy
    """
    if font_name and case:
        return font_name, case

    stem = result_path.stem

    if mission_id and stem.startswith(mission_id + "-"):
        stem = stem[len(mission_id) + 1 :]

    left = stem.split("-l", 1)[0]

    if "-" not in left:
        return font_name, case

    inferred_font, inferred_case = left.split("-", 1)

    return font_name or inferred_font, case or inferred_case


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def load_img_data(
    app_root: Path,
    mission_id: str,
    font_name: str,
):
    img_data_path = app_root / "img_data" / f"{mission_id}-{font_name}.pkl"

    if not img_data_path.exists():
        return None, None, None

    data = load_pickle(img_data_path)

    if not isinstance(data, tuple) or len(data) < 3:
        raise ValueError(f"Unsupported img_data pickle format: {img_data_path}")

    input_img, len_x, value_txt = data[:3]

    return input_img, len_x, value_txt


def load_metadata(
    app_root: Path,
    mission_id: str,
    font_name: str,
    case: str,
    n: int,
    fallback_value_txt: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Load sample names and value labels.

    Priority:
        1. /app/img_data/<mission_id>-<font_name>.meta.pkl
        2. /app/attack_pair/<mission_id>-<font_name>-<case>.meta.pkl
        3. /app/seed/<mission_id>/user_dataset/value.txt
        4. /app/seed/<mission_id>/user_dataset/gt.txt
        5. fallback value labels from img_data pkl
        6. generated empty labels
    """
    img_meta_path = app_root / "img_data" / f"{mission_id}-{font_name}.meta.pkl"
    attack_meta_path = (
        app_root / "attack_pair" / f"{mission_id}-{font_name}-{case}.meta.pkl"
    )

    for meta_path in [img_meta_path, attack_meta_path]:
        if meta_path.exists():
            meta = load_pickle(meta_path)

            sample_names = list(meta.get("sample_names", []))[:n]
            value_txt = list(meta.get("value_txt", meta.get("gt_txt", [])))[:n]

            if sample_names:
                return {
                    "sample_names": sample_names,
                    "value_txt": value_txt,
                }

    seed_user_dataset = app_root / "seed" / mission_id / "user_dataset"
    png_dir = seed_user_dataset / "png_dir"
    value_path = seed_user_dataset / "value.txt"
    legacy_gt_path = seed_user_dataset / "gt.txt"

    png_paths = sorted(glob(str(png_dir / "*.png")))

    if png_paths:
        sample_names = [Path(p).name for p in png_paths[:n]]
    else:
        sample_names = [f"img_{i:04d}.png" for i in range(n)]

    if value_path.exists():
        value_txt = parse_named_or_ordered_labels(
            value_path,
            sample_names,
            "value_txt",
        )
    elif legacy_gt_path.exists():
        value_txt = parse_named_or_ordered_labels(
            legacy_gt_path,
            sample_names,
            "gt_txt",
        )
    elif fallback_value_txt is not None:
        value_txt = list(fallback_value_txt)[:n]
    else:
        value_txt = [""] * n

    return {
        "sample_names": sample_names,
        "value_txt": value_txt,
    }


def safe_ssim(gray_a_255: np.ndarray, gray_b_255: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")

    a = np.asarray(gray_a_255, dtype=np.float32)
    b = np.asarray(gray_b_255, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(f"SSIM image shape mismatch: {a.shape} vs {b.shape}")

    min_side = min(a.shape[0], a.shape[1])

    if min_side < 3:
        return float("nan")

    win_size = min(7, min_side)

    if win_size % 2 == 0:
        win_size -= 1

    return float(skimage_ssim(a, b, data_range=255.0, win_size=win_size))


def calc_ssim_for_output_dir(
    original_png_dir: Path,
    output_png_dir: Path,
    sample_names: List[str],
) -> List[float]:
    scores = []

    for name in sample_names:
        original_path = original_png_dir / name
        output_path = output_png_dir / name

        if not original_path.exists() or not output_path.exists():
            scores.append(float("nan"))
            continue

        try:
            original_img = Image.open(original_path).convert("L")
            output_img = Image.open(output_path).convert("L")

            if original_img.size != output_img.size:
                original_img = original_img.resize(output_img.size)

            original_arr = np.asarray(original_img, dtype=np.float32)
            output_arr = np.asarray(output_img, dtype=np.float32)

            score = safe_ssim(original_arr, output_arr)
            scores.append(score)
        except Exception:
            scores.append(float("nan"))

    return scores


def write_ssim_file(path: Path, names: List[str], scores: List[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for name, score in zip(names, scores):
            if math.isnan(score):
                f.write(f"{name} nan\n")
            else:
                f.write(f"{name} {score:.8f}\n")


def process_pkl(
    path: Path,
    out_root: Path,
    app_root: Path,
    mission_id: Optional[str],
    font_name: Optional[str],
    case: Optional[str],
    save_adv: bool,
    save_rgb: bool,
    save_wm0: bool,
) -> None:
    data = load_pickle(path)
    out_root.mkdir(parents=True, exist_ok=True)

    inferred_font_name, inferred_case = parse_identity_from_result_name(
        result_path=path,
        mission_id=mission_id,
        font_name=font_name,
        case=case,
    )

    if not mission_id:
        raise ValueError("--mission_id is required for label and SSIM export")

    if not inferred_font_name:
        raise ValueError(
            "Cannot infer font_name from result pkl name. Please pass --font_name."
        )

    if not inferred_case:
        raise ValueError(
            "Cannot infer case from result pkl name. Please pass --case."
        )

    adv_img = None
    wm0_img = None
    rgb_img = None
    record_adv_text: List[str] = []

    # wm_grad.py 当前保存格式：
    # (
    #   0 pos,
    #   1 wm_mask,
    #   2 text_mask,
    #   3 wm0_img,
    #   4 record_text,
    #   5 accuracy,
    #   6 adv_img,
    #   7 record_adv_text,
    #   8 record_iter,
    #   9 (duration, last_iter),
    #   10 rgb_img,
    # )
    if len(data) > 6 and isinstance(data[6], np.ndarray):
        adv_img = data[6]

    if len(data) > 3 and isinstance(data[3], np.ndarray):
        wm0_img = data[3]

    if len(data) > 7 and isinstance(data[7], list):
        record_adv_text = [str(x) for x in data[7]]

    if len(data) > 10 and isinstance(data[10], np.ndarray):
        rgb_img = data[10]

    if adv_img is not None:
        image_source = adv_img
    elif rgb_img is not None:
        image_source = rgb_img
    elif wm0_img is not None:
        image_source = wm0_img
    else:
        image_source = None

    if image_source is None:
        raise ValueError(f"No exportable image array found in {path}")

    n = image_source.shape[0]

    _original_img, _len_x, fallback_value_txt = load_img_data(
        app_root=app_root,
        mission_id=mission_id,
        font_name=inferred_font_name,
    )

    meta = load_metadata(
        app_root=app_root,
        mission_id=mission_id,
        font_name=inferred_font_name,
        case=inferred_case,
        n=n,
        fallback_value_txt=fallback_value_txt,
    )

    sample_names = meta["sample_names"][:n]
    value_txt = meta["value_txt"][:n]

    if len(sample_names) < n:
        sample_names += [
            f"img_{i:04d}.png" for i in range(len(sample_names), n)
        ]

    if len(value_txt) < n:
        value_txt += [""] * (n - len(value_txt))

    seed_user_dataset = app_root / "seed" / mission_id / "user_dataset"
    original_png_dir = seed_user_dataset / "png_dir"

    if save_adv and adv_img is not None:
        adv_dir = out_root / "adv_png_dir"

        save_grayscale_images(
            arr=adv_img,
            out_dir=adv_dir,
            names=sample_names,
            fallback_prefix="adv",
        )
        print(f"Saved {adv_img.shape[0]} adv images to {adv_dir}")

        write_label_file(
            path=adv_dir / "value.txt",
            names=sample_names,
            labels=value_txt,
        )

        adv_ssim_scores = calc_ssim_for_output_dir(
            original_png_dir=original_png_dir,
            output_png_dir=adv_dir,
            sample_names=sample_names,
        )

        write_ssim_file(
            path=adv_dir / "ssim.txt",
            names=sample_names,
            scores=adv_ssim_scores,
        )

        if record_adv_text:
            write_prediction_file(
                path=adv_dir / "adv_pred.txt",
                names=sample_names,
                preds=record_adv_text[:n],
            )

    if save_rgb and rgb_img is not None:
        rgb_dir = out_root / "rgb_png_dir"

        save_rgb_images(
            arr=rgb_img,
            out_dir=rgb_dir,
            names=sample_names,
            fallback_prefix="rgb",
        )
        print(f"Saved {rgb_img.shape[0]} rgb images to {rgb_dir}")

        write_label_file(
            path=rgb_dir / "value.txt",
            names=sample_names,
            labels=value_txt,
        )

        rgb_ssim_scores = calc_ssim_for_output_dir(
            original_png_dir=original_png_dir,
            output_png_dir=rgb_dir,
            sample_names=sample_names,
        )

        write_ssim_file(
            path=rgb_dir / "ssim.txt",
            names=sample_names,
            scores=rgb_ssim_scores,
        )

        if record_adv_text:
            write_prediction_file(
                path=rgb_dir / "adv_pred.txt",
                names=sample_names,
                preds=record_adv_text[:n],
            )

    if save_wm0 and wm0_img is not None:
        wm0_dir = out_root / "wm0_png_dir"

        save_grayscale_images(
            arr=wm0_img,
            out_dir=wm0_dir,
            names=sample_names,
            fallback_prefix="wm0",
        )
        print(f"Saved {wm0_img.shape[0]} wm0 images to {wm0_dir}")

        write_label_file(
            path=wm0_dir / "value.txt",
            names=sample_names,
            labels=value_txt,
        )

        wm0_ssim_scores = calc_ssim_for_output_dir(
            original_png_dir=original_png_dir,
            output_png_dir=wm0_dir,
            sample_names=sample_names,
        )

        write_ssim_file(
            path=wm0_dir / "ssim.txt",
            names=sample_names,
            scores=wm0_ssim_scores,
        )

    print(f"Wrote per-folder value.txt and ssim.txt under {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Export images, per-folder value.txt and per-folder SSIM from mission-tagged wm_result pickle files."
    )

    parser.add_argument(
        "--mission_id",
        type=str,
        default=None,
        help="Mission id. Required for value.txt and SSIM export.",
    )
    parser.add_argument(
        "--app_root",
        type=str,
        default="/app",
        help="Application root. Default: /app",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to a wm_result pickle file or a directory. Default: /app/wm_result",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory. Default: /app/exported_wm_images/<mission_id>",
    )
    parser.add_argument(
        "--font_name",
        type=str,
        default=None,
        help="Optional font name. If not given, inferred from pkl filename.",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Optional case name. If not given, inferred from pkl filename.",
    )
    parser.add_argument(
        "--save_adv",
        action="store_true",
        help="Save adv_img images and its own value.txt/ssim.txt.",
    )
    parser.add_argument(
        "--save_rgb",
        action="store_true",
        help="Save rgb_img images and its own value.txt/ssim.txt.",
    )
    parser.add_argument(
        "--save_wm0",
        action="store_true",
        help="Save wm0_img images and its own value.txt/ssim.txt.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern when input is a directory. Default: <mission_id>-*.pkl or *.pkl",
    )

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

        process_pkl(
            path=p,
            out_root=output_path / p.stem,
            app_root=app_root,
            mission_id=mission_id,
            font_name=args.font_name,
            case=args.case,
            save_adv=args.save_adv,
            save_rgb=args.save_rgb,
            save_wm0=args.save_wm0,
        )


if __name__ == "__main__":
    main()