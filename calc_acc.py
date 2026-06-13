#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate exact-match ACC for CalamariOCR.

Input dataset:
    /app/seed/<mission_id>/user_dataset/png_dir/
    /app/seed/<mission_id>/user_dataset/value.txt
        or /app/seed/<mission_id>/user_dataset/gt.txt

Supported label formats:
    1) filename label
       img_0000.png toy
       img_0001.png states

    2) label only, aligned to sorted images
       toy
       states

Model:
    model json is passed via --model_dir and --model_path, for example:
    /app/weight/<mission_id>/<any_folder>/4.ckpt.json

Output:
    /app/adv_eval/acc_<mission_id>.txt
        one ACC value only, for example:
        0.873333

    /app/ACC_result/ACC_<mission_id>.txt
        per-sample prediction detail:
        image_name ground_truth prediction

        example:
        img_0000.png toy foy
        img_0001.png toy toy

ACC definition:
    strict exact match after strip(). One wrong letter means incorrect.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image
import tensorflow as tf
import zipfile

from calamari_ocr.ocr import Predictor
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def validate_mission_id(mission_id: str) -> str:
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError("mission_id only supports letters, digits, underscore, dot and hyphen")
    return mission_id


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def invert(data: np.ndarray) -> np.ndarray:
    if data.max() < 1.5:
        return 1.0 - data
    return 255.0 - data


def transpose(data: np.ndarray) -> np.ndarray:
    if len(data.shape) != 2:
        return np.swapaxes(data, 1, 2)
    return data.T


def image_to_model_input(path: Path) -> np.ndarray:
    """Convert a gray image to the model input layout used by the attack code.

    In the attack code, show(img) reconstructs the visual image with:
        raw = transpose(invert(img))

    Therefore, from a normal raw image, the model-side input is:
        img = transpose(invert(raw))

    Output shape is roughly [width, height] and values are float32 in [0, 1].
    """
    img = Image.open(path).convert("L")
    raw = np.asarray(img).astype(np.float32) / 255.0
    model_input = transpose(invert(raw)).astype(np.float32)
    return model_input


def pad_batch(images: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not images:
        raise ValueError("empty image batch")

    max_w = max(img.shape[0] for img in images)
    max_h = max(img.shape[1] for img in images)

    batch = np.ones((len(images), max_w, max_h), dtype=np.float32)
    seq_lens = []

    for i, img in enumerate(images):
        w, h = img.shape[:2]
        batch[i, :w, :h] = img
        seq_lens.append(w)

    return batch, np.asarray(seq_lens, dtype=np.int32)


def parse_label_line(line: str) -> Tuple[str | None, str]:
    line = line.strip()
    if not line:
        return None, ""

    parts = line.split(maxsplit=1)
    if len(parts) == 1:
        return None, parts[0].strip()

    first, rest = parts[0].strip(), parts[1].strip()
    if Path(first).suffix.lower() in IMAGE_EXTS:
        return Path(first).name, rest

    # Fallback: treat the full line as label-only when the first token is not an image name.
    return None, line


def load_labels(label_path: Path, image_paths: Sequence[Path]) -> List[str]:
    lines = [
        ln.strip()
        for ln in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]

    named: Dict[str, str] = {}
    plain: List[str] = []

    for line in lines:
        name, label = parse_label_line(line)
        if name is None:
            plain.append(label)
        else:
            named[name] = label

    if named:
        labels = []
        missing = []
        for img_path in image_paths:
            if img_path.name not in named:
                missing.append(img_path.name)
            else:
                labels.append(named[img_path.name])
        if missing:
            raise ValueError(
                f"label file {label_path} is missing labels for {len(missing)} images, "
                f"first missing: {missing[:5]}"
            )
        return labels

    if len(plain) < len(image_paths):
        raise ValueError(
            f"label count is less than image count: labels={len(plain)}, images={len(image_paths)}"
        )

    return plain[: len(image_paths)]


def find_label_file(user_dataset: Path) -> Path:
    value_path = user_dataset / "value.txt"
    gt_path = user_dataset / "gt.txt"

    if value_path.exists():
        return value_path
    if gt_path.exists():
        return gt_path

    raise FileNotFoundError(f"missing value.txt or gt.txt under {user_dataset}")


def resolve_model_path(model_dir: Path, model_path: str) -> Path:
    raw = Path(model_path)
    if raw.is_absolute() and raw.exists():
        return raw

    candidate = model_dir / raw.name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"model json not found: {candidate}")


def predict_texts(model_json: Path, image_paths: Sequence[Path], batch_size: int) -> List[str]:
    predictor = Predictor(checkpoint=str(model_json), batch_size=1, processes=10)
    network = predictor.network
    sess = network.session
    graph = network.graph
    codec = network.codec
    decode = codec.decode

    predictions: List[str] = []

    with graph.as_default():
        inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
        (
            output_seq_len,
            time_major_logits,
            time_major_softmax,
            logits,
            softmax,
            sparse_decoded,
            decoded_tuple,
            other,
        ) = network.create_network(inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE)

        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            imgs = [image_to_model_input(p) for p in batch_paths]
            batch_img, batch_len = pad_batch(imgs)

            decoded = sess.run(
                sparse_decoded,
                feed_dict={
                    inputs: batch_img,
                    input_seq_len: batch_len,
                    dropout_rate: 0,
                },
            )
            batch_indices = TensorflowModel._TensorflowModel__sparse_to_lists(decoded)
            predictions.extend(["".join(decode(index)).strip() for index in batch_indices])

    return predictions


def clean_field(value: str) -> str:
    """Keep result txt one-line and whitespace-separated.

    Labels/predictions in this task are normally single words. This function prevents
    tabs/newlines from breaking the output format.
    """
    value = "" if value is None else str(value)
    value = value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    value = " ".join(value.split())
    return value


def write_one_value(path: Path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{value:.2f}\n", encoding="utf-8")


def write_prediction_lines(
    path: Path,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    predictions: Sequence[str],
    total: int,
) -> None:
    """Write ACC result detail.

    Format:
        image_name ground_truth prediction

    Example:
        img_0000.png toy foy
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for img_path, label, pred in zip(image_paths[:total], labels[:total], predictions[:total]):
        image_name = img_path.name
        label = clean_field(label)
        pred = clean_field(pred)

        if pred == "":
            pred = "<EMPTY>"

        lines.append(f"{image_name} {label} {pred}")

    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_detail_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")

def package_acc_result(acc_result_file: Path, output_zip: Path) -> None:
    """Package ACC_<mission_id>.txt into /app/adv_sample/<mission_id>.zip.

    Zip 内只保留文件名，不带 /app/ACC_result/ 这种路径。
    """
    if not acc_result_file.exists():
        raise FileNotFoundError(f"ACC result txt not found: {acc_result_file}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    if output_zip.exists():
        output_zip.unlink()

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(acc_result_file, arcname=acc_result_file.name)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate strict exact-match ACC for CalamariOCR")
    parser.add_argument("--mission_id", required=True, type=str)
    parser.add_argument("--app_root", default="/app", type=str)
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument(
        "--write_detail",
        action="store_true",
        help="Also write /app/adv_eval/acc_<mission_id>_detail.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(args.app_root)

    user_dataset = app_root / "seed" / mission_id / "user_dataset"
    png_dir = user_dataset / "png_dir"

    if not user_dataset.exists():
        raise FileNotFoundError(f"user_dataset not found: {user_dataset}")
    if not png_dir.exists():
        raise FileNotFoundError(f"png_dir not found: {png_dir}")

    image_paths = sorted(
        [p for p in png_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=natural_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"no images found under {png_dir}")

    label_file = find_label_file(user_dataset)
    labels = [x.strip() for x in load_labels(label_file, image_paths)]

    model_json = resolve_model_path(Path(args.model_dir), args.model_path)
    predictions = predict_texts(model_json, image_paths, args.batch_size)

    total = min(len(labels), len(predictions))
    correct = 0

    for pred, label in zip(predictions[:total], labels[:total]):
        if pred.strip() == label.strip():
            correct += 1

    acc = (float(correct) / float(total) * 100.0) if total else 0.0

    adv_eval_acc = app_root / "adv_eval" / f"acc_{mission_id}.txt"
    acc_result = app_root / "ACC_result" / f"ACC_{mission_id}.txt"

    # /app/adv_eval/acc_<mission_id>.txt 只写 ACC 数值
    write_one_value(adv_eval_acc, acc)

    # /app/ACC_result/ACC_<mission_id>.txt 写逐样本结果：
    # img_0000.png toy foy
    write_prediction_lines(acc_result, image_paths, labels, predictions, total)
    final_zip = app_root / "ACC_result" / f"{mission_id}.zip"
    package_acc_result(acc_result, final_zip)

    if args.write_detail:
        detail_path = app_root / "adv_eval" / f"acc_{mission_id}_detail.json"
        examples = []
        for idx, (img_path, pred, label) in enumerate(zip(image_paths, predictions, labels)):
            if idx >= 50:
                break
            examples.append({
                "image": img_path.name,
                "pred": pred,
                "label": label,
                "correct": pred.strip() == label.strip(),
            })
        write_detail_json(detail_path, {
            "mission_id": mission_id,
            "acc": acc,
            "correct": correct,
            "total": total,
            "label_file": str(label_file),
            "model_json": str(model_json),
            "examples_first_50": examples,
        })

    print(f"ACC={acc:.6f}, correct={correct}, total={total}")
    print(f"Wrote ACC value: {adv_eval_acc}")
    print(f"Wrote ACC result detail: {acc_result}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise