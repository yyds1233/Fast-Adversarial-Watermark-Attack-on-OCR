# -*- coding: utf-8 -*-
"""
Apply mission_id path/output adaptation for Fast-Adversarial-Watermark-Attack-on-OCR.
Run this script from the repository root, for example:

    cd /app
    python apply_mission_id_patch.py

It backs up modified files as *.bak_mission before overwriting/updating them.
"""
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path.cwd()
BACKUP_SUFFIX = ".bak_mission"


def backup(path: Path) -> None:
    if path.exists():
        bak = path.with_name(path.name + BACKUP_SUFFIX)
        if not bak.exists():
            bak.write_bytes(path.read_bytes())


def write_file(rel_path: str, content: str) -> None:
    path = ROOT / rel_path
    backup(path)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    print(f"updated {rel_path}")


def patch_util() -> None:
    path = ROOT / "util.py"
    if not path.exists():
        print("skip util.py: not found")
        return
    text = path.read_text(encoding="utf-8")
    if "--mission_id" in text:
        print("util.py already has --mission_id")
        return
    backup(path)
    needle = 'parser.add_argument("--clip_max", help="the maximum value of images", type=float)'
    insert = '''parser.add_argument("--clip_max", help="the maximum value of images", type=float)
    parser.add_argument("--mission_id", required=True, type=str,
                        help="Mission id. Used to read /app/seed/<mission_id>, /app/weight/<mission_id> and tag all outputs.")
    parser.add_argument("--app_root", default="/app", type=str,
                        help="Application root. Default: /app")
    parser.add_argument("--model_dir", default=None, type=str,
                        help="Optional override for model directory. Default: /app/weight/<mission_id>")'''
    if needle in text:
        text = text.replace(needle, insert, 1)
    else:
        # Fallback: insert before "return parser" inside get_argparse().
        text = re.sub(
            r"(\n\s*return\s+parser)",
            "\n    parser.add_argument(\"--mission_id\", required=True, type=str,\n"
            "                        help=\"Mission id. Used to read /app/seed/<mission_id>, /app/weight/<mission_id> and tag all outputs.\")\n"
            "    parser.add_argument(\"--app_root\", default=\"/app\", type=str,\n"
            "                        help=\"Application root. Default: /app\")\n"
            "    parser.add_argument(\"--model_dir\", default=None, type=str,\n"
            "                        help=\"Optional override for model directory. Default: /app/weight/<mission_id>\")"
            r"\1",
            text,
            count=1,
        )
    path.write_text(text, encoding="utf-8")
    print("updated util.py")


PREPROCESS = r'''
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
    mission_seed_dir = seed_root / mission_id

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
'''

BASIC_GRAD = r'''
# -*- coding: utf-8 -*-
# @Time : 13/1/20 15:31
# @Author : Lu Chen
"""Mission-aware basic gradient attack.

Reads:
    /app/img_data/<mission_id>-<font_name>.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.pkl
    /app/weight/<mission_id>/<model_path>

Writes:
    /app/attack_result/<mission_id>-<font_name>-<case>-l<pert_type>-eps<eps>-ieps<eps_iter>-iter<nb_iter>.pkl
"""
from __future__ import annotations

import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from cleverhans import utils_tf
from util import get_argparse, sparse_tuple_from
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel
from calamari_ocr.ocr import Predictor


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: str) -> str:
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError("mission_id only supports letters, digits, underscore, dot and hyphen")
    return mission_id


def resolve_model_path(app_root: Path, mission_id: str, model_path: str, model_dir: str | None = None) -> Path:
    raw_path = Path(model_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    base_dir = Path(model_dir) if model_dir else app_root / "weight" / mission_id
    candidates = [
        base_dir / model_path,
        base_dir / raw_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Model checkpoint not found. Tried: " + ", ".join(str(c) for c in candidates)
    )


def result_title(mission_id: str, font_name: str, case: str, pert_type: str,
                 eps: float, eps_iter: float, nb_iter: int) -> str:
    return f"{mission_id}-{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"


def main():
    parser = get_argparse()
    args = parser.parse_args()

    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(getattr(args, "app_root", "/app"))
    model_path = resolve_model_path(app_root, mission_id, args.model_path, getattr(args, "model_dir", None))

    predictor = Predictor(checkpoint=str(model_path), batch_size=1, processes=10)
    network = predictor.network
    sess, graph = network.session, network.graph
    encode, decode = network.codec.encode, network.codec.decode

    with graph.as_default():
        inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
        output_seq_len, time_major_logits, time_major_softmax, logits, softmax, sparse_decoded, decoded_tuple, other = network.create_network(
            inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE
        )
        decoded = sparse_decoded

        loss = tf.nn.ctc_loss(
            labels=targets,
            inputs=time_major_logits,
            sequence_length=output_seq_len,
            time_major=True,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=True,
        )
        loss = -tf.reduce_mean(loss, name="loss")
        grad, = tf.gradients(loss, inputs)

        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = tf.cast(1e-12, grad.dtype)
        divisor = tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True)
        norm_grad = grad / tf.maximum(avoid_zero_div, divisor)

        m = tf.placeholder(tf.float32, shape=inputs.get_shape().as_list(), name="momentum")
        acc_m = m + norm_grad
        grad = acc_m

        # ord = inf
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)

        # ord = 2
        square = tf.maximum(1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
        scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)

    font_name = args.font_name
    case = args.case
    pert_type = args.pert_type
    eps = args.eps
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    batch_size = args.batch_size
    clip_min, clip_max = args.clip_min, args.clip_max

    img_data_path = app_root / "img_data" / f"{mission_id}-{font_name}.pkl"
    attack_pair_path = app_root / "attack_pair" / f"{mission_id}-{font_name}-{case}.pkl"
    attack_result_dir = app_root / "attack_result"
    attack_result_dir.mkdir(parents=True, exist_ok=True)

    if not img_data_path.exists():
        raise FileNotFoundError(f"img_data pkl not found: {img_data_path}")
    if not attack_pair_path.exists():
        raise FileNotFoundError(f"attack_pair pkl not found: {attack_pair_path}")

    with img_data_path.open("rb") as f:
        input_img, len_x, gt_txt = pickle.load(f)
    with attack_pair_path.open("rb") as f:
        _, target_txt = pickle.load(f)

    n_img = min(200, len(input_img), len(target_txt))
    input_img = input_img[:n_img]
    len_x = len_x[:n_img]
    gt_txt = gt_txt[:n_img]
    target_txt = target_txt[:n_img]

    with graph.as_default():
        adv_img = input_img.copy()
        m0 = np.zeros(input_img.shape)
        record_iter = np.zeros(input_img.shape[0])  # 0 means unsuccessful
        record_adv_text = []
        last_iter = 0

        batch_iter = len(input_img) // batch_size
        batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1
        start = time.time()

        for batch_i in tqdm(range(batch_iter)):
            start_idx = batch_size * batch_i
            end_idx = min(batch_size * (batch_i + 1), len(input_img))
            cur_batch_size = end_idx - start_idx

            batch_input_img = input_img[start_idx:end_idx]
            batch_adv_img = adv_img[start_idx:end_idx]
            batch_len_x = len_x[start_idx:end_idx]
            batch_m0 = m0[start_idx:end_idx]
            batch_target_text = target_txt[start_idx:end_idx]
            batch_target_index = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_text]
            batch_y = sparse_tuple_from(batch_target_index)
            batch_record_iter = np.zeros(cur_batch_size)
            scaled_perturbation = scaled_perturbation_2 if pert_type == "2" else scaled_perturbation_inf
            batch_adv_text = [""] * cur_batch_size

            for i in range(nb_iter):
                last_iter = i
                batch_pert, batch_adv_text_sparse = sess.run(
                    [scaled_perturbation, decoded],
                    feed_dict={
                        inputs: batch_adv_img,
                        input_seq_len: batch_len_x,
                        m: batch_m0,
                        targets: batch_y,
                        dropout_rate: 0,
                    },
                )

                batch_pert[batch_record_iter != 0] = 0
                batch_adv_img = batch_adv_img + eps_iter * batch_pert
                batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
                batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)
                adv_img[start_idx:end_idx] = batch_adv_img

                batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_text_sparse)
                batch_adv_text = ["".join(decode(index)) for index in batch_adv_index]

                for j in range(cur_batch_size):
                    if batch_adv_text[j] == batch_target_text[j] and batch_record_iter[j] == 0:
                        batch_record_iter[j] = i

                if np.sum(batch_record_iter == 0) == 0:
                    print(f"{i} break")
                    break

            record_iter[start_idx:end_idx] = batch_record_iter
            record_adv_text += batch_adv_text

        duration = time.time() - start

    title = result_title(mission_id, font_name, case, pert_type, eps, eps_iter, nb_iter)
    output_path = attack_result_dir / f"{title}.pkl"
    with output_path.open("wb") as f:
        pickle.dump((adv_img, record_adv_text, record_iter, (duration, last_iter)), f)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
'''

WM_GRAD = r'''
# -*- coding: utf-8 -*-
"""Mission-aware watermark gradient attack.

Reads:
    /app/img_data/<mission_id>-<font_name>.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.pkl
    /app/attack_result/<mission_id>-<font_name>-<case>-l<pert_type>-eps<eps>-ieps<eps_iter>-iter<nb_iter>.pkl
    /app/weight/<mission_id>/<model_path>

Writes:
    /app/wm_result/<mission_id>-<font_name>-<case>-l<pert_type>-eps<eps>-ieps<eps_iter>-iter<nb_iter>-positive.pkl
"""
from __future__ import annotations

import argparse
import pickle
import re
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage import morphology
from tqdm import tqdm
from trdg.generators import GeneratorFromStrings

from cleverhans import utils_tf
from util import cvt2Image, sparse_tuple_from
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel
from calamari_ocr.ocr import Predictor


MISSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def validate_mission_id(mission_id: str) -> str:
    if not mission_id or not MISSION_RE.match(mission_id):
        raise ValueError("mission_id only supports letters, digits, underscore, dot and hyphen")
    return mission_id


def resolve_model_path(app_root: Path, mission_id: str, model_path: str, model_dir: str | None = None) -> Path:
    raw_path = Path(model_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    base_dir = Path(model_dir) if model_dir else app_root / "weight" / mission_id
    candidates = [base_dir / model_path, base_dir / raw_path.name]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Model checkpoint not found. Tried: " + ", ".join(str(c) for c in candidates)
    )


def result_title(mission_id: str, font_name: str, case: str, pert_type: str,
                 eps: float, eps_iter: float, nb_iter: int, positive: bool = False) -> str:
    title = f"{mission_id}-{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
    return f"{title}-positive" if positive else title


def invert(data):
    if data.max() < 1.5:
        return 1 - data
    return 255 - data


def transpose(data):
    if len(data.shape) != 2:
        return np.swapaxes(data, 1, 2)
    return data.T


def cvt2raw(data):
    return transpose(invert(data))


def show(img):
    return cvt2Image(cvt2raw(img))


def rgb2hex(rgb):
    color = "#"
    for num in rgb:
        color += str(hex(num))[-2:].replace("x", "0").upper()
    return color


def gen_wm(rgb):
    generator = GeneratorFromStrings(
        strings=["eccv"],
        count=1,
        fonts=["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
        language="en",
        size=100,
        skewing_angle=10,
        random_skew=False,
        blur=0,
        random_blur=False,
        background_type=1,
        distorsion_type=0,
        distorsion_orientation=0,
        is_handwritten=False,
        width=-1,
        alignment=1,
        text_color=rgb2hex(rgb),
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(0, 0, 0, 0),
        fit=True,
    )
    img_list = [img for img, _ in generator]
    return img_list[0]


def get_text_mask(img: np.array):
    if img.max() <= 1:
        return img < 1 / 1.25
    return img < 255 / 1.25


def find_wm_pos(adv_img, input_img, ret_frame_img=False):
    pert = np.abs(cvt2raw(adv_img) - cvt2raw(input_img))
    pert = (pert > 1e-2) * 255.0
    wm_pos_list = []
    frame_img_list = []

    for src in pert:
        kernel = np.ones((3, 3), np.uint8)
        dilate = cv2.dilate(src, kernel, iterations=2)
        erode = cv2.erode(dilate, kernel, iterations=2)
        remove = morphology.remove_small_objects(erode.astype(bool), min_size=0)
        contours, _ = cv2.findContours(
            (remove * 255).astype("uint8"),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        wm_pos, frame_img = [], []
        for cont in contours:
            left_point = cont.min(axis=1).min(axis=0)
            right_point = cont.max(axis=1).max(axis=0)
            wm_pos.append(np.hstack((left_point, right_point)))
            if ret_frame_img:
                img = cv2.rectangle(
                    (remove * 255).astype("uint8").copy(),
                    (left_point[0], left_point[1]),
                    (right_point[0], right_point[1]),
                    (255, 255, 255),
                    2,
                )
                frame_img.append(img)

        wm_pos_list.append(wm_pos)
        frame_img_list.append(frame_img)

    if ret_frame_img:
        return wm_pos_list, frame_img_list
    return wm_pos_list


def cvt2rgb(gray_img, text_mask):
    gray_img = invert(gray_img)
    op_mask = (~(gray_img == 1)) & (~text_mask)
    rgb_img = np.ones(list(gray_img.shape) + [3])
    rgb_img[:, :, :, 0] = gray_img
    rgb_img[:, :, :, 1] = gray_img
    rgb_img[:, :, :, 2] = gray_img
    rgb_img[op_mask, 0] = 1
    rgb_img[op_mask, 1] = (gray_img[op_mask] - 0.299) / 0.587
    rgb_img[op_mask, 2] = 0
    return invert(rgb_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Mission-aware watermark gradient attack")
    parser.add_argument("font_name", type=str, help="Font name, e.g. Arial")
    parser.add_argument("case", type=str, help="Case name, e.g. easy")
    parser.add_argument("pert_type", type=str, choices=["2", "inf"], help="Perturbation type")
    parser.add_argument("eps", type=float, help="Perturbation clipping bound")
    parser.add_argument("eps_iter", type=float, help="Step size")
    parser.add_argument("nb_iter", type=int, help="Number of iterations")
    parser.add_argument("--mission_id", required=True, type=str,
                        help="Mission id used to read/write mission-tagged files")
    parser.add_argument("--model_path", default="4.ckpt.json", type=str,
                        help="Checkpoint file name under /app/weight/<mission_id>")
    parser.add_argument("--app_root", default="/app", type=str,
                        help="Application root. Default: /app")
    parser.add_argument("--model_dir", default=None, type=str,
                        help="Optional override for model directory. Default: /app/weight/<mission_id>")
    parser.add_argument("--batch_size", default=100, type=int,
                        help="Batch size. Default: 100")
    parser.add_argument("--clip_min", default=0.0, type=float)
    parser.add_argument("--clip_max", default=1.0, type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(args.app_root)

    font_name = args.font_name
    case = args.case
    pert_type = args.pert_type
    eps = args.eps
    eps_iter = args.eps_iter
    nb_iter = args.nb_iter
    batch_size = args.batch_size
    clip_min, clip_max = args.clip_min, args.clip_max

    model_path = resolve_model_path(app_root, mission_id, args.model_path, args.model_dir)
    predictor = Predictor(checkpoint=str(model_path), batch_size=1, processes=10)
    network = predictor.network
    sess, graph = network.session, network.graph
    codec = network.codec
    encode, decode = codec.encode, codec.decode

    with graph.as_default():
        inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()
        output_seq_len, time_major_logits, time_major_softmax, logits, softmax, sparse_decoded, decoded_tuple, other = network.create_network(
            inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE
        )
        decoded = sparse_decoded

        loss = tf.nn.ctc_loss(
            labels=targets,
            inputs=time_major_logits,
            sequence_length=output_seq_len,
            time_major=True,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=True,
        )
        loss = -tf.reduce_mean(loss, name="loss")
        grad, = tf.gradients(loss, inputs)

        red_ind = list(range(1, len(grad.get_shape())))
        avoid_zero_div = tf.cast(1e-12, grad.dtype)
        divisor = tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True)
        norm_grad = grad / tf.maximum(avoid_zero_div, divisor)

        m = tf.placeholder(tf.float32, shape=inputs.get_shape().as_list(), name="momentum")
        acc_m = m + norm_grad
        mask = tf.placeholder(tf.float32, shape=inputs.get_shape().as_list(), name="mask")
        grad = tf.multiply(acc_m, mask, name="mask_op")

        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)

        abs_grad = tf.abs(grad)
        max_abs_grad = tf.reduce_max(abs_grad, axis=red_ind, keepdims=True)
        tied_for_max = tf.cast(tf.equal(abs_grad, max_abs_grad), tf.float32)
        num_ties = tf.reduce_sum(tied_for_max, axis=red_ind, keepdims=True)
        optimal_perturbation = tf.sign(grad) * tied_for_max / num_ties
        scaled_perturbation_1 = utils_tf.mul(0.01, optimal_perturbation)

        square = tf.maximum(1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True))
        optimal_perturbation = grad / tf.sqrt(square)
        scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)

    img_data_path = app_root / "img_data" / f"{mission_id}-{font_name}.pkl"
    attack_pair_path = app_root / "attack_pair" / f"{mission_id}-{font_name}-{case}.pkl"
    attack_result_path = app_root / "attack_result" / f"{result_title(mission_id, font_name, case, pert_type, eps, eps_iter, nb_iter)}.pkl"
    wm_result_dir = app_root / "wm_result"
    wm_result_dir.mkdir(parents=True, exist_ok=True)

    if not img_data_path.exists():
        raise FileNotFoundError(f"img_data pkl not found: {img_data_path}")
    if not attack_pair_path.exists():
        raise FileNotFoundError(f"attack_pair pkl not found: {attack_pair_path}")
    if not attack_result_path.exists():
        raise FileNotFoundError(f"basic attack result pkl not found: {attack_result_path}")

    with img_data_path.open("rb") as f:
        input_img, len_x, gt_txt = pickle.load(f)
    input_img = np.asarray(input_img)

    with attack_result_path.open("rb") as f:
        adv_img, record_adv_text, record_iter, (duration, total_iter) = pickle.load(f)

    n_img = len(adv_img)
    input_img = input_img[:n_img]
    len_x = len_x[:n_img]
    gt_txt = gt_txt[:n_img]

    with attack_pair_path.open("rb") as f:
        _, target_txt = pickle.load(f)
    target_txt = target_txt[:n_img]

    pos, frames = find_wm_pos(adv_img, input_img, True)

    new_pos = []
    for _pos in pos:
        if len(_pos) > 1:
            new_pos.append(
                sorted(
                    _pos,
                    key=lambda x: (x[3] - x[1]) * (x[2] - x[0]),
                    reverse=True,
                )
            )
        else:
            new_pos.append(_pos)
    pos = new_pos

    grayscale = 0
    color = (grayscale, grayscale, grayscale)
    wm_img = gen_wm(color)
    wm_arr = np.array(wm_img.convert("L"))
    kernel = np.ones((5, 5), np.uint8)
    wm_arr = cv2.dilate(wm_arr, kernel, 2)
    wm_arr = cv2.erode(wm_arr, kernel, 2)
    bg_mask = ~(wm_arr != 255)

    grayscale = 174
    color = (grayscale, grayscale, grayscale)
    wm_img_arr = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
    wm_img_arr[bg_mask] = 255
    wm_img = Image.fromarray(wm_img_arr)

    wm0_img_list = []
    wm_mask_list = []
    text_mask_list = []
    for idx in range(len(input_img)):
        text_img = show(input_img[idx])
        text_mask = get_text_mask(np.array(text_img))
        rgb_img = Image.new(mode="RGB", size=text_img.size, color=(255, 255, 255))
        p = -int(wm_img.size[0] * np.tan(10 * np.pi / 180))
        right_shift = 10
        xp = pos[idx][0][0] + right_shift if len(pos[idx]) != 0 else right_shift
        rgb_img.paste(wm_img, box=(xp, p))
        wm_mask = (np.array(rgb_img.convert("L")) != 255)
        rgb_img.paste(text_img, mask=cvt2Image(text_mask))
        wm0_img_list.append(rgb_img)
        wm_mask_list.append(transpose(wm_mask))
        text_mask_list.append(transpose(text_mask))

    wm_mask = np.asarray(wm_mask_list)
    text_mask = np.asarray(text_mask_list)

    record_text = []
    wm0_img = pred_img = np.asarray([cvt2raw(np.array(img.convert("L"))) / 255 for img in wm0_img_list])
    batch_iter = len(input_img) // batch_size
    batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1

    for batch_i in range(batch_iter):
        start_idx = batch_size * batch_i
        end_idx = min(batch_size * (batch_i + 1), len(input_img))
        batch_img = pred_img[start_idx:end_idx]
        batch_len_x = len_x[start_idx:end_idx]
        batch_text = sess.run(
            decoded,
            feed_dict={inputs: batch_img, input_seq_len: batch_len_x, dropout_rate: 0},
        )
        batch_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_text)
        record_text += ["".join(decode(index)) for index in batch_index]

    cnt = 0
    for pred_txt, raw_txt in zip(record_text, gt_txt):
        if pred_txt == raw_txt:
            cnt += 1
    accuracy = cnt / len(gt_txt) if len(gt_txt) else 0

    target_index_list = [np.asarray([c for c in encode(t)]) for t in target_txt]
    wm_img_np = wm0_img

    with graph.as_default():
        adv_img = wm_img_np.copy()
        m0 = np.zeros(input_img.shape)
        record_iter = np.zeros(input_img.shape[0])
        record_mse = []
        record_mse_plus = []
        record_adv_text = []
        last_iter = 0
        start = time.time()

        for i in tqdm(range(nb_iter)):
            last_iter = i
            batch_iter = len(input_img) // batch_size
            batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1

            for batch_i in range(batch_iter):
                start_idx = batch_size * batch_i
                end_idx = min(batch_size * (batch_i + 1), len(input_img))
                batch_input_img = wm_img_np[start_idx:end_idx]
                batch_adv_img = adv_img[start_idx:end_idx]
                batch_len_x = len_x[start_idx:end_idx]
                batch_m0 = m0[start_idx:end_idx]
                batch_target_txt = target_txt[start_idx:end_idx]
                batch_mask = wm_mask[start_idx:end_idx]
                batch_record_iter = record_iter[start_idx:end_idx]
                batch_tmp_y = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_txt]
                batch_y = sparse_tuple_from(batch_tmp_y)
                scaled_perturbation = scaled_perturbation_2 if pert_type == "2" else scaled_perturbation_inf

                batch_pert = sess.run(
                    scaled_perturbation,
                    feed_dict={
                        inputs: batch_adv_img,
                        input_seq_len: batch_len_x,
                        m: batch_m0,
                        targets: batch_y,
                        mask: batch_mask,
                        dropout_rate: 0,
                    },
                )

                batch_pert[batch_record_iter != 0] = 0
                batch_adv_img = batch_adv_img + eps_iter * batch_pert * (batch_pert > 0)
                batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
                batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)
                adv_img[start_idx:end_idx] = batch_adv_img

            record_mse.append(np.mean(((adv_img - wm_img_np) * 255) ** 2))
            record_mse_plus.append(
                np.mean((((adv_img - wm_img_np) * ((adv_img - wm_img_np) > 0)) * 255) ** 2)
            )

            record_adv_text = []
            for batch_i in range(batch_iter):
                start_idx = batch_size * batch_i
                end_idx = min(batch_size * (batch_i + 1), len(input_img))
                batch_adv_img = adv_img[start_idx:end_idx]
                batch_len_x = len_x[start_idx:end_idx]
                batch_target_index = target_index_list[start_idx:end_idx]
                batch_adv_text = sess.run(
                    decoded,
                    feed_dict={inputs: batch_adv_img, input_seq_len: batch_len_x, dropout_rate: 0},
                )
                batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_text)
                record_adv_text += ["".join(decode(index)) for index in batch_adv_index]

                for j in range(len(batch_target_index)):
                    adv_index, target_index = batch_adv_index[j], batch_target_index[j]
                    idx_j = start_idx + j
                    if np.array_equal(adv_index, target_index) and record_iter[idx_j] == 0:
                        record_iter[idx_j] = i

            if np.sum(record_iter == 0) == 0:
                break

        duration = time.time() - start
        print(f"{last_iter} break. Time cost {duration:.4f} s")

    rgb_img = cvt2rgb(adv_img, text_mask)

    title = result_title(mission_id, font_name, case, pert_type, eps, eps_iter, nb_iter, positive=True)
    output_path = wm_result_dir / f"{title}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(
            (
                pos,
                wm_mask,
                text_mask,
                wm0_img,
                record_text,
                accuracy,
                adv_img,
                record_adv_text,
                record_iter,
                (duration, last_iter),
                rgb_img,
            ),
            f,
        )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
'''

EXPORT = r'''
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
'''


def main() -> None:
    patch_util()
    write_file("preprocess_png_to_pkl.py", PREPROCESS)
    write_file("basic_grad.py", BASIC_GRAD)
    write_file("wm_grad.py", WM_GRAD)
    write_file("export_wm_result_images.py", EXPORT)
    print("done")


if __name__ == "__main__":
    main()
