# -*- coding: utf-8 -*-
# @Time : 13/1/20 15:31
# @Author : Lu Chen
"""Mission-aware basic gradient attack.

Reads:
    /app/img_data/<mission_id>-<font_name>.pkl
    /app/attack_pair/<mission_id>-<font_name>-<case>.pkl
    /app/weight/<mission_id>/*.json
    /app/weight/<mission_id>/*.h5

Writes:
    /app/attack_result/<mission_id>-<font_name>-<case>-l<pert_type>-eps<eps>-ieps<eps_iter>-iter<nb_iter>.pkl
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Optional

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
        raise ValueError(
            "mission_id only supports letters, digits, underscore, dot and hyphen"
        )
    return mission_id


def resolve_mission_checkpoint(
    app_root: Path,
    mission_id: str,
    model_dir: Optional[str] = None,
) -> Path:
    """
    Resolve checkpoint json from:
        /app/weight/<mission_id>/*.json

    Requirements:
        1. /app/weight/<mission_id>/ must exist.
        2. It must contain exactly one .json file.
        3. It must contain at least one .h5 file.
    """
    weight_dir = Path(model_dir) if model_dir else app_root / "weight" / mission_id

    if not weight_dir.exists():
        raise FileNotFoundError(f"Weight directory not found: {weight_dir}")

    if not weight_dir.is_dir():
        raise NotADirectoryError(f"Weight path is not a directory: {weight_dir}")

    json_files = sorted(weight_dir.glob("*.json"))
    h5_files = sorted(weight_dir.glob("*.h5"))

    if len(json_files) == 0:
        raise FileNotFoundError(
            f"No .json checkpoint file found in: {weight_dir}"
        )

    if len(json_files) > 1:
        raise RuntimeError(
            "Multiple .json checkpoint files found in "
            f"{weight_dir}: {[p.name for p in json_files]}. "
            "Please keep only one .json file in this mission weight directory."
        )

    if len(h5_files) == 0:
        raise FileNotFoundError(
            f"No .h5 weight file found in: {weight_dir}. "
            "Please put the corresponding .h5 file together with the .json checkpoint."
        )

    return json_files[0]


def result_title(
    mission_id: str,
    font_name: str,
    case: str,
    pert_type: str,
    eps: float,
    eps_iter: float,
    nb_iter: int,
) -> str:
    return (
        f"{mission_id}-{font_name}-{case}-l{pert_type}"
        f"-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
    )


def main():
    parser = get_argparse()
    args = parser.parse_args()

    mission_id = validate_mission_id(args.mission_id)
    app_root = Path(getattr(args, "app_root", "/app"))

    checkpoint = resolve_mission_checkpoint(
        app_root=app_root,
        mission_id=mission_id,
        model_dir=getattr(args, "model_dir", None),
    )

    print(f"[basic_grad] Using checkpoint: {checkpoint}")

    predictor = Predictor(
        checkpoint=str(checkpoint),
        batch_size=1,
        processes=10,
    )

    network = predictor.network
    sess, graph = network.session, network.graph
    encode, decode = network.codec.encode, network.codec.decode

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
        ) = network.create_network(
            inputs,
            input_seq_len,
            dropout_rate,
            reuse_variables=tf.AUTO_REUSE,
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

        m = tf.placeholder(
            tf.float32,
            shape=inputs.get_shape().as_list(),
            name="momentum",
        )

        acc_m = m + norm_grad
        grad = acc_m

        # ord = inf
        optimal_perturbation = tf.sign(grad)
        optimal_perturbation = tf.stop_gradient(optimal_perturbation)
        scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)

        # ord = 2
        square = tf.maximum(
            1e-12,
            tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True),
        )
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

    n_img = min(len(input_img), len(input_img), len(target_txt))
    input_img = input_img[:n_img]
    len_x = len_x[:n_img]
    gt_txt = gt_txt[:n_img]
    target_txt = target_txt[:n_img]

    with graph.as_default():
        adv_img = input_img.copy()
        m0 = np.zeros(input_img.shape)

        # 0 means unsuccessful
        record_iter = np.zeros(input_img.shape[0])
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

            batch_target_index = [
                np.asarray([c - 1 for c in encode(t)])
                for t in batch_target_text
            ]

            batch_y = sparse_tuple_from(batch_target_index)
            batch_record_iter = np.zeros(cur_batch_size)

            if pert_type == "2":
                scaled_perturbation = scaled_perturbation_2
            else:
                scaled_perturbation = scaled_perturbation_inf

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
                batch_adv_img = batch_input_img + np.clip(
                    batch_adv_img - batch_input_img,
                    -eps,
                    eps,
                )
                batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)

                adv_img[start_idx:end_idx] = batch_adv_img

                batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(
                    batch_adv_text_sparse
                )

                batch_adv_text = [
                    "".join(decode(index))
                    for index in batch_adv_index
                ]

                for j in range(cur_batch_size):
                    if (
                        batch_adv_text[j] == batch_target_text[j]
                        and batch_record_iter[j] == 0
                    ):
                        batch_record_iter[j] = i

                if np.sum(batch_record_iter == 0) == 0:
                    print(f"{i} break")
                    break

            record_iter[start_idx:end_idx] = batch_record_iter
            record_adv_text += batch_adv_text

        duration = time.time() - start

    title = result_title(
        mission_id=mission_id,
        font_name=font_name,
        case=case,
        pert_type=pert_type,
        eps=eps,
        eps_iter=eps_iter,
        nb_iter=nb_iter,
    )

    output_path = attack_result_dir / f"{title}.pkl"

    with output_path.open("wb") as f:
        pickle.dump(
            (
                adv_img,
                record_adv_text,
                record_iter,
                (duration, last_iter),
            ),
            f,
        )

    print(f"[basic_grad] Wrote {output_path}")


if __name__ == "__main__":
    main()