# -*- coding: utf-8 -*-
# @Time    : 13/1/20 15:31
# @Author  : Lu Chen (modified to use latest Calamari API)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import sklearn
from PIL import Image
import numpy as np
import pickle, glob, time, sys, os
from tqdm import tqdm
from cleverhans import utils_tf

from util import get_argparse, cvt2Image, sparse_tuple_from

# 最新 Calamari Predictor API
from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams

# parse parameters
parser = get_argparse()
args = parser.parse_args()

# 初始化 Calamari Predictor
predictor = Predictor.from_checkpoint(
    params=PredictorParams(),
    checkpoint=os.path.join("ocr_model", args.model_path)
)

print(f"[INFO] Loaded Calamari OCR model from {args.model_path}")

# build adversarial graph
with tf.get_default_graph().as_default():
    inputs = tf.placeholder(tf.float32, shape=[None, None, None])
    input_seq_len = tf.placeholder(tf.int32, shape=[None])
    
    # Modified to accept sparse tensor for CTC loss
    target_indices = tf.placeholder(tf.int64, shape=[None, 2])  # indices of sparse tensor
    target_values = tf.placeholder(tf.int32, shape=[None])       # values of sparse tensor
    target_shape = tf.placeholder(tf.int64, shape=[2])           # shape of sparse tensor
    
    dropout_rate = tf.placeholder(tf.float32)

    # CTC loss and gradient graph
    output_seq_len = input_seq_len
    time_major_logits = tf.transpose(inputs, perm=[1, 0, 2])

    # Create SparseTensor for CTC loss
    labels_sparse = tf.SparseTensor(
        indices=target_indices,
        values=target_values,
        dense_shape=target_shape
    )

    loss = tf.nn.ctc_loss(
        labels=labels_sparse,
        inputs=time_major_logits,
        sequence_length=output_seq_len,
        time_major=True,
        ctc_merge_repeated=True,
        ignore_longer_outputs_than_inputs=True
    )
    loss = -tf.reduce_mean(loss, name='loss')

    grad, = tf.gradients(loss, inputs)

    red_ind = list(range(1, len(grad.get_shape())))
    avoid_zero_div = tf.cast(1e-12, grad.dtype)
    divisor = tf.reduce_mean(tf.abs(grad), red_ind, keepdims=True)
    norm_grad = grad / tf.maximum(avoid_zero_div, divisor)

    m = tf.placeholder(tf.float32, shape=inputs.get_shape().as_list(), name="momentum")
    acc_m = m + norm_grad
    grad_norm = acc_m
    optimal_perturbation = tf.sign(grad_norm)
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)
    square = tf.maximum(1e-12, tf.reduce_sum(tf.square(grad_norm), axis=red_ind, keepdims=True))
    optimal_perturbation2 = grad_norm / tf.sqrt(square)
    scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation2)

# parameters
font_name = args.font_name
case = args.case
pert_type = args.pert_type
eps = args.eps
eps_iter = args.eps_iter
nb_iter = args.nb_iter
batch_size = args.batch_size
clip_min, clip_max = args.clip_min, args.clip_max

# load img data
with open(f"img_data/{font_name}.pkl", "rb") as f:
    input_img, len_x, gt_txt = pickle.load(f)

# load attack pair
with open(f"attack_pair/{font_name}-{case}.pkl", "rb") as f:
    _, target_txt = pickle.load(f)

n_img = min(len(input_img), 200)
input_img = input_img[:n_img]
len_x = len_x[:n_img]
gt_txt = gt_txt[:n_img]
target_txt = target_txt[:n_img]

adv_img = input_img.copy()
m0 = np.zeros(input_img.shape)
record_iter = np.zeros(input_img.shape[0])
record_adv_text = []

batch_iter = len(input_img) // batch_size
batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1

sess = tf.get_default_session() or tf.Session()
start = time.time()

for batch_i in tqdm(range(batch_iter)):
    batch_input = input_img[batch_i * batch_size:(batch_i + 1) * batch_size]
    batch_adv = adv_img[batch_i * batch_size:(batch_i + 1) * batch_size]
    batch_len = len_x[batch_i * batch_size:(batch_i + 1) * batch_size]
    batch_m0 = m0[batch_i * batch_size:(batch_i + 1) * batch_size]
    batch_target = target_txt[batch_i * batch_size:(batch_i + 1) * batch_size]
    first_pred = next(predictor.predict_raw([input_img[0]]))
    codec = first_pred.outputs.codec

    # Prepare sparse tuple
    batch_target_index = [
        # np.asarray(predictor.params.model.codec.encode(t), dtype=np.int32) for t in batch_target
        np.asarray(codec.encode(t), dtype=np.int32) for t in batch_target
    ]
    batch_y = sparse_tuple_from(batch_target_index)

    # Assign perturbation type
    scaled_perturbation = scaled_perturbation_2 if pert_type == "2" else scaled_perturbation_inf

    for i in range(nb_iter):
        pert_val, = sess.run(
            [scaled_perturbation],
            feed_dict={
                inputs: batch_adv,
                input_seq_len: batch_len,
                m: batch_m0,
                target_indices: batch_y[0],  # Pass the sparse tensor parts
                target_values: batch_y[1],
                target_shape: batch_y[2],
                dropout_rate: 0,
            }
        )

        pert_val[record_iter[batch_i * batch_size:(batch_i + 1) * batch_size] != 0] = 0
        batch_adv = batch_adv + eps_iter * pert_val
        batch_adv = batch_input + np.clip(batch_adv - batch_input, -eps, eps)
        batch_adv = np.clip(batch_adv, clip_min, clip_max)

        adv_img[batch_i * batch_size:(batch_i + 1) * batch_size] = batch_adv

        # ===== use Calamari Predictor =====
        preds = list(predictor.predict_raw(batch_adv))
        batch_pred_text = [p.outputs.sentence for p in preds]  # actual recognized text

        for j, pred_txt in enumerate(batch_pred_text):
            if pred_txt == batch_target[j] and record_iter[batch_i * batch_size + j] == 0:
                record_iter[batch_i * batch_size + j] = i

        if np.all(record_iter[batch_i * batch_size:(batch_i + 1) * batch_size] != 0):
            print(f"Batch {batch_i} stopped at iteration {i}")
            break

    record_adv_text += batch_pred_text

duration = time.time() - start

title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
os.makedirs("attack_result", exist_ok=True)
with open(f"attack_result/{title}.pkl", "wb") as f:
    pickle.dump((adv_img, record_adv_text, record_iter, (duration, nb_iter)), f)

print("[INFO] Attack finished!")