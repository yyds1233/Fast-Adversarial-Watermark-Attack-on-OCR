from tqdm import tqdm
import os
import time
import sys
import pickle

import tensorflow as tf
import sklearn  # 保留，避免你环境里有隐式依赖
from PIL import Image
import numpy as np
from cleverhans import utils_tf
from util import cvt2Image, sparse_tuple_from
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel
from calamari_ocr.ocr import Predictor

from skimage import morphology
import cv2
from trdg.generators import GeneratorFromStrings


checkpoint = '/app/ocr_model/4.ckpt.json'
predictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)

network = predictor.network
sess, graph = network.session, network.graph
codec = network.codec
charset = codec.charset
encode, decode = codec.encode, codec.decode
code2char, char2code = codec.code2char, codec.char2code


def invert(data):
    if data.max() < 1.5:
        return 1 - data
    else:
        return 255 - data


def transpose(data):
    if len(data.shape) != 2:
        return np.swapaxes(data, 1, 2)
    else:
        return data.T


def cvt2raw(data):
    return transpose(invert(data))


def show(img):
    return cvt2Image(cvt2raw(img))


# build graph
with graph.as_default():
    inputs, input_seq_len, targets, dropout_rate, _, _ = network.create_placeholders()

    # 新版 Calamari 返回 8 个值
    output_seq_len, \
    time_major_logits, \
    time_major_softmax, \
    logits, \
    softmax, \
    sparse_decoded, \
    decoded_tuple, \
    other = network.create_network(
        inputs, input_seq_len, dropout_rate, reuse_variables=tf.AUTO_REUSE
    )

    # 兼容旧代码
    decoded = sparse_decoded

    loss = tf.nn.ctc_loss(
        labels=targets,
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

    m = tf.placeholder(
        tf.float32,
        shape=inputs.get_shape().as_list(),
        name="momentum"
    )
    acc_m = m + norm_grad

    mask = tf.placeholder(
        tf.float32,
        shape=inputs.get_shape().as_list(),
        name="mask"
    )
    grad = tf.multiply(acc_m, mask, name="mask_op")

    # ord = inf
    optimal_perturbation = tf.sign(grad)
    optimal_perturbation = tf.stop_gradient(optimal_perturbation)
    scaled_perturbation_inf = utils_tf.mul(0.01, optimal_perturbation)

    # ord = 1
    abs_grad = tf.abs(grad)
    max_abs_grad = tf.reduce_max(abs_grad, axis=red_ind, keepdims=True)
    tied_for_max = tf.cast(tf.equal(abs_grad, max_abs_grad), tf.float32)
    num_ties = tf.reduce_sum(tied_for_max, axis=red_ind, keepdims=True)
    optimal_perturbation = tf.sign(grad) * tied_for_max / num_ties
    scaled_perturbation_1 = utils_tf.mul(0.01, optimal_perturbation)

    # ord = 2
    square = tf.maximum(
        1e-12, tf.reduce_sum(tf.square(grad), axis=red_ind, keepdims=True)
    )
    optimal_perturbation = grad / tf.sqrt(square)
    scaled_perturbation_2 = utils_tf.mul(0.01, optimal_perturbation)


# load args
font_name, case, pert_type, eps, eps_iter, nb_iter = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    float(sys.argv[4]),
    float(sys.argv[5]),
    int(sys.argv[6]),
)

# load original image data
img_data_path = '/app/img_data'
with open(f'{img_data_path}/{font_name}.pkl', 'rb') as f:
    input_img, len_x, gt_txt = pickle.load(f)
input_img = np.asarray(input_img)

# load basic attack result
title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}"
with open(f'attack_result/{title}.pkl', 'rb') as f:
    adv_img, record_adv_text, record_iter, (duration, total_iter) = pickle.load(f)

# 对齐样本数，basic_grad.py 一般只保存前 n_img 个
n_img = len(adv_img)
input_img = input_img[:n_img]
len_x = len_x[:n_img]
gt_txt = gt_txt[:n_img]

# load target text
attack_pair_path = '/app/attack_pair'
with open(f'{attack_pair_path}/{font_name}-{case}.pkl', 'rb') as f:
    _, target_txt = pickle.load(f)
target_txt = target_txt[:n_img]


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
            (remove * 255).astype('uint8'),
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        wm_pos, frame_img = [], []
        for cont in contours:
            left_point = cont.min(axis=1).min(axis=0)
            right_point = cont.max(axis=1).max(axis=0)
            wm_pos.append(np.hstack((left_point, right_point)))

            if ret_frame_img:
                img = cv2.rectangle(
                    (remove * 255).astype('uint8').copy(),
                    (left_point[0], left_point[1]),
                    (right_point[0], right_point[1]),
                    (255, 255, 255),
                    2
                )
                frame_img.append(img)

        wm_pos_list.append(wm_pos)
        frame_img_list.append(frame_img)

    if ret_frame_img:
        return wm_pos_list, frame_img_list
    else:
        return wm_pos_list


pos, frames = find_wm_pos(adv_img, input_img, True)

# 按面积从大到小排序
new_pos = []
for _pos in pos:
    if len(_pos) > 1:
        new_pos.append(
            sorted(
                _pos,
                key=lambda x: (x[3] - x[1]) * (x[2] - x[0]),
                reverse=True
            )
        )
    else:
        new_pos.append(_pos)
pos = new_pos


def RGB2Hex(RGB):
    color = '#'
    for num in RGB:
        color += str(hex(num))[-2:].replace('x', '0').upper()
    return color


def gen_wm(RGB):
    generator = GeneratorFromStrings(
        strings=['eccv'],
        count=1,
        # fonts=['Impact.ttf'],
        fonts=['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'],
        language='en',
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
        text_color=RGB2Hex(RGB),
        orientation=0,
        space_width=1.0,
        character_spacing=0,
        margins=(0, 0, 0, 0),
        fit=True,
    )
    img_list = [img for img, _ in generator]
    return img_list[0]


# 得到水印 mask
grayscale = 0
color = (grayscale, grayscale, grayscale)
wm_img = gen_wm(color)

wm_arr = np.array(wm_img.convert('L'))
kernel = np.ones((5, 5), np.uint8)
wm_arr = cv2.dilate(wm_arr, kernel, 2)
wm_arr = cv2.erode(wm_arr, kernel, 2)
bg_mask = ~(wm_arr != 255)

# 灰色水印
grayscale = 174
color = (grayscale, grayscale, grayscale)
wm_img = np.array(Image.new(mode="RGB", size=wm_img.size, color=color))
wm_img[bg_mask] = 255
wm_img = Image.fromarray(wm_img)


def get_text_mask(img: np.array):
    if img.max() <= 1:
        return img < 1 / 1.25
    else:
        return img < 255 / 1.25


wm0_img_list = []
wm_mask_list = []
text_mask_list = []

for i in range(len(input_img)):
    text_img = show(input_img[i])
    text_mask = get_text_mask(np.array(text_img))

    rgb_img = Image.new(mode="RGB", size=text_img.size, color=(255, 255, 255))

    p = -int(wm_img.size[0] * np.tan(10 * np.pi / 180))
    right_shift = 10
    xp = pos[i][0][0] + right_shift if len(pos[i]) != 0 else right_shift

    rgb_img.paste(wm_img, box=(xp, p))
    wm_mask = (np.array(rgb_img.convert('L')) != 255)
    rgb_img.paste(text_img, mask=cvt2Image(text_mask))

    wm0_img_list.append(rgb_img)
    wm_mask_list.append(transpose(wm_mask))
    text_mask_list.append(transpose(text_mask))

wm_mask = np.asarray(wm_mask_list)
text_mask = np.asarray(text_mask_list)


batch_size = 100
clip_min, clip_max = 0.0, 1.0

# 识别带初始水印的文本
record_text = []
wm0_img = pred_img = np.asarray(
    [cvt2raw(np.array(img.convert('L'))) / 255 for img in wm0_img_list]
)

batch_iter = len(input_img) // batch_size
batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1

for batch_i in range(batch_iter):
    start_idx = batch_size * batch_i
    end_idx = batch_size * (batch_i + 1)

    batch_img = pred_img[start_idx:end_idx]
    batch_len_x = len_x[start_idx:end_idx]

    batch_text = sess.run(
        decoded,
        feed_dict={
            inputs: batch_img,
            input_seq_len: batch_len_x,
            dropout_rate: 0,
        }
    )
    batch_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_text)
    record_text += [''.join(decode(index)) for index in batch_index]

cnt = 0
for pred_txt, raw_txt in zip(record_text, gt_txt):
    if pred_txt == raw_txt:
        cnt += 1

accuracy = cnt / len(gt_txt)

# run watermark attack
target_index_list = [np.asarray([c for c in encode(t)]) for t in target_txt]
wm_img = wm0_img

with graph.as_default():
    adv_img = wm_img.copy()
    m0 = np.zeros(input_img.shape)
    record_iter = np.zeros(input_img.shape[0])
    record_mse = []
    record_mse_plus = []

    start = time.time()

    for i in tqdm(range(nb_iter)):
        batch_iter = len(input_img) // batch_size
        batch_iter = batch_iter if len(input_img) % batch_size == 0 else batch_iter + 1

        for batch_i in range(batch_iter):
            start_idx = batch_size * batch_i
            end_idx = batch_size * (batch_i + 1)

            batch_input_img = wm_img[start_idx:end_idx]
            batch_adv_img = adv_img[start_idx:end_idx]
            batch_len_x = len_x[start_idx:end_idx]
            batch_m0 = m0[start_idx:end_idx]
            batch_target_txt = target_txt[start_idx:end_idx]
            batch_mask = wm_mask[start_idx:end_idx]
            batch_record_iter = record_iter[start_idx:end_idx]

            batch_tmp_y = [np.asarray([c - 1 for c in encode(t)]) for t in batch_target_txt]
            batch_y = sparse_tuple_from(batch_tmp_y)

            scaled_perturbation = scaled_perturbation_2 if pert_type == '2' else scaled_perturbation_inf

            batch_pert = sess.run(
                scaled_perturbation,
                feed_dict={
                    inputs: batch_adv_img,
                    input_seq_len: batch_len_x,
                    m: batch_m0,
                    targets: batch_y,
                    mask: batch_mask,
                    dropout_rate: 0,
                }
            )

            batch_pert[batch_record_iter != 0] = 0
            batch_adv_img = batch_adv_img + eps_iter * batch_pert * (batch_pert > 0)
            batch_adv_img = batch_input_img + np.clip(batch_adv_img - batch_input_img, -eps, eps)
            batch_adv_img = np.clip(batch_adv_img, clip_min, clip_max)

            adv_img[start_idx:end_idx] = batch_adv_img

        record_mse.append(np.mean(((adv_img - wm_img) * 255) ** 2))
        record_mse_plus.append(
            np.mean((((adv_img - wm_img) * ((adv_img - wm_img) > 0)) * 255) ** 2)
        )

        record_adv_text = []
        for batch_i in range(batch_iter):
            start_idx = batch_size * batch_i
            end_idx = batch_size * (batch_i + 1)

            batch_adv_img = adv_img[start_idx:end_idx]
            batch_len_x = len_x[start_idx:end_idx]
            batch_target_index = target_index_list[start_idx:end_idx]

            batch_adv_text = sess.run(
                decoded,
                feed_dict={
                    inputs: batch_adv_img,
                    input_seq_len: batch_len_x,
                    dropout_rate: 0,
                }
            )

            batch_adv_index = TensorflowModel._TensorflowModel__sparse_to_lists(batch_adv_text)
            record_adv_text += [''.join(decode(index)) for index in batch_adv_index]

            for j in range(len(batch_target_index)):
                adv_index, target_index = batch_adv_index[j], batch_target_index[j]
                idx_j = start_idx + j
                if np.sum(adv_index != target_index) == 0 and record_iter[idx_j] == 0:
                    record_iter[idx_j] = i

        if np.sum(record_iter == 0) == 0:
            break

    duration = time.time() - start
    print(f"{i} break. Time cost {duration:.4f} s")


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


rgb_img = cvt2rgb(adv_img, text_mask)

os.makedirs("wm_result", exist_ok=True)

title = f"{font_name}-{case}-l{pert_type}-eps{eps}-ieps{eps_iter}-iter{nb_iter}-positive"
with open(f'wm_result/{title}.pkl', 'wb') as f:
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
            (duration, i),
            rgb_img
        ),
        f
    )