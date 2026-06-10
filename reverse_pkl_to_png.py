# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from pathlib import Path
from PIL import Image
import numpy as np


def invert(data):
    if data.max() <= 1.0:
        return 1.0 - data
    return 255 - data


def transpose(data):
    if data.ndim == 2:
        return data.T
    return np.swapaxes(data, 1, 2)


def cvt2raw(data):
    return transpose(invert(data))


def array_to_image(array):
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return Image.fromarray(array, mode='L')
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode='RGB')
    raise ValueError(f'Unsupported image array shape: {array.shape}')


def save_png_images(img_array, out_dir, prefix='img'):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = img_array.shape[0]
    for i in range(n):
        arr = cvt2raw(img_array[i])
        img = array_to_image(arr)
        img.save(out_dir / f'{prefix}_{i:04d}.png')


def write_text_list(lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f'{line}\n')


def main():
    parser = argparse.ArgumentParser(description='Reverse img_data and attack_pair pickle files to PNG + text files.')
    parser.add_argument('--font_name', required=True, help='Font name, e.g. Arial')
    parser.add_argument('--case', required=True, help='Attack case, e.g. easy')
    parser.add_argument('--img_data_dir', default='img_data', help='Directory containing img_data pickle files')
    parser.add_argument('--attack_pair_dir', default='attack_pair', help='Directory containing attack_pair pickle files')
    parser.add_argument('--output_dir', default='recovered_input', help='Output root directory')
    parser.add_argument('--png_dir_name', default='png_dir', help='Subdirectory name for recovered PNG images')
    parser.add_argument('--png_prefix', default='img', help='Prefix for generated PNG filenames')
    args = parser.parse_args()

    img_data_path = Path(args.img_data_dir) / f'{args.font_name}.pkl'
    attack_pair_path = Path(args.attack_pair_dir) / f'{args.font_name}-{args.case}.pkl'

    if not img_data_path.exists():
        raise FileNotFoundError(f'img_data pickle not found: {img_data_path}')
    if not attack_pair_path.exists():
        raise FileNotFoundError(f'attack_pair pickle not found: {attack_pair_path}')

    with open(img_data_path, 'rb') as f:
        input_img, len_x, gt_txt = pickle.load(f)

    with open(attack_pair_path, 'rb') as f:
        attack_gt_txt, target_txt = pickle.load(f)

    if len(gt_txt) != len(input_img):
        raise ValueError('Length mismatch: img_data gt_txt and input_img count differ')
    if len(attack_gt_txt) != len(target_txt):
        raise ValueError('Length mismatch: attack_pair contents differ')
    if len(attack_gt_txt) != len(input_img):
        raise ValueError('Length mismatch: attack_pair count and img_data count differ')

    out_root = Path(args.output_dir)
    png_dir = out_root / args.png_dir_name
    png_dir.mkdir(parents=True, exist_ok=True)

    print(f'Writing {len(input_img)} PNG images to {png_dir}')
    save_png_images(input_img, png_dir, prefix=args.png_prefix)

    gt_txt_path = out_root / 'gt.txt'
    target_txt_path = out_root / 'target.txt'

    print(f'Writing ground truth to {gt_txt_path}')
    write_text_list(gt_txt, gt_txt_path)

    print(f'Writing target text to {target_txt_path}')
    write_text_list(target_txt, target_txt_path)

    print('Done.')
    print(f'Output directory structure:')
    print(f'  {out_root}/{args.png_dir_name}/')
    print(f'  {out_root}/gt.txt')
    print(f'  {out_root}/target.txt')


if __name__ == '__main__':
    main()
