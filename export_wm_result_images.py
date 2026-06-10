# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from glob import glob
from pathlib import Path
from PIL import Image
import numpy as np


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
        return Image.fromarray(array, mode='L')
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode='RGB')
    raise ValueError(f'Unsupported image array shape: {array.shape}')


def save_grayscale_images(arr, out_dir, prefix):
    for i in range(arr.shape[0]):
        img = array_to_image(cvt2raw(arr[i]))
        img.save(out_dir / f'{prefix}_{i:04d}.png')


def save_rgb_images(arr, out_dir, prefix):
    for i in range(arr.shape[0]):
        img = array_to_image(arr[i])
        img.save(out_dir / f'{prefix}_{i:04d}.png')


def process_pkl(path, out_root, save_adv, save_rgb, save_wm0):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    out_root.mkdir(parents=True, exist_ok=True)
    base = Path(path).stem

    if save_adv and len(data) > 6:
        adv_img = data[6]
        if isinstance(adv_img, np.ndarray) and adv_img.ndim == 3:
            save_dir = out_root / f'{base}_adv'
            save_dir.mkdir(exist_ok=True)
            save_grayscale_images(adv_img, save_dir, 'adv')
            print(f'Saved {adv_img.shape[0]} adv images to {save_dir}')

    if save_wm0 and len(data) > 3:
        wm0_img = data[3]
        if isinstance(wm0_img, np.ndarray) and wm0_img.ndim == 3:
            save_dir = out_root / f'{base}_wm0'
            save_dir.mkdir(exist_ok=True)
            save_grayscale_images(wm0_img, save_dir, 'wm0')
            print(f'Saved {wm0_img.shape[0]} wm0 images to {save_dir}')

    if save_rgb and len(data) > 10:
        rgb_img = data[10]
        if isinstance(rgb_img, np.ndarray) and rgb_img.ndim == 4 and rgb_img.shape[3] == 3:
            save_dir = out_root / f'{base}_rgb'
            save_dir.mkdir(exist_ok=True)
            save_rgb_images(rgb_img, save_dir, 'rgb')
            print(f'Saved {rgb_img.shape[0]} rgb images to {save_dir}')


def main():
    parser = argparse.ArgumentParser(description='Export images from wm_result pickle files.')
    parser.add_argument('--input', type=str, default='wm_result',
                        help='Path to a wm_result pickle file or a directory containing wm_result pickles.')
    parser.add_argument('--output', type=str, default='exported_wm_images',
                        help='Output directory for PNG images.')
    parser.add_argument('--save_adv', action='store_true', help='Save adv_img images.')
    parser.add_argument('--save_rgb', action='store_true', help='Save rgb_img images.')
    parser.add_argument('--save_wm0', action='store_true', help='Save wm0_img images.')
    parser.add_argument('--pattern', type=str, default='*.pkl', help='Glob pattern when input is a directory.')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        paths = sorted(input_path.glob(args.pattern))
    else:
        paths = [input_path]

    if not paths:
        raise FileNotFoundError(f'No pickle files found in {input_path}')

    for p in paths:
        print(f'Processing {p}')
        process_pkl(p, output_path / p.stem, args.save_adv, args.save_rgb, args.save_wm0)

if __name__ == '__main__':
    main()
