# -*- coding: utf-8 -*-
import os
import argparse
import pickle
from glob import glob
from PIL import Image
import numpy as np


def normalize_image(img, target_height):
    """Convert a PIL image to a normalized grayscale array of shape (width, height)."""
    if img.mode != "L":
        img = img.convert("L")
    w, h = img.size
    if h != target_height:
        new_w = int(round(w * target_height / float(h)))
        img = img.resize((new_w, target_height), Image.LANCZOS)
        w, h = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # In this repository, stored arrays use text=1.0 and background=0.0.
    # If the image is dark text on light background, invert it.
    if arr.mean() > 0.5:
        arr = 1.0 - arr
    return arr.T, w


def load_text_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


def build_img_data(png_paths, target_height, pad_width=None):
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
    parser = argparse.ArgumentParser(description='Preprocess PNGs into img_data.pkl and attack_pair.pkl')
    parser.add_argument('--png_dir', type=str, required=True,
                        help='Directory containing input PNG files')
    parser.add_argument('--gt_txt', type=str, required=True,
                        help='Path to ground truth text file, one line per PNG in sorted order')
    parser.add_argument('--target_txt', type=str, required=True,
                        help='Path to target text file, one line per PNG in sorted order')
    parser.add_argument('--font_name', type=str, required=True,
                        help='Font name used for img_data/{font_name}.pkl')
    parser.add_argument('--case', type=str, required=True,
                        help='Case name used for attack_pair/{font_name}-{case}.pkl')
    parser.add_argument('--height', type=int, default=48,
                        help='Target image height after resize')
    parser.add_argument('--pad_width', type=int, default=None,
                        help='Optional width to pad all images to. Default=max image width')
    parser.add_argument('--output_img_data', type=str, default='img_data',
                        help='Output directory for img_data pickle')
    parser.add_argument('--output_attack_pair', type=str, default='attack_pair',
                        help='Output directory for attack_pair pickle')
    parser.add_argument('--ext', type=str, default='png',
                        help='Image file extension to load from png_dir')
    args = parser.parse_args()

    png_paths = sorted(glob(os.path.join(args.png_dir, f'*.{args.ext}')))
    if len(png_paths) == 0:
        raise FileNotFoundError(f'No .{args.ext} files found in {args.png_dir}')

    gt_txt = load_text_lines(args.gt_txt)
    target_txt = load_text_lines(args.target_txt)

    if len(gt_txt) != len(png_paths) or len(target_txt) != len(png_paths):
        raise ValueError('The number of PNG files and the number of text lines must match.')

    input_img, len_x = build_img_data(png_paths, args.height, args.pad_width)

    os.makedirs(args.output_img_data, exist_ok=True)
    os.makedirs(args.output_attack_pair, exist_ok=True)

    img_data_path = os.path.join(args.output_img_data, f'{args.font_name}.pkl')
    attack_pair_path = os.path.join(args.output_attack_pair, f'{args.font_name}-{args.case}.pkl')

    with open(img_data_path, 'wb') as f:
        pickle.dump((input_img, len_x, gt_txt), f)
    with open(attack_pair_path, 'wb') as f:
        pickle.dump((gt_txt, target_txt), f)

    print(f'Wrote {img_data_path}  shape={input_img.shape}  count={len(input_img)}')
    print(f'Wrote {attack_pair_path}  count={len(gt_txt)}')


if __name__ == '__main__':
    main()
