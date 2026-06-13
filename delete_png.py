#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path


IMG_RE = re.compile(r"^img_(\d+)\.png$")


def main():
    parser = argparse.ArgumentParser(
        description="Delete img_XXXX.png files whose index is >= keep_count."
    )
    parser.add_argument(
        "png_dir",
        type=str,
        help="Directory containing img_0000.png, img_0001.png, ..."
    )
    parser.add_argument(
        "--keep_count",
        type=int,
        default=300,
        help="Number of images to keep. Default: 300, keeps img_0000.png to img_0299.png."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print files that would be deleted, do not actually delete."
    )

    args = parser.parse_args()

    png_dir = Path(args.png_dir)

    if not png_dir.exists():
        raise FileNotFoundError(f"Directory not found: {png_dir}")

    if not png_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {png_dir}")

    delete_files = []

    for path in sorted(png_dir.iterdir()):
        if not path.is_file():
            continue

        match = IMG_RE.match(path.name)
        if not match:
            continue

        idx = int(match.group(1))

        if idx >= args.keep_count:
            delete_files.append(path)

    print(f"Directory: {png_dir}")
    print(f"Keep count: {args.keep_count}")
    print(f"Files to delete: {len(delete_files)}")

    for path in delete_files:
        print(f"delete: {path}")
        if not args.dry_run:
            path.unlink()

    if args.dry_run:
        print("Dry run only. No files were deleted.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()