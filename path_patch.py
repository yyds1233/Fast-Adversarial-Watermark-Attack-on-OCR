#!/usr/bin/env python3
"""Patch FAWA OCR attack scripts so --model_path is optional.

After this patch, both basic_grad.py and wm_grad.py resolve the Calamari
checkpoint automatically from:
    /app/weight/<mission_id>/*.json

Expected layout:
    /app/weight/<mission_id>/<only-one-json-file>.json
    /app/weight/<mission_id>/<corresponding-weight-file>.h5

Run from the repository root, e.g.:
    cd /app
    python apply_auto_model_path_patch.py
"""

from __future__ import annotations

import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path.cwd()
BACKUP_SUFFIX = ".bak_auto_model_path"

UTIL_HELPER = r'''


def resolve_mission_checkpoint(mission_id, model_path=None, weight_root="/app/weight"):
    """Resolve Calamari checkpoint json for a mission.

    If model_path is given, it is interpreted as an absolute path when absolute,
    otherwise relative to /app/weight/<mission_id>.

    If model_path is omitted, exactly one *.json file must exist under
    /app/weight/<mission_id>.
    """
    import glob
    import os

    if not mission_id:
        raise ValueError("mission_id is required to resolve checkpoint")

    weight_dir = os.path.join(weight_root, str(mission_id))
    if not os.path.isdir(weight_dir):
        raise FileNotFoundError(
            "Weight directory not found: {}. Expected: /app/weight/<mission_id>/".format(weight_dir)
        )

    if model_path:
        checkpoint = model_path if os.path.isabs(model_path) else os.path.join(weight_dir, model_path)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("Checkpoint json not found: {}".format(checkpoint))
        return checkpoint

    json_files = sorted(glob.glob(os.path.join(weight_dir, "*.json")))
    if len(json_files) == 0:
        raise FileNotFoundError(
            "No checkpoint json found in {}. Put exactly one *.json file there.".format(weight_dir)
        )
    if len(json_files) > 1:
        raise RuntimeError(
            "Multiple checkpoint json files found in {}: {}. Keep only one or pass --model_path.".format(
                weight_dir, json_files
            )
        )

    h5_files = sorted(glob.glob(os.path.join(weight_dir, "*.h5")))
    if len(h5_files) == 0:
        raise FileNotFoundError(
            "No .h5 weight file found in {}. The json checkpoint usually needs its corresponding .h5 file.".format(
                weight_dir
            )
        )

    return json_files[0]
'''


def read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: pathlib.Path, text: str) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")


def patch_util() -> bool:
    path = ROOT / "util.py"
    text = read(path)
    original = text

    if "def resolve_mission_checkpoint(" not in text:
        text = text.rstrip() + UTIL_HELPER + "\n"

    if "--mission_id" not in text:
        # Insert mission_id before model_path in get_argparse.
        text = text.replace(
            'parser.add_argument("--model_path", help="Calamari-OCR model path.", type=str)',
            'parser.add_argument("--mission_id", help="mission id. Weight is read from /app/weight/<mission_id> by default.", type=str, required=True)\n    parser.add_argument("--model_path", help="Optional Calamari-OCR checkpoint json. If omitted, auto-detect the only *.json under /app/weight/<mission_id>.", type=str, default=None)',
        )
        text = text.replace(
            "parser.add_argument('--model_path', help='Calamari-OCR model path.', type=str)",
            "parser.add_argument('--mission_id', help='mission id. Weight is read from /app/weight/<mission_id> by default.', type=str, required=True)\n    parser.add_argument('--model_path', help='Optional Calamari-OCR checkpoint json. If omitted, auto-detect the only *.json under /app/weight/<mission_id>.', type=str, default=None)",
        )

    if text != original:
        write(path, text)
        return True
    return False


def patch_basic_grad() -> bool:
    path = ROOT / "basic_grad.py"
    text = read(path)
    original = text

    # Add import helper.
    text = text.replace(
        "from util import get_argparse, cvt2Image, sparse_tuple_from, resolve_mission_paths",
        "from util import get_argparse, cvt2Image, sparse_tuple_from, resolve_mission_paths, resolve_mission_checkpoint",
    )
    text = text.replace(
        "from util import get_argparse, cvt2Image, sparse_tuple_from",
        "from util import get_argparse, cvt2Image, sparse_tuple_from, resolve_mission_checkpoint",
    )

    # Replace Predictor initialization. This covers original and mission-aware variants.
    predictor_pattern = re.compile(
        r"predictor\s*=\s*Predictor\(\s*checkpoint\s*=\s*[^\n]+?,\s*batch_size\s*=\s*1,\s*processes\s*=\s*10\s*\)",
        re.S,
    )
    replacement = (
        'checkpoint = resolve_mission_checkpoint(args.mission_id, getattr(args, "model_path", None))\n'
        'print(f"[basic_grad] Using checkpoint: {checkpoint}")\n'
        'predictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)'
    )
    text, count = predictor_pattern.subn(replacement, text, count=1)

    if count == 0 and "Predictor(checkpoint=" in text and "resolve_mission_checkpoint" in text:
        print("[WARN] basic_grad.py: did not match Predictor line automatically; please patch manually.")

    if text != original:
        write(path, text)
        return True
    return False


def patch_wm_grad() -> bool:
    path = ROOT / "wm_grad.py"
    text = read(path)
    original = text

    # Add import helper.
    text = text.replace(
        "from util import cvt2Image, sparse_tuple_from, resolve_mission_paths",
        "from util import cvt2Image, sparse_tuple_from, resolve_mission_paths, resolve_mission_checkpoint",
    )
    text = text.replace(
        "from util import cvt2Image, sparse_tuple_from",
        "from util import cvt2Image, sparse_tuple_from, resolve_mission_checkpoint",
    )

    # Add argparse import if needed.
    if "import argparse" not in text:
        text = text.replace("import sys", "import sys\nimport argparse", 1)

    parse_block = r'''

def parse_wm_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("font_name", type=str, choices=["Courier", "Georgia", "Helvetica", "Times", "Arial"])
    parser.add_argument("case", type=str)
    parser.add_argument("pert_type", type=str, choices=["2", "inf"])
    parser.add_argument("eps", type=float)
    parser.add_argument("eps_iter", type=float)
    parser.add_argument("nb_iter", type=int)
    parser.add_argument("--mission_id", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional checkpoint json. If omitted, auto-detect the only *.json under /app/weight/<mission_id>.",
    )
    return parser.parse_args(argv)

args = parse_wm_args(sys.argv[1:])
mission_id = args.mission_id
checkpoint = resolve_mission_checkpoint(mission_id, args.model_path)
print(f"[wm_grad] Using checkpoint: {checkpoint}")
'''

    # Replace original hard-coded checkpoint block, if present.
    hardcoded_patterns = [
        "checkpoint = '/app/ocr_model/4.ckpt.json'\npredictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)",
        'checkpoint = "/app/ocr_model/4.ckpt.json"\npredictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)',
    ]
    replaced = False
    for pat in hardcoded_patterns:
        if pat in text:
            text = text.replace(pat, parse_block + "\npredictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)", 1)
            replaced = True
            break

    # If already has a generic checkpoint assignment but not parse_wm_args, insert before Predictor.
    if not replaced and "def parse_wm_args(" not in text:
        text = re.sub(
            r"checkpoint\s*=\s*[^\n]+\n\s*predictor\s*=\s*Predictor\(checkpoint=checkpoint, batch_size=1, processes=10\)",
            parse_block + "\npredictor = Predictor(checkpoint=checkpoint, batch_size=1, processes=10)",
            text,
            count=1,
        )

    # Replace the old sys.argv tuple parsing with args fields.
    old_args_pattern = re.compile(
        r"# load args\s*font_name,\s*case,\s*pert_type,\s*eps,\s*eps_iter,\s*nb_iter\s*=\s*\(\s*sys\.argv\[1\],\s*sys\.argv\[2\],\s*sys\.argv\[3\],\s*float\(sys\.argv\[4\]\),\s*float\(sys\.argv\[5\]\),\s*int\(sys\.argv\[6\]\),\s*\)",
        re.S,
    )
    text = old_args_pattern.sub(
        "# load args\nfont_name = args.font_name\ncase = args.case\npert_type = args.pert_type\neps = args.eps\neps_iter = args.eps_iter\nnb_iter = args.nb_iter",
        text,
        count=1,
    )

    if text != original:
        write(path, text)
        return True
    return False


def main() -> int:
    required = ["util.py", "basic_grad.py", "wm_grad.py"]
    missing = [name for name in required if not (ROOT / name).exists()]
    if missing:
        print("Missing files in current directory:", missing, file=sys.stderr)
        print("Run this script from the repository root, e.g. cd /app", file=sys.stderr)
        return 1

    changed = []
    for name, func in [
        ("util.py", patch_util),
        ("basic_grad.py", patch_basic_grad),
        ("wm_grad.py", patch_wm_grad),
    ]:
        try:
            if func():
                changed.append(name)
        except Exception as exc:
            print(f"[ERROR] Failed to patch {name}: {exc}", file=sys.stderr)
            return 1

    print("Changed files:", ", ".join(changed) if changed else "none")
    print("Backups use suffix:", BACKUP_SUFFIX)
    print("Now basic_grad.py can omit --model_path when /app/weight/<mission_id>/ contains exactly one *.json and at least one *.h5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
