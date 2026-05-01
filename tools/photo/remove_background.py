"""Remove the background from an image using rembg, output transparent PNG."""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove, new_session

MODELS = [
    "u2net",
    "u2net_human_seg",
    "isnet-general-use",
    "birefnet-portrait",
    "birefnet-general",
]


def threshold_alpha(rgba_image):
    arr = np.array(rgba_image)
    alpha = arr[:, :, 3]
    hard = np.where(alpha < 128, 0, 255).astype(np.uint8)
    soft = np.where(alpha >= 200, alpha, hard).astype(np.uint8)
    arr[:, :, 3] = soft
    return Image.fromarray(arr, mode="RGBA")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Source image path")
    p.add_argument("output", help="Destination .png path")
    p.add_argument("--model", default="birefnet-portrait", choices=MODELS,
                   help="rembg segmentation model")
    p.add_argument("--threshold", action="store_true",
                   help="Apply hard alpha threshold to clean haze artifacts")
    p.add_argument("--size", type=int, default=None,
                   help="Optional output square size in px")
    args = p.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(f"FAIL: cannot find {src}", file=sys.stderr)
        sys.exit(1)

    pil_in = Image.open(src).convert("RGB")
    session = new_session(args.model)
    out_img = remove(pil_in, session=session)

    if args.threshold:
        out_img = threshold_alpha(out_img)

    if args.size:
        out_img = out_img.resize((args.size, args.size), Image.LANCZOS)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out, "PNG", optimize=True)
    print(f"saved {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
