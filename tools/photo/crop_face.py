"""Detect the largest face in an image and crop a square portrait around it."""
import argparse
import sys
from pathlib import Path

import cv2
from PIL import Image


def detect_largest_face(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def crop_square_portrait(image_bgr, face_bbox, padding=2.4, face_y=0.40):
    h, w = image_bgr.shape[:2]
    fx, fy, fw, fh = [int(v) for v in face_bbox]
    side = int(fh * padding)
    cx = fx + fw // 2
    cy = fy + fh // 2
    x = cx - side // 2
    y = cy - int(side * face_y)
    if side > w:
        side = w
    if side > h:
        side = h
    x = max(0, min(x, w - side))
    y = max(0, min(y, h - side))
    return image_bgr[y:y + side, x:x + side]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Source image path")
    p.add_argument("output", help="Destination .jpg or .png path")
    p.add_argument("--size", type=int, default=400, help="Output square size in px")
    p.add_argument("--padding", type=float, default=2.4,
                   help="Crop side as multiple of face height")
    p.add_argument("--face-y", type=float, default=0.40,
                   help="Face center vertical position 0-1 from top")
    p.add_argument("--quality", type=int, default=88, help="JPEG quality 1-100")
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"FAIL: cannot load {args.input}", file=sys.stderr)
        sys.exit(1)

    face = detect_largest_face(img)
    if face is None:
        print("FAIL: no face detected", file=sys.stderr)
        sys.exit(2)

    cropped = crop_square_portrait(img, face, args.padding, args.face_y)
    pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).resize(
        (args.size, args.size), Image.LANCZOS
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".png":
        pil.save(out, "PNG", optimize=True)
    else:
        pil.save(out, "JPEG", quality=args.quality, optimize=True)
    print(f"saved {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
