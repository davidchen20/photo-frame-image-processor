#!/usr/bin/env python3
"""
smart_crop.py

Crop an image to a target size/aspect ratio, centered on the most
"important" part of the image:
  1. If faces are detected (via OpenCV's FaceDetectorYN / YuNet DNN model),
     center on the bounding box of all faces.
  2. Otherwise, fall back to saliency detection to find the most
     visually interesting region and center on that.

The face detection model (~230KB ONNX file) is downloaded automatically
on first run and cached next to this script.

Usage:
    python smart_crop.py input.jpg output.jpg --width 800 --height 800
    python smart_crop.py input.jpg output.jpg --aspect 4:5
    python smart_crop.py input.jpg output.jpg --aspect 1:1 --scale 0.9
"""

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np

# YuNet face detector model (ONNX). Dynamic-input-shape build, required for
# OpenCV 5.x's ONNX Runtime DNN engine. Cached next to this script after the
# first run so it's only downloaded once.
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2026may.onnx"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_detection_yunet.onnx")


def ensure_model():
    """Download the YuNet ONNX model on first use, then reuse the cached copy."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH

    print(f"Downloading face detection model to {MODEL_PATH} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            f"Failed to download face detection model from {MODEL_URL}: {e}\n"
            "You can also download it manually and place it at "
            f"{MODEL_PATH}"
        )
    return MODEL_PATH


def detect_face_center(img):
    """Return (cx, cy, bbox) of the union of all detected faces, or None.

    Uses OpenCV's FaceDetectorYN (a small ONNX/DNN model), which replaced
    CascadeClassifier-based Haar detection in OpenCV 5.x.
    """
    model_path = ensure_model()
    img_h, img_w = img.shape[:2]

    detector = cv2.FaceDetectorYN.create(
        model_path,
        "",                # config path, unused for ONNX
        (img_w, img_h),    # input size
        0.9,               # score threshold
        0.3,               # NMS threshold
        5000,              # top_k
    )
    detector.setInputSize((img_w, img_h))

    _, faces = detector.detect(img)

    if faces is None or len(faces) == 0:
        return None

    # Each row: [x, y, w, h, <5 landmark x,y pairs>, score]
    xs = faces[:, 0]
    ys = faces[:, 1]
    xe = faces[:, 0] + faces[:, 2]
    ye = faces[:, 1] + faces[:, 3]

    x_min, y_min = xs.min(), ys.min()
    x_max, y_max = xe.max(), ye.max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    return cx, cy, (x_min, y_min, x_max, y_max)


def detect_saliency_center(img):
    """Return (cx, cy) of the most salient region.

    Implements the spectral residual saliency algorithm (Hou & Zhang, 2007)
    directly with FFT/numpy, rather than relying on cv2.saliency. That
    module lives in opencv_contrib and is not present in every OpenCV
    build (notably, it was dropped from OpenCV 5.0's default build), so
    this avoids depending on it.
    """
    h, w = img.shape[:2]

    # Work on a small grayscale version for speed; the algorithm is scale
    # invariant enough that this doesn't hurt accuracy.
    small_w, small_h = 128, int(128 * h / w) if w else 128
    small_h = max(1, small_h)
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    dft = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    magnitude, phase = cv2.cartToPolar(dft[:, :, 0], dft[:, :, 1])

    log_magnitude = np.log(magnitude + 1e-9)
    # Local average of the log spectrum, approximating the "smooth" baseline
    avg_log_magnitude = cv2.blur(log_magnitude, (3, 3))
    spectral_residual = log_magnitude - avg_log_magnitude

    real = np.exp(spectral_residual) * np.cos(phase)
    imag = np.exp(spectral_residual) * np.sin(phase)
    combined = cv2.merge([real, imag])

    saliency_map = cv2.idft(combined, flags=cv2.DFT_SCALE)
    saliency_map = cv2.magnitude(saliency_map[:, :, 0], saliency_map[:, :, 1])
    saliency_map = saliency_map ** 2
    saliency_map = cv2.GaussianBlur(saliency_map, (7, 7), sigmaX=3)

    cv2.normalize(saliency_map, saliency_map, 0, 255, cv2.NORM_MINMAX)
    saliency_map = saliency_map.astype("uint8")

    # Threshold to keep only the most salient pixels, then find their centroid
    thresh_val = np.percentile(saliency_map, 90)
    _, thresh = cv2.threshold(saliency_map, thresh_val, 255, cv2.THRESH_BINARY)

    ys, xs = np.nonzero(thresh)
    if len(xs) == 0:
        return w / 2, h / 2

    # Map centroid from the small working image back to full resolution
    scale_x = w / small_w
    scale_y = h / small_h
    cx = xs.mean() * scale_x
    cy = ys.mean() * scale_y
    return cx, cy


def parse_aspect(aspect_str):
    w_str, h_str = aspect_str.split(":")
    return float(w_str), float(h_str)


def compute_crop_box(img_w, img_h, cx, cy, target_w, target_h):
    """Compute a crop box of size (target_w, target_h) centered on (cx, cy),
    clamped so it stays fully inside the image."""

    # If the requested crop is bigger than the image in either dimension,
    # scale it down to fit while preserving aspect ratio.
    scale = min(img_w / target_w, img_h / target_h, 1.0)
    crop_w = target_w * scale
    crop_h = target_h * scale

    x1 = cx - crop_w / 2
    y1 = cy - crop_h / 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Shift box back inside image bounds if it overflows
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


def smart_crop(input_path, output_path, width=None, height=None,
                aspect=None, scale=1.0, resize_to=None):
    img = cv2.imread(input_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    img_h, img_w = img.shape[:2]

    # Determine the target crop size
    if width and height:
        target_w, target_h = float(width), float(height)
    elif aspect:
        ar_w, ar_h = parse_aspect(aspect)
        # Use the largest crop of that aspect ratio that fits in the image
        if img_w / img_h > ar_w / ar_h:
            target_h = img_h
            target_w = target_h * ar_w / ar_h
        else:
            target_w = img_w
            target_h = target_w * ar_h / ar_w
    else:
        # Default: square crop as large as fits
        side = min(img_w, img_h)
        target_w = target_h = side

    target_w *= scale
    target_h *= scale

    try:
        face_result = detect_face_center(img)
    except Exception as e:
        print(f"Warning: face detection unavailable ({e}); falling back to saliency.", file=sys.stderr)
        face_result = None

    if face_result is not None:
        cx, cy, bbox = face_result
        source = "face detection"
    else:
        cx, cy = detect_saliency_center(img)
        source = "saliency detection"

    x1, y1, x2, y2 = compute_crop_box(img_w, img_h, cx, cy, target_w, target_h)
    cropped = img[y1:y2, x1:x2]

    if resize_to:
        out_w, out_h = resize_to
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, cropped)
    print(f"Centered using {source} at ({cx:.0f}, {cy:.0f})")
    print(f"Cropped region: x[{x1}:{x2}] y[{y1}:{y2}] -> saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to input image")
    parser.add_argument("output", help="Path to save cropped output image")
    parser.add_argument("--width", type=float, help="Target crop width in pixels")
    parser.add_argument("--height", type=float, help="Target crop height in pixels")
    parser.add_argument("--aspect", type=str, help="Target aspect ratio, e.g. '4:5' or '1:1'")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Shrink the crop box by this factor (0-1) to zoom in more tightly, default 1.0")
    parser.add_argument("--resize-width", type=int, help="Resize final output to this width")
    parser.add_argument("--resize-height", type=int, help="Resize final output to this height")

    args = parser.parse_args()

    resize_to = None
    if args.resize_width and args.resize_height:
        resize_to = (args.resize_width, args.resize_height)

    try:
        smart_crop(
            args.input,
            args.output,
            width=args.width,
            height=args.height,
            aspect=args.aspect,
            scale=args.scale,
            resize_to=resize_to,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()