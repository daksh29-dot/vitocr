"""
quantize_model.py
==================
Dynamic INT8 quantization for the ViT-CTC Indic OCR ONNX model.

Why dynamic (not static) quantization:
    ONNX Runtime's own guidance is dynamic quantization for transformer /
    attention-based models (weights quantized ahead of time, activations
    quantized on-the-fly at inference) and static quantization for CNNs
    (which need a calibration dataset to pre-compute activation ranges).
    A ViT-CTC model is dominated by MatMul/Gemm ops from the attention and
    feed-forward blocks, so dynamic quantization is the right default here
    and needs no calibration images.

Usage:
    pip install onnx onnxruntime

    python quantize_model.py \
        --model /path/to/vit_ctc.onnx \
        --out   /path/to/vit_ctc_int8.onnx

    # Optional sanity check with a real image (needs the same preprocessing
    # as IndicOCREngine.java: grayscale, resize to H=32/max-W=256, dual
    # channel with Otsu threshold, normalize mean=0.5/std=0.5):
    python quantize_model.py \
        --model /path/to/vit_ctc.onnx \
        --out   /path/to/vit_ctc_int8.onnx \
        --labels /path/to/label_map.json \
        --test-image /path/to/sample_word.png
"""

import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# ── Must match training config / IndicOCREngine.java constants ─────────────
IMG_H = 32
IMG_W = 256
CHANNELS = 2  # 1 = grayscale only, 2 = grayscale + Otsu-binarised
NORM_MEAN = 0.5
NORM_STD = 0.5


def human_size(path):
    return f"{os.path.getsize(path) / (1024 * 1024):.2f} MB"


def quantize(model_path, out_path, weight_type, per_channel):
    print(f"Loading model: {model_path} ({human_size(model_path)})")
    onnx.checker.check_model(onnx.load(model_path))
    print("Model check passed.")

    print(f"Running dynamic quantization (weight_type={weight_type.name}, "
          f"per_channel={per_channel}) ...")
    t0 = time.time()
    quantize_dynamic(
        model_input=model_path,
        model_output=out_path,
        weight_type=weight_type,
        per_channel=per_channel,
        # MatMul/Gemm are where a ViT's weight is concentrated; Conv shows
        # up in the patch-embedding layer. Leave default op set unless you
        # need to exclude an op that errors out below.
    )
    print(f"Done in {time.time() - t0:.1f}s")

    orig_mb = os.path.getsize(model_path) / (1024 * 1024)
    quant_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\nOriginal:  {orig_mb:.2f} MB")
    print(f"Quantized: {quant_mb:.2f} MB")
    print(f"Reduction: {(1 - quant_mb / orig_mb) * 100:.1f}%")


def sanity_check_random(model_path):
    """Confirms the quantized model loads and runs with the expected input shape."""
    print("\nSanity check (random input) ...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, CHANNELS, IMG_H, IMG_W).astype(np.float32)
    t0 = time.time()
    outputs = session.run(None, {input_name: dummy})
    ms = (time.time() - t0) * 1000
    print(f"Inference OK. Output shape: {outputs[0].shape}  ({ms:.1f} ms)")


def preprocess_image(image_path):
    """Mirrors IndicOCREngine.java preprocessBitmap(): grayscale, resize to
    H=32 keeping aspect ratio (max W=256), right-pad with white, normalize,
    optionally add an Otsu-binarised second channel."""
    from PIL import Image

    img = Image.open(image_path).convert("L")  # grayscale
    src_w, src_h = img.size
    new_h = IMG_H
    new_w = min(IMG_W, max(1, round(src_w * (new_h / src_h))))
    img = img.resize((new_w, new_h), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    canvas = np.ones((IMG_H, IMG_W), dtype=np.float32)  # white padding
    canvas[:, :new_w] = arr
    norm = (canvas - NORM_MEAN) / NORM_STD

    if CHANNELS == 1:
        return norm[np.newaxis, np.newaxis, :, :].astype(np.float32)

    # Otsu threshold on the [0,1] pixel values
    pixel_vals = (norm * NORM_STD + NORM_MEAN) * 255.0
    hist, _ = np.histogram(pixel_vals, bins=256, range=(0, 256))
    total = pixel_vals.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b, max_var, threshold = 0.0, 0.0, 0.0, 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > max_var:
            max_var, threshold = var, t

    otsu = np.where(pixel_vals >= threshold, 255.0, 0.0)
    otsu_norm = (otsu / 255.0 - NORM_MEAN) / NORM_STD

    dual = np.stack([norm, otsu_norm], axis=0)  # (2, H, W)
    return dual[np.newaxis, :, :, :].astype(np.float32)


def greedy_ctc_decode(log_probs, id2char, blank_id):
    ids = np.argmax(log_probs, axis=-1)
    out, prev = [], -1
    for i in ids:
        if i != prev and i != blank_id:
            out.append(id2char.get(str(i), ""))
        prev = i
    return "".join(out)


def sanity_check_real_image(model_path, image_path, label_map_path):
    print(f"\nSanity check (real image): {image_path}")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id2char = label_map["id2char"]
    blank_id = next((int(k) for k, v in id2char.items() if v == "<BLANK>"), max(int(k) for k in id2char))

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    inp = preprocess_image(image_path)
    outputs = session.run(None, {input_name: inp})
    text = greedy_ctc_decode(outputs[0][0], id2char, blank_id)
    print(f"Decoded text: {text!r}")


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for the ViT-CTC ONNX model")
    parser.add_argument("--model", required=True, help="Path to the FP32 .onnx model")
    parser.add_argument("--out", default="vit_ctc_int8.onnx", help="Output path for the quantized model")
    parser.add_argument(
        "--weight-type", choices=["QUInt8", "QInt8"], default="QUInt8",
        help="QUInt8 is the safer default for ARM/mobile (NNAPI); try QInt8 if you hit op-support issues."
    )
    parser.add_argument("--per-channel", action="store_true",
                         help="Per-channel weight quantization: usually better accuracy, slightly larger model.")
    parser.add_argument("--labels", default=None, help="Optional path to label_map.json for a real-image sanity check")
    parser.add_argument("--test-image", default=None, help="Optional word-crop image to sanity-check decoding")
    args = parser.parse_args()

    weight_type = QuantType.QUInt8 if args.weight_type == "QUInt8" else QuantType.QInt8

    quantize(args.model, args.out, weight_type, args.per_channel)
    sanity_check_random(args.out)

    if args.test_image and args.labels:
        sanity_check_real_image(args.out, args.test_image, args.labels)
    elif args.test_image or args.labels:
        print("\nSkipping real-image check: need both --labels and --test-image.")

    print(f"\nDone. Point IndicOCREngine's MODEL_ASSET at: {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()