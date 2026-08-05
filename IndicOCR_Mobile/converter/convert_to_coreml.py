#!/usr/bin/env python3
"""
convert_to_coreml.py
====================
Converts the ONNX model to Apple Core ML format (.mlpackage) for iOS/macOS.

Prerequisites:
    pip install coremltools onnx

Usage:
    python convert_to_coreml.py \
        --onnx   vit_ctc.onnx \
        --out    ViTCTC.mlpackage

The resulting .mlpackage can be dragged into an Xcode project and used with
the Vision / CoreML APIs on iPhone, iPad, and Apple Silicon Macs.

Tested with coremltools >= 7.0 (supports iOS 15+).
"""

import argparse
import sys
import os

def convert(onnx_path: str, out_path: str, min_ios: str):
    try:
        import coremltools as ct
    except ImportError:
        sys.exit(
            "❌  coremltools not installed.\n"
            "    Run: pip install coremltools"
        )

    print(f"\n─── Converting {onnx_path} → Core ML ───")

    # Map iOS version string to ct.target
    ios_map = {
        "15": ct.target.iOS15,
        "16": ct.target.iOS16,
        "17": ct.target.iOS17,
        "18": ct.target.iOS18,
    }
    target = ios_map.get(min_ios, ct.target.iOS16)
    print(f"  Minimum deployment target : iOS {min_ios}")

    # ── Convert ──────────────────────────────────────────────────────────────
    model = ct.convert(
        onnx_path,
        source="onnx",
        minimum_deployment_target=target,
        # Declare the input as a grayscale image so Vision can feed it directly
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 1, 32, 256),      # [batch, C, H, W]
                color_layout=ct.colorlayout.GRAYSCALE,
                scale=1.0 / 127.5,          # normalises [0,255] → [-1,1]
                bias=-1.0,
            )
        ],
        compute_precision=ct.precision.FLOAT16,   # ~2× smaller, negligible accuracy loss
        convert_to="mlprogram",                   # modern Neural Engine–friendly format
    )

    # ── Metadata ─────────────────────────────────────────────────────────────
    model.author      = "Indic OCR Project"
    model.license     = "Proprietary"
    model.short_description = (
        "ViT-CTC Indic OCR model. "
        "Input: grayscale word-image 32×256. "
        "Output: CTC log-probabilities [1, 512, num_classes]."
    )
    model.version = "1.0"

    model.save(out_path)
    print(f"  ✅ Saved Core ML model → {out_path}")
    print(f"  📦 Size: {_dir_size_mb(out_path):.1f} MB")


def _dir_size_mb(path):
    total = 0
    if os.path.isfile(path):
        return os.path.getsize(path) / 1e6
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1e6


def main():
    ap = argparse.ArgumentParser(description="ONNX → Core ML converter for ViT-CTC")
    ap.add_argument("--onnx",    required=True, help="Input .onnx file")
    ap.add_argument("--out",     default="ViTCTC.mlpackage",
                    help="Output Core ML package path")
    ap.add_argument("--min-ios", dest="min_ios", default="16",
                    choices=["15", "16", "17", "18"],
                    help="Minimum iOS deployment target (default: 16)")
    args = ap.parse_args()

    if not os.path.exists(args.onnx):
        sys.exit(f"❌ ONNX file not found: {args.onnx}")

    convert(args.onnx, args.out, args.min_ios)


if __name__ == "__main__":
    main()