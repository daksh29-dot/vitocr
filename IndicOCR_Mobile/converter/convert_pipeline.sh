#!/usr/bin/env bash
# =============================================================================
# convert_pipeline.sh
# =============================================================================
# One-shot conversion pipeline:
#   PyTorch .pt  →  ONNX (FP32)  →  Core ML (.mlpackage)
#
# Usage:
#   chmod +x convert_pipeline.sh
#   ./convert_pipeline.sh \
#       --weights  /path/to/best_model.pt \
#       --labelmap /path/to/label_map.json \
#       --channels 1           # 1=grayscale, 2=dual-channel
#       --outdir   ./output
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
WEIGHTS=""
LABELMAP=""
CHANNELS=1
OUTDIR="./converted_models"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights)   WEIGHTS="$2";   shift 2 ;;
        --labelmap)  LABELMAP="$2";  shift 2 ;;
        --channels)  CHANNELS="$2";  shift 2 ;;
        --outdir)    OUTDIR="$2";    shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -z "$WEIGHTS"  ]] && { echo "❌  --weights required";  exit 1; }
[[ -z "$LABELMAP" ]] && { echo "❌  --labelmap required"; exit 1; }
[[ ! -f "$WEIGHTS"  ]] && { echo "❌  Weights file not found: $WEIGHTS";  exit 1; }
[[ ! -f "$LABELMAP" ]] && { echo "❌  Label map not found:  $LABELMAP";  exit 1; }

mkdir -p "$OUTDIR"

ONNX_FP32="$OUTDIR/vit_ctc_fp32.onnx"
ONNX_INT8="$OUTDIR/vit_ctc_int8.onnx"
COREML="$OUTDIR/ViTCTC.mlpackage"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Indic OCR – ARM Mobile Conversion Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo "  Weights   : $WEIGHTS"
echo "  Label map : $LABELMAP"
echo "  Channels  : $CHANNELS"
echo "  Output dir: $OUTDIR"
echo "════════════════════════════════════════════════════════════════"

# ── Step 1: PyTorch → ONNX FP32 ──────────────────────────────────────────────
echo ""
echo "▶ Step 1/3 – Export to ONNX (FP32)"
python3 "$SCRIPT_DIR/export_to_onnx.py" \
    --weights   "$WEIGHTS"  \
    --labelmap  "$LABELMAP" \
    --out       "$ONNX_FP32" \
    --channels  "$CHANNELS"
echo "  ✅ FP32 ONNX → $ONNX_FP32"

# ── Step 2: ONNX INT8 quantisation ───────────────────────────────────────────
echo ""
echo "▶ Step 2/3 – INT8 dynamic quantisation"
python3 "$SCRIPT_DIR/quantize_onnx.py" \
    --onnx "$ONNX_FP32" \
    --out  "$ONNX_INT8" \
    --mode int8
echo "  ✅ INT8 ONNX → $ONNX_INT8"

# ── Step 3: ONNX → Core ML (iOS only) ────────────────────────────────────────
echo ""
echo "▶ Step 3/3 – Convert to Core ML (.mlpackage)"
if python3 -c "import coremltools" 2>/dev/null; then
    python3 "$SCRIPT_DIR/convert_to_coreml.py" \
        --onnx    "$ONNX_FP32" \
        --out     "$COREML"    \
        --min-ios "16"
    echo "  ✅ Core ML → $COREML"
else
    echo "  ⚠️  coremltools not found – skipping Core ML step."
    echo "     Install with: pip install coremltools"
    echo "     Then run:  python3 convert_to_coreml.py --onnx $ONNX_FP32 --out $COREML"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Conversion complete!  Output files:"
echo ""
ls -lh "$OUTDIR"
echo ""
echo "  Android deployment:"
echo "    • Copy  $ONNX_INT8  → android/app/src/main/assets/vit_ctc.onnx"
echo "    • Copy  $LABELMAP   → android/app/src/main/assets/label_map.json"
echo ""
echo "  iOS deployment:"
echo "    • Drag  $COREML      into your Xcode project (add to target)"
echo "    • Drag  $LABELMAP    into your Xcode project (add to target)"
echo "════════════════════════════════════════════════════════════════"