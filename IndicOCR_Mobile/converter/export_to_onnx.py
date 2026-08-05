#!/usr/bin/env python3
"""
export_to_onnx.py
=================
Converts your trained ViT-CTC PyTorch model (.pt) to ONNX format,
which is the universal intermediate format for both Android (ONNX Runtime)
and iOS (Core ML via onnx-coreml or directly via ONNX Runtime Mobile).
 
Usage:
    python export_to_onnx.py \
        --weights /path/to/best_model.pt \
        --labelmap /path/to/label_map.json \
        --out model.onnx \
        --channels 1        # use 2 if your model was trained with binarised+original dual-channel
 
The exported model:
  - Input  : float32 tensor  [1, C, 32, 256]   (C=1 grayscale or C=2 dual-channel)
  - Output : float32 tensor  [1, 512, num_classes]  (CTC logits)
"""
 
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxsim
 
# ── Model definition (must match your training code exactly) ──────────────────
 
IMG_HEIGHT  = 32
FIXED_WIDTH = 256
PATCH_SIZE  = (4, 4)
EMBED_DIM   = 384
NUM_HEADS   = 6
MLP_DIM     = 1024
NUM_LAYERS  = 8
DROPOUT     = 0.1
MAX_PATCHES = 1024
 
 
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=(4, 4), embed_dim=384):
        super().__init__()
        ph, pw = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=(ph, pw), stride=(ph, pw))
 
    def forward(self, x):
        x = self.proj(x)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (H, W)
 
 
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.mha  = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.fc1  = nn.Linear(embed_dim, mlp_dim)
        self.fc2  = nn.Linear(mlp_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
 
    def forward(self, x):
        h = self.ln1(x)
        attn, _ = self.mha(h, h, h)
        x = x + self.drop(attn)
        h = self.ln2(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + self.drop(h)
 
 
class ViT_CTC(nn.Module):
    def __init__(self, img_channels, patch_size, embed_dim,
                 num_layers, num_heads, mlp_dim, dropout,
                 num_classes, max_patches=MAX_PATCHES):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_channels, patch_size, embed_dim)
        self.layers      = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier  = nn.Linear(embed_dim, num_classes)
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
 
    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        N = H * W
        # Use min() instead of if/else so ONNX trace stays static
        N = min(N, self.pos_embed.size(1))
        x = x[:, :N, :]
        x = x + self.pos_embed[:, :N, :]
        for blk in self.layers:
            x = blk(x)
        return self.classifier(x)  # [B, T, num_classes]
 
 
# ── ONNX-friendly wrapper: bakes log_softmax in ──────────────────────────────
 
class ViT_CTC_ONNX(nn.Module):
    """Wraps the model so the output is already log-softmax probabilities.
    This avoids doing log_softmax on-device and simplifies decoding."""
 
    def __init__(self, base_model: ViT_CTC):
        super().__init__()
        self.model = base_model
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)                     # [1, T, C]
        return F.log_softmax(logits, dim=-1)       # [1, T, C]
 
 
# ── Load & export ────────────────────────────────────────────────────────────
 
def load_model(weights_path: str, num_classes: int, img_channels: int) -> ViT_CTC:
    model = ViT_CTC(
        img_channels=img_channels,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        num_classes=num_classes,
        max_patches=MAX_PATCHES,
    )
    state = torch.load(weights_path, map_location="cpu")
    # Tolerate minor mismatches (e.g. classifier shape changes between runs)
    model_state = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in model_state and model_state[k].shape == v.shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  ⚠️  Missing keys : {missing}")
    if unexpected:
        print(f"  ⚠️  Unexpected   : {unexpected}")
    print(f"  ✅ Loaded {len(filtered)} / {len(state)} tensors.")
    return model
 
 
def export(args):
    print("\n─── Loading label map ───────────────────────────────────────────")
    with open(args.labelmap, "r", encoding="utf-8") as f:
        lm = json.load(f)
    char2id    = lm["char2id"]
    num_classes = len(char2id)          # blank is already in the map
    print(f"  num_classes = {num_classes}  (including blank)")
 
    print("\n─── Building model ──────────────────────────────────────────────")
    base = load_model(args.weights, num_classes, args.channels)
    base.eval()
 
    export_model = ViT_CTC_ONNX(base)
    export_model.eval()
 
    dummy = torch.zeros(1, args.channels, IMG_HEIGHT, FIXED_WIDTH)
 
    print("\n─── Exporting to ONNX ───────────────────────────────────────────")
    onnx_path = args.out
    torch.onnx.export(
        export_model,
        dummy,
        onnx_path,
        opset_version=17,               # opset 17 has best ARM support
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={
            "image":     {0: "batch"},
            "log_probs": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,                   # use stable TorchScript exporter
    )
    print(f"  ✅ Saved raw ONNX → {onnx_path}")
 
    print("\n─── Verifying ONNX model ────────────────────────────────────────")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ✅ Model check passed.")
 
    print("\n─── Simplifying ONNX model ──────────────────────────────────────")
    try:
        simplified, ok = onnxsim.simplify(onnx_model)
        if ok:
            onnx.save(simplified, onnx_path)
            print(f"  ✅ Simplified model saved → {onnx_path}")
        else:
            print("  ⚠️  Simplification failed; original kept.")
    except Exception as e:
        print(f"  ⚠️  onnxsim error: {e}; original kept.")
 
    # Quick inference sanity-check with onnxruntime
    print("\n─── Sanity-checking with ONNX Runtime ───────────────────────────")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {"image": dummy.numpy()})
        print(f"  ✅ Output shape: {out[0].shape}  (expected [1, 512, {num_classes}])")
    except Exception as e:
        print(f"  ⚠️  ORT check failed: {e}")
 
    print(f"\n🎉 ONNX export complete → {onnx_path}\n")
    return onnx_path
 
 
def main():
    ap = argparse.ArgumentParser(description="Export ViT-CTC to ONNX for mobile")
    ap.add_argument("--weights",  required=True, help="Path to .pt checkpoint")
    ap.add_argument("--labelmap", required=True, help="Path to label_map.json")
    ap.add_argument("--out",      default="vit_ctc.onnx", help="Output ONNX file")
    ap.add_argument("--channels", type=int, default=2,
                    help="1 = grayscale (inferFolder.py), 2 = dual-channel (VIT_CTC.py trainer)")
    args = ap.parse_args()
 
    if not os.path.exists(args.weights):
        sys.exit(f"❌ Weights not found: {args.weights}")
    if not os.path.exists(args.labelmap):
        sys.exit(f"❌ Label map not found: {args.labelmap}")
 
    export(args)
 
 
if __name__ == "__main__":
    main()
 