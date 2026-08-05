#Model Summary Script

#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any, Dict, Tuple
import sys, os, argparse, traceback

# from VIT_CTC import ViT_CTC   # <-- Update if your model file name differs
from VIT_CTC import (
    ViT_CTC,
    PATCH_SIZE,
    EMBED_DIM,
    NUM_LAYERS,
    NUM_HEADS,
    MLP_DIM,
    DROPOUT,
    MAX_PATCHES,
    IMG_HEIGHT,
    FIXED_WIDTH,
    BEAM_WIDTH
)
# Try importing torchinfo (for detailed architecture)
try:
    from torchinfo import summary as torchinfo_summary
except ImportError:
    print("⚠️ torchinfo not found. Install with: pip install torchinfo")
    torchinfo_summary = None


# ====================================================
# 1) Vocabulary builder
# ====================================================
def build_vocab(tsv_path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"Validation file not found: {tsv_path}")

    counts = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if "\t" not in line:
                continue
            _, txt = line.rstrip("\n").split("\t", 1)
            for ch in txt:
                if ch.strip():
                    counts[ch] = counts.get(ch, 0) + 1

    chars = sorted(counts.keys())
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}
    print(f"✅ Built vocab of size {len(char2id)} from {tsv_path}")
    return char2id, id2char


# ====================================================
# 2) Build model with weights
# ====================================================
def build_model(weights_path: str, label_path: str) -> nn.Module:

    # ---------- Build vocab automatically ----------
    char2id, id2char = build_vocab(label_path)

    blank_token_id = len(char2id)

    num_classes = len(char2id) + 1

    print(f"Vocabulary Size : {len(char2id)}")
    print(f"Number of Classes : {num_classes}")
    print(f"Blank Token ID : {blank_token_id}")

    # ---------- Create model ----------
    model = ViT_CTC(
        img_channels=1,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        dropout=DROPOUT,
        num_classes=num_classes,
        max_patches=MAX_PATCHES,
    )

    # ---------- Load weights ----------
    if weights_path and os.path.exists(weights_path):

        print(f"\nLoading weights from: {weights_path}")

        state = torch.load(
            weights_path,
            map_location="cpu"
        )

        model_state = model.state_dict()

        filtered_state = {}

        skipped = []

        for k, v in state.items():

            if k in model_state and model_state[k].shape == v.shape:

                filtered_state[k] = v

            else:

                skipped.append(k)

        missing, unexpected = model.load_state_dict(
            filtered_state,
            strict=False
        )

        print(f"✅ Loaded {len(filtered_state)} layers successfully.")

        if skipped:
            print(f"⚠️ Skipped {len(skipped)} mismatched layers")

        if missing:
            print(f"⚠️ Missing keys: {missing}")

        if unexpected:
            print(f"⚠️ Unexpected keys: {unexpected}")

    else:
        print("⚠️ No valid weights provided.")

    return model, num_classes


# ====================================================
# 3) Helper utilities
# ====================================================
def fmt_int(x: int) -> str:
    return f"{x:,}"

def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def extract_vit_hparams(model: nn.Module) -> Dict[str, Any]:
    hp = {}
    if hasattr(model, "patch_embed"):
        pe = getattr(model, "patch_embed")
        if hasattr(pe, "proj"):
            hp["patch_size"] = pe.proj.kernel_size
    for k, v in vars(model).items():
        if isinstance(v, (int, float, str)):
            hp[k] = v
    return hp

def print_hparams(hp: Dict[str, Any]):
    print("=== Hyperparameters ===")
    for k, v in hp.items():
        print(f"{k:15s}: {v}")
    print()

def print_param_summary(model: nn.Module):
    total, trainable = count_params(model)
    print("=== Parameters ===")
    print(f"Total params    : {fmt_int(total)}")
    print(f"Trainable params: {fmt_int(trainable)}")
    print()


# ====================================================
# 4) Detailed architecture summary (robust version)
# ====================================================
def summarize_model(model: nn.Module,
                    num_classes,
                    img_size= (IMG_HEIGHT, FIXED_WIDTH),
                    in_chans: int = 1,
                    device: str = "cpu",
                    batch_size: int = 1):
    from datetime import datetime
    print(f"\nModel Summary Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 100)

    hp = extract_vit_hparams(model)
    hp.setdefault("img_size", img_size)
    hp.setdefault("in_chans", in_chans)
    print_hparams(hp)
    print_param_summary(model)
    # print_param_summary(model)

    blank_id = num_classes - 1

    print_ctc_summary(
        model=model,
        num_classes=num_classes,
        blank_id=blank_id,
        img_size=img_size
    )

    H, W = img_size
    x = torch.zeros(batch_size, in_chans, H, W, device=device)

    print("\n=== Detailed Architecture ===")
    print("-" * 100)
    if torchinfo_summary:
        try:
            torchinfo_summary(
                model,
                input_data=x,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                depth=5,
                verbose=1,
            )
        except Exception as e:
            print(f"⚠️ torchinfo failed: {e}")
            print("Full traceback:\n" + "".join(traceback.format_exc()))
            print("\n📄 Static architecture (module tree only):")
            print(model)
    else:
        print("torchinfo not installed. Run: pip install torchinfo")
        print("\n📄 Static architecture (module tree only):")
        print(model)

    print("-" * 100)
    print("✅ Summary generation complete.\n")
    
    
# ====================================================
# 6) CTC model summary

def print_ctc_summary(model, num_classes, blank_id, img_size):
    print("=== CTC Configuration ===")

    H, W = img_size
    ph, pw = model.patch_embed.proj.kernel_size

    # Number of patches
    num_patches_h = H // ph
    num_patches_w = W // pw
    sequence_length = num_patches_h * num_patches_w

    print(f"Input Image Size        : {img_size}")
    print(f"Patch Size              : ({ph}, {pw})")
    print(f"Patches Along Height    : {num_patches_h}")
    print(f"Patches Along Width     : {num_patches_w}")
    print(f"Total Sequence Length   : {sequence_length}")

    print(f"Embedding Dimension     : {model.classifier.in_features}")
    print(f"Transformer Layers      : {len(model.layers)}")
    print(f"Attention Heads         : {model.layers[0].mha.num_heads}")

    print(f"Vocabulary Size         : {num_classes - 1}")
    print(f"CTC Output Classes      : {num_classes}")
    print(f"Blank Token ID          : {blank_id}")

    print(f"CTC Loss                : nn.CTCLoss")
    print(f"Beam Width              : {BEAM_WIDTH}")

    print()

# ====================================================


# ====================================================
# 5) CLI Entry Point
# ====================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate model summary from trained ViT-CTC model"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default= "/bhupendra-data/Indic_OCR/VIT_OCR_SK_Binarised/checkpoints/best_model_epoch69.pt",
        help="Path to trained .pt model"
    )

    parser.add_argument(
        "--labels",
        type=str,
        default="/bhupendra-data/Indic_OCR/VIT_OCR_SK_Binarised/checkpoints/label_map.json",
        help="Path to train/val label TSV file"
    )

    parser.add_argument(
        "--out",
        type=str,
        default="model_summary.txt",
        help="Output summary txt file"
    )

    args = parser.parse_args()

    device = "cpu"

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True
    )

    with open(args.out, "w", encoding="utf-8") as f:

        old_stdout = sys.stdout

        sys.stdout = f

        # ---------- Build model ----------
        model, num_classes = build_model(
            args.weights,
            args.labels
        )

        # ---------- Generate summary ----------
        summarize_model(
            model,
            num_classes=num_classes,
            img_size=(IMG_HEIGHT, FIXED_WIDTH),
            in_chans=1,
            device=device
        )

        sys.stdout = old_stdout

    print(f"\n✅ Model summary saved to: {args.out}")
