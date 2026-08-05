#Model Summary Script

#!/usr/bin/env python3
import torch
import torch.nn as nn
from collections import defaultdict
from typing import Any, Dict, Tuple
import sys, os, argparse, traceback

# Import your ViT model
from VIT_CTC import ViT_CTC   # <-- Update if your model file name differs

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
def build_model(weights_path: str, num_classes: int) -> nn.Module:
    model = ViT_CTC(
        img_channels=1,
        patch_size=(4, 4),
        embed_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_dim=1024,
        dropout=0.1,
        num_classes=num_classes,
        max_patches=1024,
    )

    if weights_path and os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state = torch.load(weights_path, map_location="cpu")

        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                skipped.append(k)

        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        print(f"✅ Loaded {len(filtered_state)} layers successfully.")
        if skipped:
            print(f"⚠️ Skipped {len(skipped)} mismatched layers: {skipped}")
        if missing:
            print(f"⚠️ Missing keys in checkpoint: {missing}")
        if unexpected:
            print(f"⚠️ Unexpected keys in checkpoint: {unexpected}")
    else:
        print("⚠️ No valid weights provided; model initialized randomly.")
    return model


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
                    img_size: Tuple[int, int] = (32, 256),
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
# 5) CLI Entry Point
# ====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize ViT+CTC model with detailed architecture.")
    parser.add_argument("--weights", default="/bhupendra-data/Indic_OCR/VIT_OCR_SK/checkpoints/best_model_epoch54.pt",
                        type=str, help="Path to .pt weights")
    parser.add_argument("--val_tsv", type=str, required=True, help="Path to validation TSV (for vocab)")
    parser.add_argument("--out", type=str, default="vit_summary.txt", help="Output summary file")
    args = parser.parse_args()

    device = "cpu" if torch.cuda.is_available() else "cpu"

    # Build vocab → compute num_classes automatically
    char2id, id2char = build_vocab(args.val_tsv)
    blank_token_id = len(char2id)
    num_classes = len(char2id) + 1  # include <blank>

    # Redirect stdout to file
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        model = build_model(args.weights, num_classes)
        summarize_model(model, img_size=(32, 256), in_chans=1, device=device)
        sys.stdout = old_stdout

    print(f"✅ Model summary (with vocab size {num_classes}) saved to: {args.out}")
