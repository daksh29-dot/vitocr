#!/usr/bin/env python3
"""
Adaptive-memory ViT + CTC training script with fixed-size preprocessing.
Images are resized to fixed height and width while preserving aspect ratio:
 - If image < fixed_width → pad with white (255)
 - If image > fixed_width → resize proportionally to fit
This ensures stable memory and consistent model input.
"""

import os, math, logging, argparse
from typing import List, Dict, Tuple
from PIL import Image, ImageOps
import cv2
import numpy as np
from tqdm import tqdm
import editdistance
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------- Hyperparameters ----------
IMG_HEIGHT = 32
FIXED_WIDTH = 256
PATCH_SIZE = (4, 4)
EMBED_DIM = 384
NUM_HEADS = 6
MLP_DIM = 1024
NUM_LAYERS = 8
DROPOUT = 0.1
MAX_PATCHES = 1024
BLANK_TOKEN = "<BLANK>"
BEAM_WIDTH = 32

# ---------- Logger ----------
def setup_logger(save_dir="checkpoints", log_name="train.log"):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, log_name)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a", encoding="utf-8")]
    )
    return logging.getLogger(__name__), log_file

# ---------- Vocabulary ----------
def build_vocab(tsv_paths: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts = {}
    for p in tsv_paths:
        if not p:
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if "\t" not in line:
                    continue
                _, txt = line.rstrip("\n").split("\t", 1)
                for ch in txt:
                    # ✅ Skip unwanted whitespace or control characters
                    if ch.strip() == "":
                        continue
                    counts[ch] = counts.get(ch, 0) + 1

    #Sort characters deterministically (for reproducibility)
    chars = sorted(counts.keys())

    # Map characters to indices and back
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}

    return char2id, id2char


# ---------- Preprocessing ----------

def preprocess_image(img_path: str, img_height: int = IMG_HEIGHT, fixed_width: int = FIXED_WIDTH):
    try:
        img = Image.open(img_path).convert('L')
    except Exception as e:
        raise ValueError(f"Cannot open image: {img_path} ({e})")

    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError(f"Invalid image, size=({w},{h}): {img_path}")

    # --- Resize while preserving aspect ratio ---
    new_h = img_height
    new_w = max(1, int(w * (new_h / h)))
    new_w = min(new_w, fixed_width)

    img = img.resize((new_w, new_h), Image.BILINEAR)

    if new_w < fixed_width:
        pad_width = fixed_width - new_w
        img = ImageOps.expand(img, border=(0, 0, pad_width, 0), fill=255)

    img_np = np.array(img)

    # -------------------------------
    # 1️⃣ Original normalized channel
    # -------------------------------
    orig = img_np.astype(np.float32) / 255.0
    orig = (orig - 0.5) / 0.5

    # -------------------------------
    # 2️⃣ Otsu binarized channel
    # -------------------------------
    _, otsu = cv2.threshold(
        img_np,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    otsu = otsu.astype(np.float32) / 255.0
    otsu = (otsu - 0.5) / 0.5

    # Stack → 2 channel tensor
    arr = np.stack([orig, otsu], axis=0)

    return arr


# ---------- Dataset ----------
class OCRDataset(Dataset):
    def __init__(self, tsv_path:str, char2id:Dict[str,int], img_height=IMG_HEIGHT, fixed_width=FIXED_WIDTH):
        self.samples = []
        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                if "\t" not in line: continue
                img_path, text = line.strip().split("\t",1)
                self.samples.append((img_path, text))
        self.char2id = char2id
        self.img_height = img_height
        self.fixed_width = fixed_width

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        try:
            arr = preprocess_image(img_path, self.img_height, self.fixed_width)
        except Exception as e:
            # Mark this sample as unusable → collate_fn will skip it
            print(f"[WARN] Skipping corrupt/bad image: {img_path} ({e})")
            return {"image": None, "labels": np.array([], dtype=np.int32), "text": ""}

        labels = [self.char2id[ch] for ch in text if ch in self.char2id]
        return {"image": arr, "labels": np.array(labels, dtype=np.int32), "text": text}


def collate_fn(batch):
    batch = [b for b in batch if b["image"] is not None and len(b['labels']) > 0]
    if len(batch) == 0: return None
    imgs = [b['image'] for b in batch]
    labels = [b['labels'] for b in batch]
    texts = [b['text'] for b in batch]
    padded = np.stack(imgs, axis=0)
    targets = np.concatenate(labels)
    target_lengths = np.array([len(l) for l in labels], dtype=np.int32)
    return {
        "images": torch.from_numpy(padded),
        "targets": torch.from_numpy(targets).long(),
        "target_lengths": torch.from_numpy(target_lengths).long(),
        "texts": texts
    }

# ---------- Vision Transformer ----------
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=(4,4), embed_dim=384):
        super().__init__()
        ph,pw = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(ph,pw), stride=(ph,pw))
    def forward(self,x):
        x = self.proj(x)
        B,D,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        return x, (H,W)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout,batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim,mlp_dim)
        self.fc2 = nn.Linear(mlp_dim,embed_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self,x):
        h=self.ln1(x)
        attn,_ = self.mha(h,h,h)
        x = x + self.drop(attn)
        h = self.ln2(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        x = x + self.drop(h)
        return x

class ViT_CTC(nn.Module):
    def __init__(self, img_channels, patch_size, embed_dim, num_layers, num_heads, mlp_dim, dropout, num_classes, max_patches=MAX_PATCHES):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_channels, patch_size, embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)])
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        N = H * W
        if N > self.pos_embed.size(1):
            x = x[:, :self.pos_embed.size(1), :]
            N = self.pos_embed.size(1)
        x = x + self.pos_embed[:, :N, :]
        for blk in self.layers:
            x = blk(x)
        return self.classifier(x)

# ---------- Beam Search Decoding ----------
def ctc_beam_search_decode_gpu(logits, id2char, blank_id, beam_width=32):
    id2char = {int(k): v for k, v in id2char.items()}
    B, T, C = logits.shape
    preds = []
    for b in range(B):
        log_probs = F.log_softmax(logits[b], dim=-1)
        topk = torch.topk(log_probs, k=beam_width, dim=-1)
        seq = torch.argmax(log_probs, dim=-1).tolist()
        out = []
        prev = None
        for s in seq:
            if s != prev and s != blank_id:
                out.append(id2char.get(s, ""))
            prev = s
        preds.append("".join(out))
    return preds

# ---------- Validation ----------
def validate(model, dataloader, blank_id, id2char, device, logger, beam_width=BEAM_WIDTH):
    model.eval()
    total_cer, total_wer, total_crr, total_wrr = 0,0,0,0
    n_samples = 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None: continue
            imgs = batch["images"].to(device)
            texts = batch["texts"]
            logits = model(imgs)
            preds = ctc_beam_search_decode_gpu(logits, id2char, blank_id, beam_width=beam_width)
            for gt, pr in zip(texts, preds):
                gt_w, pr_w = gt.split(), pr.split()
                cer = editdistance.eval(pr, gt)/max(1,len(gt))
                wer = editdistance.eval(pr_w, gt_w)/max(1,len(gt_w))
                total_cer += cer; total_wer += wer
                total_crr += (1-cer); total_wrr += (1-wer)
                n_samples += 1
    return {
        "CER": total_cer/n_samples,
        "WER": total_wer/n_samples,
        "CRR": total_crr/n_samples,
        "WRR": total_wrr/n_samples
    }

# ---------- Run Epoch ----------
def run_epoch(model, dataloader, criterion, optimizer, scaler, blank_id, device, train=True, epoch=0, logger=None):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        if batch is None: continue
        imgs = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)
        try:
            with torch.set_grad_enabled(train):
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    logits = model(imgs)
                    B,T,_ = logits.size()
                    logits_ctc = logits.log_softmax(2).permute(1,0,2)
                    input_lengths = torch.full((B,), logits_ctc.size(0), dtype=torch.long, device=device)
                    loss = criterion(logits_ctc, targets, input_lengths, target_lengths)
                if train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n_batches += 1
            del imgs, targets, target_lengths, logits, logits_ctc, loss
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                if logger: logger.warning(f"OOM at batch {batch_idx}. Skipping.")
                continue
            else:
                raise e
    return total_loss / max(1, n_batches)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default="/bhupendra-data/Indic_OCR/Training_data/OdiaTrainingData/TrainingData_05_12_25/bhashiniTrain_05_12_25.txt")
    ap.add_argument('--val', default="/bhupendra-data/Indic_OCR/Training_data/OdiaTrainingData/TrainingData_05_12_25/bhashiniVal_05_12_25.txt")
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--save_dir', type=str, default='checkpoints')
    ap.add_argument('--fixed_width', type=int, default=FIXED_WIDTH)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger, _ = setup_logger(args.save_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    vocab_paths = [p for p in [args.train, args.val] if p]
    char2id, id2char = build_vocab(vocab_paths)
    blank_id = len(char2id)
    id2char[blank_id] = BLANK_TOKEN
    num_classes = len(char2id) + 1

    train_dl = DataLoader(OCRDataset(args.train, char2id, IMG_HEIGHT, args.fixed_width),
                          batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(OCRDataset(args.val, char2id, IMG_HEIGHT, args.fixed_width),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # model = ViT_CTC(1, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT, num_classes).to(device)
    model = ViT_CTC(2, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT, num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')

    best_wer, patience, no_improve = 1e9, 10, 0

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_dl, criterion, optimizer, scaler, blank_id, device, train=True, epoch=epoch, logger=logger)
        logger.info(f"[Epoch {epoch}] Train Loss: {tr_loss:.4f}")
        torch.cuda.empty_cache()

        metrics = validate(model, val_dl, blank_id, id2char, device, logger, beam_width=BEAM_WIDTH)
        logger.info(f"[Epoch {epoch}] CER={metrics['CER']:.4f}, WER={metrics['WER']:.4f}, CRR={metrics['CRR']:.4f}, WRR={metrics['WRR']:.4f}")

        if metrics['WER'] < best_wer:
            best_wer = metrics['WER']
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"best_model_epoch{epoch}.pt"))
            logger.info(f"✅ Saved best model (WER={best_wer:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping due to no improvement.")
                break

if __name__ == '__main__':
    main()
