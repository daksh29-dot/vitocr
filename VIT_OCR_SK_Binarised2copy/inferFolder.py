import os
import json
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import cv2
import numpy as np

IMG_HEIGHT = 32
FIXED_WIDTH = 256
PATCH_SIZE = (4, 4)
EMBED_DIM = 384
NUM_HEADS = 6
MLP_DIM = 1024
NUM_LAYERS = 8
DROPOUT = 0.1
MAX_PATCHES = 1024
BEAM_WIDTH = 32

MODEL_PATH = "/bhupendra-data/Indic_OCR/VIT_OCR_SK/checkpoints_Odia_11_02_26/best_model_epoch47.pt"                                                                             # Change model path
LABELMAP_PATH = "/bhupendra-data/Indic_OCR/VIT_OCR_SK/checkpoints_Odia_11_02_26/label_map.json"                                                                                 # Change label map file path
IMAGE_FOLDER = "/bhupendra-data/Indic_OCR/VIT_OCR_SK/test/"  # Change input test image folder
OUTPUT_EXCEL = "/bhupendra-data/Indic_OCR/VIT_OCR_SK/test_2.xlsx"        # Change output excel sheet path

#DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") change 1
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
def preprocess_image(img_path, img_height=IMG_HEIGHT, fixed_width=FIXED_WIDTH):
    img = Image.open(img_path).convert('L')
    w, h = img.size
    new_h = img_height
    new_w = int(w * (new_h / h))
    new_w = max(1, new_w)
    if new_w > fixed_width:
        new_w = fixed_width
    img = img.resize((new_w, new_h), Image.BILINEAR)
    if new_w < fixed_width:
        pad_width = fixed_width - new_w
        img = ImageOps.expand(img, border=(0, 0, pad_width, 0), fill=255)
        
    img_np = np.array(img)
    # img_np = cv2.GaussianBlur(img_np, (3, 3), 3)
    
    # save_dir = "/bhupendra-data/Indic_OCR/VIT_OCR_SK/test_img_op"
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)

    #     filename = os.path.basename(img_path)  # keeps original name
    #     save_path = os.path.join(save_dir, filename)

        # cv2.imwrite(save_path, img_np)
    
    # arr = np.array(img).astype(np.float32) / 255.0
    # arr = (arr - 0.5) / 0.5
    # arr = arr[np.newaxis, :, :]
    # return torch.from_numpy(arr).unsqueeze(0)
    orig = img_np.astype(np.float32) / 255.0
    orig = (orig - 0.5) / 0.5
    _, otsu = cv2.threshold(
    img_np,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = otsu.astype(np.float32) / 255.0
    otsu = (otsu - 0.5) / 0.5
    arr = np.stack([orig, otsu], axis=0)
    return torch.from_numpy(arr).unsqueeze(0)
#change 2 
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

def ctc_beam_search_decode_gpu(logits, id2char, blank_id, beam_width=32):
    B, T, C = logits.shape
    preds = []
    for b in range(B):
        log_probs = F.log_softmax(logits[b], dim=-1)
        seq = torch.argmax(log_probs, dim=-1).tolist()
        out = []
        prev = None
        for s in seq:
            if s != prev and s != blank_id:
                out.append(id2char.get(str(s), ""))
            prev = s
        preds.append("".join(out))
    return preds

def main():
    print("...Starting inference...")
    device = DEVICE


    with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    char2id = label_map["char2id"]
    id2char = label_map["id2char"]

    blank_id = len(char2id) - 1
    num_classes = len(char2id)
    
    # model = ViT_CTC(1, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT, num_classes).to(device)
    #change 3 2 channel output was 1
    model = ViT_CTC(2, PATCH_SIZE, EMBED_DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, DROPOUT, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    image_files = sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ])

    results = []
    #with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
    #change 4 improve speed 
    with torch.inference_mode(), torch.amp.autocast('cuda',enabled=(device.type == 'cuda')):
       for img_path in tqdm(image_files, desc="Predicting"):
            img_tensor = preprocess_image(img_path).to(device)
            logits = model(img_tensor)
            preds = ctc_beam_search_decode_gpu(logits, id2char, blank_id, beam_width=BEAM_WIDTH)
            results.append({"Image Path": img_path, "Prediction": preds[0]})

    df = pd.DataFrame(results)
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"Predictions saved to: {OUTPUT_EXCEL}")

if __name__ == "__main__":
    main()
