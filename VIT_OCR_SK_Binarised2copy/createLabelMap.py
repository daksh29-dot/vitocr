#!/usr/bin/env python3
"""
create_labelmap.py
Builds and saves a label map (character to ID mapping) from training/validation data.
"""

import os
import json

# ---------- Paths ----------
TRAIN_TXT = "/bhupendra-data/Indic_OCR/Training_data/OdiaTrainingData/TrainingData_05_12_25/bhashiniTrain_05_12_25.txt"
VAL_TXT   = "/bhupendra-data/Indic_OCR/Training_data/OdiaTrainingData/TrainingData_05_12_25/bhashiniVal_05_12_25.txt"
OUTPUT_JSON = "/bhupendra-data/Indic_OCR/VIT_OCR_SK_Binarised/checkpoints/label_map.json"

BLANK_TOKEN = "<BLANK>"

# ---------- Build Vocab ----------
def build_vocab(tsv_paths):
    counts = {}
    for p in tsv_paths:
        if not os.path.exists(p):
            print(f"⚠️ File not found: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if "\t" not in line:
                    continue
                _, txt = line.rstrip("\n").split("\t", 1)
                for ch in txt:
                    if ch.strip() == "":
                        continue
                    counts[ch] = counts.get(ch, 0) + 1
    chars = sorted(counts.keys())
    char2id = {ch: i for i, ch in enumerate(chars)}
    id2char = {i: ch for i, ch in enumerate(chars)}
    return char2id, id2char

def main():
    print("📖 Building label map...")
    char2id, id2char = build_vocab([TRAIN_TXT, VAL_TXT])
    blank_id = len(char2id)
    char2id[BLANK_TOKEN] = blank_id
    id2char[blank_id] = BLANK_TOKEN

    label_map = {
        "char2id": char2id,
        "id2char": id2char
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved label map to: {OUTPUT_JSON}")
    print(f"Total characters: {len(char2id)}")

if __name__ == "__main__":
    main()
