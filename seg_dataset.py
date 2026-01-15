import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

DATA_ROOT = "/Users/katiexie/Desktop/unet2/Dataset_BUSI_with_GT/malignant"

def build_pairs(root=DATA_ROOT):
    imgs = sorted(glob.glob(os.path.join(root, "*.png")))
    pairs = []
    for p in imgs:
        if p.endswith("_mask.png"):
            continue
        mask_p = p.replace(".png", "_mask.png")
        if os.path.exists(mask_p):
            pairs.append((p, mask_p))
    return pairs

class SegDataset(Dataset):
    def __init__(self, pairs, size=256):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        img = Image.open(img_path).convert("L").resize((self.size, self.size))
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), resample=Image.NEAREST)

        img = np.array(img, dtype=np.float32) / 255.0
        mask = (np.array(mask, dtype=np.uint8) > 0).astype(np.float32)

        img = torch.from_numpy(img)[None, ...]   # (1,H,W)
        mask = torch.from_numpy(mask)[None, ...] # (1,H,W)
        return img, mask