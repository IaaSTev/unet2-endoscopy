import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

DATA_ROOT = "/content/drive/MyDrive/datasets/Dataset_BUSI_with_GT"

import os, glob, re

def build_pairs(root="/content/drive/MyDrive/datasets/Dataset_BUSI_with_GT"):
    root = os.path.expanduser(root)
    assert os.path.exists(root), f"root not exist: {root}"

    all_png = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))

    pairs = []
    for img in all_png:
        base, ext = os.path.splitext(img)
        fname = os.path.basename(base).lower()

        # 跳过 mask 文件本身（避免把 mask 当 image）
        if "_mask" in fname:
            continue

        # 1) 优先找 xxx_mask.png
        mask0 = base + "_mask" + ext
        if os.path.exists(mask0):
            pairs.append((img, mask0))
            continue

        # 2) 找 xxx_mask_*.png（例如 _mask_1）
        cand = glob.glob(base + "_mask_*" + ext)
        if len(cand) > 0:
            # 选编号最小的（_mask_1 最先）
            def mask_index(p):
                m = re.search(r"_mask_(\d+)\.png$", p.lower())
                return int(m.group(1)) if m else 10**9
            cand_sorted = sorted(cand, key=mask_index)
            pairs.append((img, cand_sorted[0]))
            continue

        # 找不到就跳过
        # print("[WARN] no mask for", img)

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