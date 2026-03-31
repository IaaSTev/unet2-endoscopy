import os
import glob
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Albumentations
import albumentations as A
import cv2


DATA_ROOT = "/content/drive/MyDrive/datasets/Dataset_BUSI_with_GT"


def build_pairs(root="/content/drive/MyDrive/datasets/Dataset_BUSI_with_GT"):
    root = os.path.expanduser(root)
    assert os.path.exists(root), f"root not exist: {root}"

    all_png = sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))

    pairs = []
    for img in all_png:
        rel = img.lower()

        # ===== 保留 benign / malignant / normal =====
        if ("/benign/" not in rel) and ("/malignant/" not in rel) and ("/normal/" not in rel):
            continue

        base, ext = os.path.splitext(img)
        fname = os.path.basename(base).lower()

        # 跳过 mask 文件本身（避免把 mask 当 image）
        if "_mask" in fname:
            continue

        mask_paths = []

        # 1) 标准 mask: xxx_mask.png
        mask0 = base + "_mask" + ext
        if os.path.exists(mask0):
            mask_paths.append(mask0)

        # 2) 额外 mask: xxx_mask_*.png（例如 _mask_1, _mask_2）
        cand = glob.glob(base + "_mask_*" + ext)
        if len(cand) > 0:
            def mask_index(p):
                m = re.search(r"_mask_(\d+)\.png$", p.lower())
                return int(m.group(1)) if m else 10**9
            cand_sorted = sorted(cand, key=mask_index)
            mask_paths.extend(cand_sorted)

        # 去重并保持顺序
        mask_paths = list(dict.fromkeys(mask_paths))

        if len(mask_paths) > 0:
            pairs.append((img, mask_paths))

    return pairs


class SegDataset(Dataset):
    def __init__(self, pairs, size=256, augment=False):
        """
        pairs: list of (img_path, mask_path)
        size:  output resize size (H=W=size)
        augment: whether to apply training augmentations
        """
        self.pairs = pairs
        self.size = size
        self.augment = augment

        # ===== Train augmentations (stable for your albumentations version) =====
        self.train_tf = A.Compose([
            # ---- 强度类（推荐）----
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),

            # ---- 轻量几何类（mask 同步，填充为 0）----
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.10,
                rotate_limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.4
            ),
        ])

        # ===== Resize (mask must be nearest) =====
        self.resize_img = A.Resize(self.size, self.size, interpolation=cv2.INTER_LINEAR)
        self.resize_msk = A.Resize(self.size, self.size, interpolation=cv2.INTER_NEAREST)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_paths = self.pairs[idx]

        # 读原图（不在 PIL 里 resize，避免影响增强）
        img = Image.open(img_path).convert("L")
        img = np.array(img, dtype=np.uint8)   # (H,W)

        # 支持单个 mask 路径或多个 mask 路径；多个时做并集
        if isinstance(mask_paths, str):
            mask_paths = [mask_paths]

        mask = None
        for mp in mask_paths:
            m = Image.open(mp).convert("L")
            m = np.array(m, dtype=np.uint8)
            m = (m > 0).astype(np.uint8)
            if mask is None:
                mask = m
            else:
                mask = np.maximum(mask, m)

        if mask is None:
            raise ValueError(f"No valid masks found for image: {img_path}")

        if self.augment:
            out = self.train_tf(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        # resize（mask nearest）
        img = self.resize_img(image=img)["image"]
        mask = self.resize_msk(image=mask)["image"]

        # normalize / to float
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)  # 0/1

        # to torch (1,H,W)
        img = torch.from_numpy(img)[None, ...]
        mask = torch.from_numpy(mask)[None, ...]
        return img, mask
