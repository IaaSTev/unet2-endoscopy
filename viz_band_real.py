"生成真实mask的边界带的可视化"

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# 你已有的函数：从 logits 做 band
from boundary import make_boundary_band_from_logits

def save_overlay(img2d, band2d, out_path, title=""):
    """
    img2d: (H,W) float in [0,1]
    band2d: (H,W) {0,1}
    """
    plt.figure()
    plt.imshow(img2d, cmap="gray")
    # 叠加边界带（用透明度，不指定颜色也行；默认会用colormap）
    plt.imshow(band2d, alpha=0.5)
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def load_gray(path, size=256):
    img = Image.open(path).convert("L").resize((size, size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (H,W) float

def load_mask01(path, size=256):
    m = Image.open(path).convert("L").resize((size, size), resample=Image.NEAREST)
    arr = (np.array(m, dtype=np.uint8) > 0).astype(np.float32)
    return arr  # (H,W) {0,1}

def main():
    # ✅ 改成你想看的某一张样本
    img_path  = "/Users/katiexie/Desktop/unet2/Dataset_BUSI_with_GT/malignant/malignant (20).png"
    mask_path = "/Users/katiexie/Desktop/unet2/Dataset_BUSI_with_GT/malignant/malignant (20)_mask.png"

    size = 256
    img = load_gray(img_path, size=size)
    mask01 = load_mask01(mask_path, size=size)

    # ✅ 用“mask 当作 logits”来复用你已有的 make_boundary_band_from_logits
    # 让 mask01 ∈ {0,1} 映射成 logits：1 -> +10, 0 -> -10 （非常稳）
    logits = torch.from_numpy(mask01)[None, None, ...] * 20.0 - 10.0  # (1,1,H,W)

    band = make_boundary_band_from_logits(logits, thr=0.5, k=7)[0, 0].numpy().astype(np.float32)

    os.makedirs("roi_debug", exist_ok=True)
    save_overlay(img, band, "roi_debug/band_overlay.png", title="Boundary band overlay (GT mask, k=7)")
    print("Saved:", os.path.abspath("roi_debug/band_overlay.png"))

if __name__ == "__main__":
    main()