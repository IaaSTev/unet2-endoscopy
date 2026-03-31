# roi_tiling.py
import os
import json
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

# box format: (y1, y2, x1, x2)  -- y2/x2 are exclusive


def sanitize_stem(s: str) -> str:
    """Make a filesystem-friendly stem."""
    s = s.replace(" ", "_")
    s = s.replace("(", "").replace(")", "")
    s = s.replace("[", "").replace("]", "")
    s = s.replace("{", "").replace("}", "")
    s = s.replace(",", "_")
    return s


def roi_to_boxes(
    roi_hw: np.ndarray,
    patch: int = 128,
    stride: int = 64,
    min_roi_pixels: int = 200,
    roi_thr: float = 0.5,
) -> List[Tuple[int, int, int, int]]:
    """
    Convert ROI mask (H,W) to a list of patch boxes by sliding window.
    Keep a patch if it contains >= min_roi_pixels ROI pixels.

    roi_hw: float/bool array in [0,1] or {0,1}
    """
    assert roi_hw.ndim == 2, f"roi_hw must be (H,W), got {roi_hw.shape}"
    H, W = roi_hw.shape
    roi = (roi_hw >= roi_thr).astype(np.uint8)

    boxes: List[Tuple[int, int, int, int]] = []

    # Ensure we cover the image even if (H-patch) not divisible by stride
    ys = list(range(0, max(H - patch + 1, 1), stride))
    xs = list(range(0, max(W - patch + 1, 1), stride))
    if len(ys) == 0:
        ys = [0]
    if len(xs) == 0:
        xs = [0]
    if ys[-1] != H - patch:
        ys.append(max(H - patch, 0))
    if xs[-1] != W - patch:
        xs.append(max(W - patch, 0))

    for y1 in ys:
        y2 = y1 + patch
        for x1 in xs:
            x2 = x1 + patch
            # clip (just in case)
            y1c, y2c = max(0, y1), min(H, y2)
            x1c, x2c = max(0, x1), min(W, x2)

            roi_pixels = int(roi[y1c:y2c, x1c:x2c].sum())
            if roi_pixels >= min_roi_pixels:
                boxes.append((y1c, y2c, x1c, x2c))

    return boxes


def crop_patch(
    img_hw: np.ndarray,
    mask_hw: Optional[np.ndarray],
    box: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Crop image (and mask) by box.
    img_hw: (H,W) float [0,1] or uint8
    mask_hw: (H,W) {0,1} or uint8 (optional)
    """
    y1, y2, x1, x2 = box
    img_p = img_hw[y1:y2, x1:x2]
    if mask_hw is None:
        return img_p, None
    mask_p = mask_hw[y1:y2, x1:x2]
    return img_p, mask_p


def save_patches(
    outdir: str,
    stem: str,
    img_hw: np.ndarray,
    mask_hw: Optional[np.ndarray],
    boxes: List[Tuple[int, int, int, int]],
    save_png: bool = True,
    save_npy: bool = False,
) -> str:
    """
    Save cropped patches into:
      outdir/stem/
        meta.json
        img_000.png, mask_000.png ...
      Also saves boxes.npy for convenience.
    Returns the directory path for this sample.
    """
    os.makedirs(outdir, exist_ok=True)
    stem = sanitize_stem(stem)
    sample_dir = os.path.join(outdir, stem)
    os.makedirs(sample_dir, exist_ok=True)

    # Save boxes
    boxes_arr = np.array(boxes, dtype=np.int32)
    np.save(os.path.join(sample_dir, "boxes.npy"), boxes_arr)

    meta: Dict = {
        "stem": stem,
        "num_patches": int(len(boxes)),
        "boxes_format": "(y1,y2,x1,x2) y2/x2 exclusive",
        "save_png": bool(save_png),
        "save_npy": bool(save_npy),
    }

    # Normalize for saving
    if img_hw.dtype != np.uint8:
        img_u8 = np.clip(img_hw * 255.0, 0, 255).astype(np.uint8)
    else:
        img_u8 = img_hw

    mask_u8 = None
    if mask_hw is not None:
        if mask_hw.dtype != np.uint8:
            mask_u8 = (mask_hw > 0.5).astype(np.uint8) * 255
        else:
            # assume already 0/255 or 0/1
            mask_u8 = (mask_hw > 0).astype(np.uint8) * 255

    for i, box in enumerate(boxes):
        img_p, mask_p = crop_patch(img_u8, mask_u8, box)

        if save_png:
            Image.fromarray(img_p).save(os.path.join(sample_dir, f"img_{i:03d}.png"))
            if mask_p is not None:
                Image.fromarray(mask_p).save(os.path.join(sample_dir, f"mask_{i:03d}.png"))

        if save_npy:
            # float [0,1] for img, {0,1} for mask
            img_f = (img_p.astype(np.float32) / 255.0)[None, ...]  # (1,h,w)
            np.save(os.path.join(sample_dir, f"img_{i:03d}.npy"), img_f)
            if mask_p is not None:
                m = (mask_p > 0).astype(np.float32)[None, ...]
                np.save(os.path.join(sample_dir, f"mask_{i:03d}.npy"), m)

    with open(os.path.join(sample_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return sample_dir