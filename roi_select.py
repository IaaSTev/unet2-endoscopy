# roi_select.py
import re
import argparse
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


from data_preparation.seg_dataset import build_pairs
from stage_1.unet import AttentionUNet
from stage_1.roi_tiling import roi_to_boxes, save_patches, sanitize_stem
from data_preparation.seg_dataset import build_pairs, SegDataset

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------------------
# IO
# ----------------------------
def load_gray_np(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr  # (H,W) float[0,1]


def load_mask_np(path: str, size: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    if size is not None:
        m = m.resize((size, size), resample=Image.NEAREST)
    arr = np.array(m, dtype=np.uint8)
    arr = (arr > 0).astype(np.uint8)  # {0,1}
    return arr


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def stem_from_path(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0]
    # remove weird chars (BUSI has parentheses)
    return sanitize_stem(base)


# ----------------------------
# Stage1 inference
# ----------------------------
@torch.no_grad()
def infer_prob_stage1(
    model: torch.nn.Module,
    img_hw: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    img_hw: (H,W) float[0,1]
    returns prob_hw: (H,W) float[0,1]
    """
    x = torch.from_numpy(img_hw).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    logits = model(x)  # (1,1,H,W)
    prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    return prob


def load_stage1_model(ckpt: str, base: int, device: str) -> torch.nn.Module:
    model = AttentionUNet(in_channels=1, out_channels=1, base=base).to(device)
    sd = torch.load(ckpt, map_location=device)
    # support {"model": state_dict} or pure state_dict
    if isinstance(sd, dict) and "model" in sd:
        model.load_state_dict(sd["model"], strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


# ----------------------------
# ROI: band (boundary band) from prob
# ----------------------------
def _dilate_bin(mask: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    return (F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad) > 0.5).float()


def _erode_bin(mask: torch.Tensor, k: int) -> torch.Tensor:
    # erode(m) = 1 - dilate(1-m)
    inv = 1.0 - mask
    dil = _dilate_bin(inv, k)
    return (1.0 - dil).clamp(0, 1)


@torch.no_grad()
def roi_band_from_prob(
    prob_hw: np.ndarray,
    thr: float = 0.5,
    bd_k: int = 3,
    band_k: int = 11,
) -> np.ndarray:
    """
    1) hard mask = prob>thr
    2) boundary = dilate - erode
    3) band = dilate(boundary, band_k)
    returns roi_hw {0,1}
    """
    p = torch.from_numpy(prob_hw).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    hard = (p > thr).float()
    dil = _dilate_bin(hard, bd_k)
    ero = _erode_bin(hard, bd_k)
    bd = (dil - ero).clamp(0, 1)
    band = _dilate_bin((bd > 0.5).float(), band_k)
    return band[0, 0].cpu().numpy().astype(np.uint8)


# ----------------------------
# ROI: RefineSeg-style uncertainty from prob
# ----------------------------
def roi_refineseg_from_prob(
    prob_hw: np.ndarray,
    t_bg: float = 0.2,
    t_fg: float = 0.8,
) -> np.ndarray:
    """
    RefineSeg-style:
      confident bg: p<=t_bg
      confident fg: p>=t_fg
      uncertain U:  t_bg < p < t_fg
    return U mask {0,1}
    """
    U = ((prob_hw > t_bg) & (prob_hw < t_fg)).astype(np.uint8)
    return U


# ----------------------------
# Save helpers
# ----------------------------
def save_mask_png(mask_hw01: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    m = (mask_hw01 > 0).astype(np.uint8) * 255
    Image.fromarray(m).save(path)


def save_prob_png(prob_hw: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = np.clip(prob_hw * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(p).save(path)


def save_boxes_npy(boxes, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.array(boxes, dtype=np.int32))


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # data / io
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset root OR a class folder. build_pairs will recursively find png.")
    parser.add_argument("--outdir", type=str, default="stage1_roi_out",
                        help="Output dir for prob/roi/boxes/patches.")
    parser.add_argument("--size", type=int, default=256)

    # stage1
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_base", type=int, default=32)

    # ROI mode
    parser.add_argument("--roi_mode", type=str, default="band",
                        choices=["band", "refineseg"])
    # band params
    parser.add_argument("--prob_thr", type=float, default=0.5,
                        help="prob threshold to form hard mask for boundary band.")
    parser.add_argument("--band_k", type=int, default=11,
                        help="band dilation kernel size (odd). Bigger -> wider band.")
    parser.add_argument("--bd_k", type=int, default=3,
                        help="boundary morph kernel size (odd).")

    # refineseg params
    parser.add_argument("--t_bg", type=float, default=0.2)
    parser.add_argument("--t_fg", type=float, default=0.8)

    # tiling params
    parser.add_argument("--patch", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--min_roi_pixels", type=int, default=200)

    # export patches for stage2
    parser.add_argument("--export_patches", type=int, default=1,
                        help="1: crop patches to disk (recommended for stage2). 0: only save boxes.")
    parser.add_argument("--save_png", type=int, default=1)
    parser.add_argument("--save_npy", type=int, default=0)

    # limit for quick test
    parser.add_argument("--max_items", type=int, default=-1,
                        help=">0 to only process first N samples for quick debug.")

    args = parser.parse_args()

    device = pick_device()
    print("device:", device)

    # load stage1 model
    model = load_stage1_model(args.stage1_ckpt, base=args.stage1_base, device=device)

    # collect pairs
    pairs = build_pairs(args.data_dir)
    if len(pairs) == 0:
        raise RuntimeError(f"No pairs found in {args.data_dir}. Check naming rules.")
    print("num_pairs:", len(pairs))

    if args.max_items > 0:
        pairs = pairs[:args.max_items]
        print("use max_items:", len(pairs))

    # output dirs
    prob_dir = os.path.join(args.outdir, "prob")
    roi_dir  = os.path.join(args.outdir, f"roi_{args.roi_mode}")
    box_dir  = os.path.join(args.outdir, f"boxes_{args.roi_mode}")
    patch_dir = os.path.join(args.outdir, f"patches_{args.roi_mode}")

    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)
    if args.export_patches == 1:
        os.makedirs(patch_dir, exist_ok=True)

    kept_total = 0

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        stem = stem_from_path(img_path)

        img_hw = load_gray_np(img_path, args.size)   # float[0,1]
        mask_hw = load_mask_np(mask_path, args.size) # {0,1}

        # stage1 prob
        prob_hw = infer_prob_stage1(model, img_hw, device=device)

        # ROI select
        if args.roi_mode == "band":
            roi_hw = roi_band_from_prob(
                prob_hw,
                thr=args.prob_thr,
                bd_k=args.bd_k,
                band_k=args.band_k
            )
        else:
            roi_hw = roi_refineseg_from_prob(
                prob_hw,
                t_bg=args.t_bg,
                t_fg=args.t_fg
            )

        # tiling -> boxes
        boxes = roi_to_boxes(
            roi_hw.astype(np.float32),
            patch=args.patch,
            stride=args.stride,
            min_roi_pixels=args.min_roi_pixels,
            roi_thr=0.5
        )

        kept_total += len(boxes)

        # save prob / roi / boxes
        save_prob_png(prob_hw, os.path.join(prob_dir, f"{stem}.png"))
        save_mask_png(roi_hw, os.path.join(roi_dir, f"{stem}.png"))
        save_boxes_npy(boxes, os.path.join(box_dir, f"{stem}.npy"))

        # optionally export cropped patches (for stage2)
        if args.export_patches == 1 and len(boxes) > 0:
            save_patches(
                outdir=patch_dir,
                stem=stem,
                img_hw=img_hw,     # float[0,1]
                mask_hw=mask_hw,   # {0,1}
                boxes=boxes,
                save_png=bool(args.save_png),
                save_npy=bool(args.save_npy),
            )

        if i % 50 == 0 or i == len(pairs):
            print(f"[{i}/{len(pairs)}] stem={stem} kept_boxes={len(boxes)}")

    print("Done.")
    print("outdir:", os.path.abspath(args.outdir))
    print("total_kept_patches:", kept_total)


if __name__ == "__main__":
    main()