# stage_1/roi_viz.py
import os, sys, argparse
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F

# ---- make imports work no matter where you run it ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stage_1.unet import AttentionUNet
from stage_1.uncertainty import (
    make_boundary_band_from_prob,
    make_refineseg_uncertainty_from_prob,
)
from stage_1.roi_tiling import roi_to_boxes, sanitize_stem


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_gray_np(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize((size, size), resample=Image.BILINEAR)
    return (np.array(img, dtype=np.float32) / 255.0)


def load_mask_np(path: str, size: int) -> np.ndarray:
    m = Image.open(path).convert("L")
    if size is not None:
        m = m.resize((size, size), resample=Image.NEAREST)
    return (np.array(m, dtype=np.uint8) > 0).astype(np.uint8)


def save_u8_gray(arr01: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    u8 = np.clip(arr01 * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def save_u8_mask(mask01: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    u8 = (mask01 > 0).astype(np.uint8) * 255
    Image.fromarray(u8, mode="L").save(path)


def model_load(ckpt: str, base: int, device: str):
    model = AttentionUNet(in_channels=1, out_channels=1, base=base).to(device)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        model.load_state_dict(sd["model"], strict=True)
    else:
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_prob(model, img_hw: np.ndarray, device: str) -> torch.Tensor:
    x = torch.from_numpy(img_hw).float().unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
    logits = model(x)
    prob = torch.sigmoid(logits)  # (1,1,H,W)
    return prob


def mask_to_outline(mask_hw01: np.ndarray) -> np.ndarray:
    # 形态学梯度：dilate - erode
    m = torch.from_numpy(mask_hw01.astype(np.float32))[None, None]  # (1,1,H,W)
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    edge = (dil - ero).clamp(0, 1)[0, 0].numpy()
    return (edge > 0.5).astype(np.uint8)


def make_overlay(
    img_hw: np.ndarray,
    roi_hw01: np.ndarray,
    boxes,
    gt_mask_hw01: np.ndarray | None = None,
    alpha: float = 0.45,
) -> Image.Image:
    """
    overlay: ROI 红色半透明; GT 轮廓绿色; boxes 黄色框
    """
    H, W = img_hw.shape
    base = (np.clip(img_hw, 0, 1) * 255).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)  # (H,W,3)

    # ROI 红色热区
    roi = (roi_hw01 > 0).astype(np.float32)
    red = rgb.copy().astype(np.float32)
    red[..., 0] = 255.0
    rgb = ((1 - alpha * roi[..., None]) * rgb.astype(np.float32) + (alpha * roi[..., None]) * red).astype(np.uint8)

    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)

    # GT 轮廓绿色
    if gt_mask_hw01 is not None:
        edge = mask_to_outline(gt_mask_hw01)
        px = im.load()
        for y in range(H):
            for x in range(W):
                if edge[y, x] > 0:
                    px[x, y] = (0, 255, 0)

    # boxes 黄色框
    for (y1, y2, x1, x2) in boxes:
        draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 255, 0), width=2)

    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--mask", type=str, default=None)
    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--stage1_base", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)

    ap.add_argument("--roi_mode", type=str, default="band", choices=["band", "refineseg"])
    ap.add_argument("--thr", type=float, default=0.5, help="band: prob thr")
    ap.add_argument("--k", type=int, default=11, help="band: band width kernel (odd)")
    ap.add_argument("--t_bg", type=float, default=0.2, help="refineseg: bg thr")
    ap.add_argument("--t_fg", type=float, default=0.8, help="refineseg: fg thr")

    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--min_roi_pixels", type=int, default=200)

    ap.add_argument("--outdir", type=str, default="viz_roi_out")
    args = ap.parse_args()

    device = pick_device()
    print("device:", device)

    img_hw = load_gray_np(args.img, args.size)
    gt_hw = load_mask_np(args.mask, args.size) if args.mask else None

    model = model_load(args.stage1_ckpt, args.stage1_base, device)
    prob = infer_prob(model, img_hw, device=device)  # (1,1,H,W)

    # ROI
    if args.roi_mode == "band":
        roi = make_boundary_band_from_prob(prob, thr=args.thr, k=args.k)  # (1,1,H,W)
    else:
        roi = make_refineseg_uncertainty_from_prob(prob, t_bg=args.t_bg, t_fg=args.t_fg)

    prob_hw = prob[0, 0].detach().cpu().numpy()
    roi_hw = roi[0, 0].detach().cpu().numpy()
    roi_hw01 = (roi_hw > 0.5).astype(np.uint8)

    # boxes from ROI
    boxes = roi_to_boxes(
        roi_hw01.astype(np.float32),
        patch=args.patch,
        stride=args.stride,
        min_roi_pixels=args.min_roi_pixels,
        roi_thr=0.5,
    )

    # output naming
    stem = sanitize_stem(os.path.splitext(os.path.basename(args.img))[0])
    tag = f"{stem}_{args.roi_mode}"
    outdir = os.path.join(args.outdir, tag)
    os.makedirs(outdir, exist_ok=True)

    save_u8_gray(img_hw, os.path.join(outdir, "input.png"))
    save_u8_gray(prob_hw, os.path.join(outdir, "prob.png"))
    save_u8_mask(roi_hw01, os.path.join(outdir, "roi_mask.png"))

    overlay = make_overlay(img_hw, roi_hw01, boxes, gt_mask_hw01=gt_hw)
    overlay.save(os.path.join(outdir, "overlay.png"))

    # also save boxes
    np.save(os.path.join(outdir, "boxes.npy"), np.array(boxes, dtype=np.int32))

    print("saved to:", os.path.abspath(outdir))
    print("num_boxes:", len(boxes))


if __name__ == "__main__":
    main()