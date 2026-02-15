import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from boundary import make_boundary_band_from_prob
from seg_dataset import build_pairs, SegDataset


# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Metrics (Dice)
# =========================
@torch.no_grad()
def dice_from_logits(logits, y, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * y).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean().item()


# =========================
# Boundary helpers (you already use these)
# =========================
@torch.no_grad()
def gt_boundary_from_mask(y: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    y: (B,1,H,W) in {0,1}
    return: gt boundary (B,1,H,W) in {0,1}
    Use morph gradient: dilate(y) - erode(y)
    """
    y = (y > 0.5).float()
    pad = k // 2
    dil = F.max_pool2d(y, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - y, kernel_size=k, stride=1, padding=pad)
    bd = (dil - ero).clamp(0, 1)
    return (bd > 0.5).float()


@torch.no_grad()
def dilate_mask(m: torch.Tensor, r: int) -> torch.Tensor:
    """
    m: (B,1,H,W) 0/1
    dilate radius r -> kernel size (2r+1)
    """
    if r <= 0:
        return (m > 0.5).float()
    k = 2 * r + 1
    pad = k // 2
    return (F.max_pool2d((m > 0.5).float(), kernel_size=k, stride=1, padding=pad) > 0.5).float()


@torch.no_grad()
def boundary_coverage(gt_bd: torch.Tensor, pred_band: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    coverage per-sample: |B_gt ∩ Band_pred| / |B_gt|
    returns (B,)
    """
    gt_bd = (gt_bd > 0.5).float()
    pred_band = (pred_band > 0.5).float()
    inter = (gt_bd * pred_band).sum(dim=(1,2,3))
    denom = gt_bd.sum(dim=(1,2,3)).clamp_min(eps)
    return inter / denom


@torch.no_grad()
def band_purity(pred_band: torch.Tensor, gt_bd: torch.Tensor, near_r: int = 3, eps: float = 1e-6) -> torch.Tensor:
    """
    purity per-sample: |Band_pred ∩ dilate(B_gt, r)| / |Band_pred|
    returns (B,)
    """
    pred_band = (pred_band > 0.5).float()
    gt_near = dilate_mask(gt_bd, r=near_r)
    inter = (pred_band * gt_near).sum(dim=(1,2,3))
    denom = pred_band.sum(dim=(1,2,3)).clamp_min(eps)
    return inter / denom


@torch.no_grad()
def sample_centers_from_band(pred_band_hw: torch.Tensor, num_patches: int, rng: torch.Generator) -> torch.Tensor:
    """
    pred_band_hw: (H,W) 0/1
    return centers: (K,2) [cy,cx]
    """
    ys, xs = torch.where(pred_band_hw > 0.5)
    if ys.numel() == 0:
        return torch.empty((0,2), dtype=torch.long, device=pred_band_hw.device)
    idx = torch.randint(0, ys.numel(), (num_patches,), generator=rng, device=pred_band_hw.device)
    return torch.stack([ys[idx], xs[idx]], dim=1)


@torch.no_grad()
def count_boundary_in_patch(gt_bd_hw: torch.Tensor, cy: int, cx: int, ph: int, pw: int) -> int:
    """
    gt_bd_hw: (H,W) 0/1
    """
    H, W = gt_bd_hw.shape
    y0 = max(0, cy - ph // 2)
    x0 = max(0, cx - pw // 2)
    y1 = min(H, y0 + ph)
    x1 = min(W, x0 + pw)
    # shift back if clipped
    y0 = max(0, y1 - ph)
    x0 = max(0, x1 - pw)
    return int(gt_bd_hw[y0:y1, x0:x1].sum().item())


@torch.no_grad()
def patch_hit_stats(gt_bd: torch.Tensor,
                    pred_band: torch.Tensor,
                    num_patches: int = 200,
                    patch_hw=(128,128),
                    hit_pixel_thresh: int = 50,
                    hit_ratio_thresh: float = 0.005,
                    seed: int = 0) -> tuple[float, float]:
    """
    Compute hit_rate + avg boundary pixels per patch over a batch.
    Hit if boundary_pixels > hit_pixel_thresh OR boundary_pixels/area > hit_ratio_thresh
    Returns: (hit_rate, avg_boundary_pixels_per_patch) aggregated across batch
    """
    B = gt_bd.shape[0]
    ph, pw = patch_hw
    area = ph * pw

    rng = torch.Generator(device=gt_bd.device)
    rng.manual_seed(seed)

    total_hits = 0
    total_patches = 0
    total_bd_pixels = 0

    for b in range(B):
        gt_hw = (gt_bd[b,0] > 0.5).float()
        band_hw = (pred_band[b,0] > 0.5).float()
        centers = sample_centers_from_band(band_hw, num_patches=num_patches, rng=rng)
        if centers.numel() == 0:
            continue
        for cy, cx in centers:
            bd_pixels = count_boundary_in_patch(gt_hw, int(cy.item()), int(cx.item()), ph, pw)
            total_bd_pixels += bd_pixels
            total_patches += 1
            if (bd_pixels > hit_pixel_thresh) or (bd_pixels / area > hit_ratio_thresh):
                total_hits += 1

    if total_patches == 0:
        return 0.0, 0.0
    hit_rate = total_hits / total_patches
    avg_bd = total_bd_pixels / total_patches
    return float(hit_rate), float(avg_bd)


# =========================
# Loss: BCE + SoftDice + BoundaryBandBCE
# =========================
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, y):
        p = torch.sigmoid(logits)
        p = p.view(p.size(0), -1)
        y = y.view(y.size(0), -1)
        inter = (p * y).sum(dim=1)
        union = p.sum(dim=1) + y.sum(dim=1)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class BoundaryBandBCELoss(nn.Module):
    """
    只在 GT 边界附近的 band 上计算 BCE，让边界学得更准。
    band_k 越大，边界带越宽（更稳但更“粗”）；太小可能不稳定。
    """
    def __init__(self, band_k=11):
        super().__init__()
        self.band_k = band_k
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, y):
        gt_bd = gt_boundary_from_mask(y, k=3)  # (B,1,H,W)
        pad = self.band_k // 2
        band = F.max_pool2d(gt_bd, kernel_size=self.band_k, stride=1, padding=pad)  # 0/1

        loss_map = self.bce(logits, y)  # (B,1,H,W)
        denom = band.sum().clamp_min(1.0)
        return (loss_map * band).sum() / denom


class ComboLoss(nn.Module):
    def __init__(self, w_bce=1.0, w_dice=1.0, w_band=0.7, band_k=11):
        super().__init__()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_band = w_band
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss()
        self.band = BoundaryBandBCELoss(band_k=band_k)

    def forward(self, logits, y):
        return (self.w_bce * self.bce(logits, y) +
                self.w_dice * self.dice(logits, y) +
                self.w_band * self.band(logits, y))


# =========================
# Attention U-Net (inline, no extra file needed)
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate(nn.Module):
    """
    x: skip feature (encoder), (B,Cx,H,W)
    g: gating feature (decoder), (B,Cg,h,w) -> resize to (H,W)
    """
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.theta_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        # 小 batch 可能不稳：如果你 batch 很小（<=2），可删掉 BatchNorm2d(1)
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        f = self.relu(self.theta_x(x) + self.phi_g(g))
        a = self.psi(f)  # (B,1,H,W)
        return x * a


class UpAttn(nn.Module):
    def __init__(self, skip_ch, dec_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        inter_ch = max(out_ch // 2, 1)
        self.attn = AttentionGate(x_ch=skip_ch, g_ch=dec_ch, inter_ch=inter_ch)
        self.conv = DoubleConv(skip_ch + dec_ch, out_ch)

    def forward(self, dec, skip):
        dec = self.up(dec)
        diff_y = skip.size(2) - dec.size(2)
        diff_x = skip.size(3) - dec.size(3)
        dec = F.pad(dec, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        skip_att = self.attn(skip, dec)
        x = torch.cat([skip_att, dec], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)

        self.up1 = UpAttn(skip_ch=base * 8,  dec_ch=base * 16, out_ch=base * 8)
        self.up2 = UpAttn(skip_ch=base * 4,  dec_ch=base * 8,  out_ch=base * 4)
        self.up3 = UpAttn(skip_ch=base * 2,  dec_ch=base * 4,  out_ch=base * 2)
        self.up4 = UpAttn(skip_ch=base,      dec_ch=base * 2,  out_ch=base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)


# =========================
# Train / Validate
# =========================
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0

    # ROI quality accumulators
    cov_sum, pur_sum, cov_n, pur_n = 0.0, 0.0, 0, 0
    hit_sum, avgB_sum, hit_n = 0.0, 0.0, 0

    band_ratio_sum, gt_bd_ratio_sum, ratio_n = 0.0, 0.0, 0

    # ROI params
    band_r = 5
    band_k = 2 * band_r + 1
    near_r = 3
    num_patches = 200
    patch_hw = (64, 64)

    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)

        total_loss += loss_fn(logits, y).item()
        total_dice += dice_from_logits(logits, y)

        # ROI quality part
        prob = torch.sigmoid(logits)  # (B,1,H,W)
        pred_band = make_boundary_band_from_prob(prob, thr=0.4, k=band_k)  # (B,1,H,W) {0,1}
        gt_bd = gt_boundary_from_mask(y, k=3)  # (B,1,H,W) {0,1}

        B, _, H, W = pred_band.shape
        band_ratio = float(pred_band.sum().item()) / (B * H * W)
        gt_bd_ratio = float(gt_bd.sum().item()) / (B * H * W)
        band_ratio_sum += band_ratio
        gt_bd_ratio_sum += gt_bd_ratio
        ratio_n += 1

        cov_b = boundary_coverage(gt_bd, pred_band)  # (B,)
        pur_b = band_purity(pred_band, gt_bd, near_r=near_r)  # (B,)

        cov_sum += float(cov_b.sum().item())
        pur_sum += float(pur_b.sum().item())
        cov_n += int(cov_b.numel())
        pur_n += int(pur_b.numel())

        hit_rate, avgB = patch_hit_stats(
            gt_bd=gt_bd,
            pred_band=pred_band,
            num_patches=num_patches,
            patch_hw=patch_hw,
            hit_pixel_thresh=300,
            hit_ratio_thresh=0.02,
            seed=1234 + it
        )
        hit_sum += hit_rate
        avgB_sum += avgB
        hit_n += 1

    n = max(len(loader), 1)
    val_loss = total_loss / n
    val_dice = total_dice / n

    coverage = cov_sum / max(cov_n, 1)
    purity = pur_sum / max(pur_n, 1)
    hit_rate = hit_sum / max(hit_n, 1)
    avg_boundary_pixels_per_patch = avgB_sum / max(hit_n, 1)

    roi_metrics = {
        "coverage": coverage,
        "purity": purity,
        "hit_rate": hit_rate,
        "avg_boundary_pixels_per_patch": avg_boundary_pixels_per_patch,
        "band_ratio": band_ratio_sum / max(ratio_n, 1),
        "gt_bd_ratio": gt_bd_ratio_sum / max(ratio_n, 1),
    }

    return val_loss, val_dice, roi_metrics


# =========================
# Main
# =========================
def main():
    set_seed(42)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    DATA_ROOT = "/content/drive/MyDrive/datasets/Dataset_BUSI_with_GT"
    pairs = build_pairs(DATA_ROOT)

    print("num_pairs:", len(pairs))
    assert len(pairs) > 0, "没找到任何 image+mask 配对。检查文件名是否是 xxx.png 和 xxx_mask.png"

    idx = np.random.RandomState(42).permutation(len(pairs))
    split = int(0.8 * len(pairs))
    train_pairs = [pairs[i] for i in idx[:split]]
    val_pairs   = [pairs[i] for i in idx[split:]]

    train_ds = SegDataset(train_pairs, size=256, augment=True)
    val_ds   = SegDataset(val_pairs, size=256, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # ✅ model: Attention U-Net
    model = AttentionUNet(in_channels=1, out_channels=1, base=32).to(device)

    # ✅ loss: boundary-prioritized combo loss
    # 起步推荐：w_band=0.7, band_k=11
    loss_fn = ComboLoss(w_bce=1.0, w_dice=1.0, w_band=0.3, band_k=11)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    best_dice = -1.0
    last_epoch = 0

    for epoch in range(1, 31):
        last_epoch = epoch

        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_dice, roi = validate(model, val_loader, loss_fn, device)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}")
        print("ROI metrics:",
              f"coverage={roi['coverage']:.3f},",
              f"purity={roi['purity']:.3f},",
              f"hit_rate={roi['hit_rate']:.3f},",
              f"avgB={roi['avg_boundary_pixels_per_patch']:.1f},",
              f"band_ratio={roi['band_ratio']:.3f},",
              f"gt_bd_ratio={roi['gt_bd_ratio']:.3f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "best_dice": best_dice},
                "checkpoints/best.pt"
            )

    torch.save(
        {"epoch": last_epoch, "model": model.state_dict(), "best_dice": best_dice},
        "checkpoints/last.pt"
    )
    print("Done. best_dice =", best_dice)


if __name__ == "__main__":
    main()
