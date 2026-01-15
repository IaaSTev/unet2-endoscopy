import torch

@torch.no_grad()
def band_to_bbox(band_hw: torch.Tensor, pad: int = 16):
    """
    band_hw: (H,W) 0/1 tensor (CPU or GPU)
    return: (y1, y2, x1, x2) with y2/x2 exclusive, or None if band empty
    """
    if band_hw.ndim != 2:
        raise ValueError(f"band_hw must be (H,W), got {band_hw.shape}")

    ys, xs = torch.where(band_hw > 0.5)
    if ys.numel() == 0:
        return None

    H, W = band_hw.shape
    y1 = int(torch.min(ys).item())
    y2 = int(torch.max(ys).item()) + 1
    x1 = int(torch.min(xs).item())
    x2 = int(torch.max(xs).item()) + 1

    y1 = max(0, y1 - pad)
    x1 = max(0, x1 - pad)
    y2 = min(H, y2 + pad)
    x2 = min(W, x2 + pad)

    return y1, y2, x1, x2


@torch.no_grad()
def crop_by_bbox(img_b1hw: torch.Tensor, bbox):
    """
    img_b1hw: (B,1,H,W)
    bbox: (y1,y2,x1,x2)
    return: cropped patch (B,1,ph,pw)
    """
    y1, y2, x1, x2 = bbox
    return img_b1hw[:, :, y1:y2, x1:x2]
