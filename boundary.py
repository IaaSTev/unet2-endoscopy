import torch
import torch.nn.functional as F

@torch.no_grad()
def make_boundary_band_from_logits(logits: torch.Tensor, thr: float = 0.7, k: int = 7) -> torch.Tensor:
    """
    logits: (B,1,H,W) UNet raw output (before sigmoid)
    thr: threshold on probability
    k: kernel size for morphological ops (odd number recommended)
    return: band mask (B,1,H,W) in {0,1}
    """
    prob = torch.sigmoid(logits)
    return make_boundary_band_from_prob(prob, thr=thr, k=k)

@torch.no_grad()
def make_boundary_band_from_prob(prob: torch.Tensor, thr: float = 0.5, k: int = 7) -> torch.Tensor:
    """
    prob: (B,1,H,W) probability in [0,1]
    Implements:
      dilate(mask) via max_pool2d
      erode(mask) = 1 - dilate(1-mask)
      band = dilate - erode
    """
    if prob.ndim != 4:
        raise ValueError(f"prob must be (B,1,H,W), got {prob.shape}")

    mask = (prob >= thr).float()
    pad = k // 2

    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)

    band = (dil - ero).clamp(0, 1)
    return band