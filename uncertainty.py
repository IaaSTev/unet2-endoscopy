# stage_1/uncertainty.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def make_boundary_band_from_prob(prob: torch.Tensor, thr: float = 0.5, k: int = 11) -> torch.Tensor:
    """
    prob: (B,1,H,W) in [0,1]
    thr: threshold to binarize prob -> mask
    k: band width control (odd recommended). larger -> wider band.

    Returns:
      band: (B,1,H,W) in {0,1}
    """
    if prob.ndim != 4 or prob.size(1) != 1:
        raise ValueError(f"prob must be (B,1,H,W), got {tuple(prob.shape)}")

    # binarize
    m = (prob > thr).float()

    # morphological gradient to get boundary: dilate(m) - erode(m)
    # use 3x3 by default
    dil = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    ero = 1.0 - F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1)
    bd = (dil - ero).clamp(0, 1)

    # dilate boundary to form a band
    if k <= 1:
        band = (bd > 0.5).float()
    else:
        pad = k // 2
        band = F.max_pool2d((bd > 0.5).float(), kernel_size=k, stride=1, padding=pad)
        band = (band > 0.5).float()
    return band


@torch.no_grad()
def make_refineseg_uncertainty_from_prob(prob: torch.Tensor, t_bg: float = 0.1, t_fg: float = 0.9) -> torch.Tensor:
    """
    RefineSeg-style:
      confident BG: prob <= t_bg
      confident FG: prob >= t_fg
      uncertain:    t_bg < prob < t_fg

    prob: (B,1,H,W) in [0,1]
    Returns:
      U: (B,1,H,W) in {0,1}
    """
    if prob.ndim != 4 or prob.size(1) != 1:
        raise ValueError(f"prob must be (B,1,H,W), got {tuple(prob.shape)}")
    U = ((prob > t_bg) & (prob < t_fg)).float()
    return U