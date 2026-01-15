import torch
from unet import UNet
from boundary import make_boundary_band_from_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(in_channels=1, out_channels=1, base=32).to(device)

x = torch.randn(2, 1, 256, 256).to(device)
logits = model(x)

print("input:", x.shape)
print("output(logits):", logits.shape)

# boundary band
band = make_boundary_band_from_logits(logits, thr=0.5, k=7)
print("band:", band.shape, "band_sum:", band.sum().item())

# fake target and one training step
target = torch.randint(0, 2, (2, 1, 256, 256)).float().to(device)
loss = torch.nn.BCEWithLogitsLoss()(logits, target)
loss.backward()
print("loss:", loss.detach().item())

from roi import band_to_bbox, crop_by_bbox

band_hw = band[0, 0].detach().cpu()   # 取第一张图的 band (H,W)
bbox = band_to_bbox(band_hw, pad=16)
print("bbox from band:", bbox)

if bbox is not None:
    patch = crop_by_bbox(x.detach().cpu(), bbox)
    print("patch shape from x:", patch.shape)
else:
    print("band empty -> no bbox (expected if coarse mask is all zeros)")
