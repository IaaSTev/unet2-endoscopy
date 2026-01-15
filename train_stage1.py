import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seg_dataset import build_pairs, SegDataset
from unet import UNet

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def dice_from_logits(logits, y, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > 0.5).float()
    inter = (pred * y).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    return ((2*inter + eps) / (union + eps)).mean().item()

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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item()
        total_dice += dice_from_logits(logits, y)
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n

def main():
    set_seed(42)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("device:", device)

    pairs = build_pairs()
    print("num_pairs:", len(pairs))
    assert len(pairs) > 0, "没找到任何 image+mask 配对。检查文件名是否是 xxx.png 和 xxx_mask.png"

    # 随机划分（先跑通 baseline；后面你再固定 split）
    idx = np.random.RandomState(42).permutation(len(pairs))
    split = int(0.8 * len(pairs))
    train_pairs = [pairs[i] for i in idx[:split]]
    val_pairs   = [pairs[i] for i in idx[split:]]

    train_ds = SegDataset(train_pairs, size=256)
    val_ds   = SegDataset(val_pairs, size=256)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, out_channels=1, base=32).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("checkpoints", exist_ok=True)
    best_dice = -1.0

    for epoch in range(1, 11):  # 先跑 10
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
        va_loss, va_dice = validate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_dice={va_dice:.4f}")

        if va_dice > best_dice:
            best_dice = va_dice
            torch.save({"epoch": epoch, "model": model.state_dict(), "best_dice": best_dice},
                       "checkpoints/best.pt")

    torch.save({"epoch": epoch, "model": model.state_dict(), "best_dice": best_dice},
               "checkpoints/last.pt")
    print("Done. best_dice =", best_dice)

if __name__ == "__main__":
    main()