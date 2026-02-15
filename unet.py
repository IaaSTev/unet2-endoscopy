import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al. Attention U-Net style)
    x: encoder skip feature  (B, Cx, H, W)
    g: decoder gating feature (B, Cg, h, w)  -> will be resized to (H, W)
    return: gated skip feature (B, Cx, H, W)
    """
    def __init__(self, x_ch: int, g_ch: int, inter_ch: int):
        super().__init__()

        self.theta_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.phi_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )

        # psi maps fused features -> attention mask in [0, 1]
        # NOTE: if your batch size is very small (e.g., 1-2),
        # BatchNorm2d(1) can be unstable; you may remove it.
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # resize g to match x spatial size
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        f = self.relu(self.theta_x(x) + self.phi_g(g))
        a = self.psi(f)   # (B,1,H,W)
        return x * a      # (B,Cx,H,W)


class UpAttn(nn.Module):
    """
    Up-sampling block with Attention Gate on skip connection.
    Keeps your original: bilinear upsample + padding alignment + concat + DoubleConv.
    """
    def __init__(self, skip_ch: int, dec_ch: int, out_ch: int):
        """
        skip_ch: encoder skip channels (Cx)
        dec_ch:  decoder feature channels before upsample (Cg)
        out_ch:  output channels after DoubleConv
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # inter_ch: bottleneck inside attention gate
        # A common choice is out_ch//2 or skip_ch//2. Keep it moderate.
        inter_ch = max(out_ch // 2, 1)
        self.attn = AttentionGate(x_ch=skip_ch, g_ch=dec_ch, inter_ch=inter_ch)

        # concat channels = skip_ch + dec_ch
        self.conv = DoubleConv(skip_ch + dec_ch, out_ch)

    def forward(self, dec: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        dec: decoder feature from deeper layer (g in AG)
        skip: encoder skip feature (x in AG)
        """
        dec = self.up(dec)

        # pad to match spatial sizes (your original logic)
        diff_y = skip.size(2) - dec.size(2)
        diff_x = skip.size(3) - dec.size(3)
        dec = F.pad(dec, [diff_x // 2, diff_x - diff_x // 2,
                          diff_y // 2, diff_y - diff_y // 2])

        # attention on skip connection using dec as gating signal
        skip_att = self.attn(skip, dec)

        # concat and convolve
        x = torch.cat([skip_att, dec], dim=1)
        return self.conv(x)


class AttentionUNet(nn.Module):
    """
    Attention U-Net based on your original UNet skeleton:
    - 4 downs
    - 4 ups
    - Attention Gates on each skip
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)

        # channel plan:
        # x1: base
        # x2: base*2
        # x3: base*4
        # x4: base*8
        # x5: base*16
        self.up1 = UpAttn(skip_ch=base * 8,  dec_ch=base * 16, out_ch=base * 8)
        self.up2 = UpAttn(skip_ch=base * 4,  dec_ch=base * 8,  out_ch=base * 4)
        self.up3 = UpAttn(skip_ch=base * 2,  dec_ch=base * 4,  out_ch=base * 2)
        self.up4 = UpAttn(skip_ch=base,      dec_ch=base * 2,  out_ch=base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


if __name__ == "__main__":
    # quick sanity check
    model = AttentionUNet(in_channels=1, out_channels=1, base=32)
    x = torch.randn(2, 1, 256, 256)
    y = model(x)
    print("output shape:", y.shape)  # expected: (2, 1, 256, 256)
