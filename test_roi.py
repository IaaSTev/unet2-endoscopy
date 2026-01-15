import torch
from roi import band_to_bbox, crop_by_bbox

def main():
    # 人造一个 band（中间一圈白）
    H, W = 256, 256
    band = torch.zeros(H, W)
    band[120:136, 60:200] = 1.0  # 假装边界带

    bbox = band_to_bbox(band, pad=16)
    print("bbox:", bbox)

    img = torch.randn(1, 1, H, W)
    patch = crop_by_bbox(img, bbox)
    print("patch shape:", patch.shape)

if __name__ == "__main__":
    main()
