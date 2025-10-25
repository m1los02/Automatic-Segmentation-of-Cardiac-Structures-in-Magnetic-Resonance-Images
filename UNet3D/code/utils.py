import os, random, numpy as np, torch

LOG_EVERY_EPOCH = 1
IGNORE_INDEX = 255
TARGET_SPACING = (1.5, 1.5, 10.0)

def set_random(seed: int = 0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_if_not(path: str):
    os.makedirs(path, exist_ok=True)

def pad_to_multiple_hw(arr_hwz, multiple_hw=(16, 16), pad_val=0):
    """Pad [H,W,Z] so H and W are divisible by given multiples."""
    import numpy as np
    H, W, Z = arr_hwz.shape
    mH, mW = multiple_hw
    pH = (mH - H % mH) % mH
    pW = (mW - W % mW) % mW
    pad = ((0, pH), (0, pW), (0, 0))
    out = np.pad(arr_hwz, pad, mode='constant', constant_values=pad_val)
    return out, pad

def apply_pad_inv(arr_hwz, pad):
    """Unpad [H,W,Z] using ((tH,bH),(tW,bW),(tD,bD))."""
    (tH, bH), (tW, bW), (tD, bD) = pad
    H, W, Z = arr_hwz.shape
    return arr_hwz[:H-bH if bH else H, :W-bW if bW else W, :Z-bD if bD else Z]
