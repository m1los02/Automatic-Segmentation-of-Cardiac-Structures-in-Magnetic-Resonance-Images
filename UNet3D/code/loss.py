import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import medpy.metric.binary as mmb

def one_hot_3d_safe(target, n_class, ignore_index=None):
    if ignore_index is not None:
        valid = (target != ignore_index)
        safe_target = torch.where(valid, target, torch.zeros_like(target))
    else:
        valid = None
        safe_target = target
    oh = F.one_hot(safe_target.long(), num_classes=n_class)  
    oh = oh.permute(0,4,1,2,3).float()
    return oh, valid

def dice_loss_3d(logits, target, n_class, smooth=1e-5, ignore_index=None):
    probs = F.softmax(logits, dim=1)               
    tgt_oh, valid = one_hot_3d_safe(target, n_class, ignore_index)

    if valid is not None:
        mask = valid.float().unsqueeze(1)           
        probs  = probs  * mask
        tgt_oh = tgt_oh * mask

    dims = (0,2,3,4)
    inter = (probs * tgt_oh).sum(dims)
    denom = probs.sum(dims) + tgt_oh.sum(dims)
    dice = (2*inter + smooth) / (denom + smooth)  
    return 1 - dice.mean()

class DiceCELoss3D(nn.Module):
    def __init__(self, n_class, smooth=1e-5, ignore_index=None, dice_w=1.0, ce_w=1.0):
        super().__init__()
        self.n_class = n_class
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.dice_w = dice_w
        self.ce_w = ce_w
        self.ce = nn.CrossEntropyLoss() if ignore_index is None else nn.CrossEntropyLoss(ignore_index=ignore_index)
    def forward(self, logits, target):
        ce   = self.ce(logits, target)
        dice = dice_loss_3d(logits, target, self.n_class, self.smooth, self.ignore_index)
        return self.ce_w * ce + self.dice_w * dice

@torch.no_grad()
def dice_per_class_np(pred, gt, C):
    out = []
    for c in range(1, C):  # skip bg=0
        p = (pred == c); g = (gt == c)
        out.append(mmb.dc(p, g))
    return np.array(out, dtype=np.float32)