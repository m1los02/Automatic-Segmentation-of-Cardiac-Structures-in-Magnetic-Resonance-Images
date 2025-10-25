import os, numpy as np, torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from preprocess import resample_itk
from utils import pad_to_multiple_hw, IGNORE_INDEX, TARGET_SPACING

class ACDCVolume3D(Dataset):
    def __init__(self, data_dir, ids, augment=False):
        self.data_dir = data_dir
        self.ids = ids
        self.augment = augment
        self.vol_dir = os.path.join(data_dir, 'volume')

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        vid = self.ids[idx]
        img_itk = sitk.ReadImage(os.path.join(self.vol_dir, f"{vid}.nii.gz"))
        seg_itk = sitk.ReadImage(os.path.join(self.vol_dir, f"{vid}_gt.nii.gz"))

        # Resample 
        img_itk = resample_itk(img_itk, TARGET_SPACING, is_label=False)
        seg_itk = resample_itk(seg_itk, TARGET_SPACING, is_label=True)

        img = sitk.GetArrayFromImage(img_itk).astype(np.float32)  
        seg = sitk.GetArrayFromImage(seg_itk).astype(np.int16)   

        # Volume normaliyation
        mu, sd = img.mean(), img.std()
        img = (img - mu) / (sd + 1e-8)

        img = np.transpose(img, (1,2,0))
        seg = np.transpose(seg, (1,2,0))

        # aug
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0).copy(); seg = np.flip(seg, axis=0).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy(); seg = np.flip(seg, axis=1).copy()

        img, pad = pad_to_multiple_hw(img, (16,16), pad_val=0)
        seg, _   = pad_to_multiple_hw(seg, (16,16), pad_val=0)

        # To tensors
        img_t = torch.from_numpy(np.transpose(img, (2,0,1))).unsqueeze(0).float()
        seg_t = torch.from_numpy(np.transpose(seg, (2,0,1))).long()

        return img_t, seg_t, pad, vid


def collate_pad_depth(batch, ignore_index=IGNORE_INDEX):
    """
    Padding variable depths to the batch max depth.
    """
    imgs, segs, pads, vids = zip(*batch)
    Dmax = max(x.shape[1] for x in imgs)
    B = len(imgs); C = imgs[0].shape[0]
    H, W = imgs[0].shape[2], imgs[0].shape[3]

    out_i = imgs[0].new_zeros((B, C, Dmax, H, W))
    out_s = segs[0].new_full((B, Dmax, H, W), fill_value=ignore_index)

    for i, (im, sg) in enumerate(zip(imgs, segs)):
        D = im.shape[1]
        out_i[i, :, :D] = im
        out_s[i, :D]    = sg

    return out_i, out_s, list(pads), list(vids)
