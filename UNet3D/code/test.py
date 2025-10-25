import os
import argparse
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.amp import autocast
import medpy.metric.binary as mmb

from unet3d import UNet3D
from preprocess import resample_itk
from utils import pad_to_multiple_hw, apply_pad_inv, create_if_not, TARGET_SPACING


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=str, default='0')
    ap.add_argument('--data_dir', type=str, default='../outputs_ACDC')
    ap.add_argument('--ckpt', type=str, required=True, help='path to trained weights, e.g. ../log/ACDC_3D/model/best.pth')
    ap.add_argument('--num_class', type=int, default=4)
    ap.add_argument('--base_ch', type=int, default=26)
    ap.add_argument('--results_dir', type=str, default='./results_3d')
    ap.add_argument('--save_pred', type=int, default=1)
    return ap.parse_args()

@torch.no_grad()
def infer_volume(model, vol_zyx):
    vol_hwz = np.transpose(vol_zyx, (1,2,0))
    vol_hwz_pad, pad = pad_to_multiple_hw(vol_hwz, (16,16), pad_val=0)

    vol_dhw = np.transpose(vol_hwz_pad, (2,0,1)) 
    x = torch.from_numpy(vol_dhw).unsqueeze(0).unsqueeze(0).float().cuda()

    with autocast('cuda'):
        logits = model(x)                                
        pred = torch.argmax(logits, dim=1).squeeze(0) 

    pred_dhw = pred.cpu().numpy().astype(np.int16)
    pred_hwz = np.transpose(pred_dhw, (1,2,0))
    pred_hwz = apply_pad_inv(pred_hwz, pad)               
    pred_zyx = np.transpose(pred_hwz, (2,0,1))           
    return pred_zyx

def dice_per_class_np(pred_zyx, gt_zyx, C):
    # without bg=0 
    out = []
    for c in range(1, C):
        p = (pred_zyx == c)
        g = (gt_zyx   == c)
        if g.sum() == 0 and p.sum() == 0:
            out.append(1.0)
        elif g.sum() == 0 or p.sum() == 0:
            out.append(0.0)
        else:
            out.append(mmb.dc(p, g))
    return np.array(out, dtype=np.float32)

def hd_per_class_np(pred_zyx, gt_zyx, C, spacing_zyx):
    out = []
    for c in range(1, C):
        p = (pred_zyx == c)
        g = (gt_zyx   == c)
        if p.sum() == 0 or g.sum() == 0:
            out.append(np.nan) 
        else:
            out.append(mmb.hd(p, g, voxelspacing=spacing_zyx))
    return np.array(out, dtype=np.float32)

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model = UNet3D(in_ch=1, n_class=args.num_class, base=args.base_ch).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    vol_dir = os.path.join(args.data_dir, 'volume')
    with open(os.path.join(args.data_dir, 'test.txt'), 'r') as f:
        test_ids = [line.strip() for line in f]

    results_dir = args.results_dir
    create_if_not(results_dir)
    results_txt = os.path.join(results_dir, 'results.txt')
    if os.path.exists(results_txt):
        os.remove(results_txt)

    # acc by phase
    dice_ED, dice_ES = [], []
    hd_ED,   hd_ES   = [], []

    spacing_zyx = (TARGET_SPACING[2], TARGET_SPACING[1], TARGET_SPACING[0])

    for idx, vid in enumerate(test_ids, 1):
        img_p = os.path.join(vol_dir, f'{vid}.nii.gz')
        gt_p  = os.path.join(vol_dir, f'{vid}_gt.nii.gz')
        itk_img = sitk.ReadImage(img_p)
        itk_gt  = sitk.ReadImage(gt_p)

        # resample
        img_r = resample_itk(itk_img, TARGET_SPACING, is_label=False)
        gt_r  = resample_itk(itk_gt,  TARGET_SPACING, is_label=True)

        img_zyx = sitk.GetArrayFromImage(img_r).astype(np.float32)  
        gt_zyx  = sitk.GetArrayFromImage(gt_r).astype(np.int16)

        # normalize
        mu, sd = img_zyx.mean(), img_zyx.std()
        img_zyx = (img_zyx - mu) / (sd + 1e-8)

        pred_zyx = infer_volume(model, img_zyx)

        # metrics per class
        dice_arr = dice_per_class_np(pred_zyx, gt_zyx, args.num_class)
        hd_arr   = hd_per_class_np(pred_zyx, gt_zyx, args.num_class, spacing_zyx)

        if args.save_pred:
            sitk.WriteImage(img_r,  os.path.join(results_dir, f'image_{idx}.nii.gz'))
            sitk.WriteImage(gt_r,   os.path.join(results_dir, f'label_{idx}.nii.gz'))
            pred_itk = sitk.GetImageFromArray(pred_zyx.astype(np.int16))
            pred_itk.CopyInformation(img_r)
            sitk.WriteImage(pred_itk, os.path.join(results_dir, f'predict_{idx}.nii.gz'))

        # ED/ES split
        phase = "ED" if "frame01" in vid else "ES"
        if phase == "ED":
            dice_ED.append(dice_arr); hd_ED.append(hd_arr)
        else:
            dice_ES.append(dice_arr); hd_ES.append(hd_arr)

        # log
        with open(results_txt, 'a') as f:
            f.write(
                f"{os.path.basename(img_p)} ({phase}) "
                f"RV/Myo/LV Dice: {dice_arr[0]:.4f}, {dice_arr[1]:.4f}, {dice_arr[2]:.4f}, "
                f"Mean: {np.nanmean(dice_arr):.4f} | "
                f"RV/Myo/LV HD: {hd_arr[0]:.2f}, {hd_arr[1]:.2f}, {hd_arr[2]:.2f}, "
                f"Mean: {np.nanmean(hd_arr):.2f}\n"
            )

    dice_ED = np.array(dice_ED) if len(dice_ED) else np.zeros((0,3), dtype=np.float32)
    dice_ES = np.array(dice_ES) if len(dice_ES) else np.zeros((0,3), dtype=np.float32)
    hd_ED   = np.array(hd_ED)   if len(hd_ED)   else np.zeros((0,3), dtype=np.float32)
    hd_ES   = np.array(hd_ES)   if len(hd_ES)   else np.zeros((0,3), dtype=np.float32)

    mean_ED_dice = np.nanmean(dice_ED, axis=0) if dice_ED.size else np.zeros(3)
    mean_ES_dice = np.nanmean(dice_ES, axis=0) if dice_ES.size else np.zeros(3)
    mean_ED_hd   = np.nanmean(hd_ED,   axis=0) if hd_ED.size   else np.zeros(3)
    mean_ES_hd   = np.nanmean(hd_ES,   axis=0) if hd_ES.size   else np.zeros(3)

    print("\n=== Final Results per Phase ===")
    print(f"ED LV Dice: {mean_ED_dice[2]:.4f}, Myo: {mean_ED_dice[1]:.4f}, RV: {mean_ED_dice[0]:.4f}")
    print(f"ES LV Dice: {mean_ES_dice[2]:.4f}, Myo: {mean_ES_dice[1]:.4f}, RV: {mean_ES_dice[0]:.4f}")
    print(f"ED LV HD: {mean_ED_hd[2]:.2f}, Myo: {mean_ED_hd[1]:.2f}, RV: {mean_ED_hd[0]:.2f}")
    print(f"ES LV HD: {mean_ES_hd[2]:.2f}, Myo: {mean_ES_hd[1]:.2f}, RV: {mean_ES_hd[0]:.2f}")

    with open(results_txt, 'a') as f:
        f.write("\n=== Summary (Mean per phase) ===\n")
        f.write(f"ED LV Dice: {mean_ED_dice[2]:.4f}, Myo: {mean_ED_dice[1]:.4f}, RV: {mean_ED_dice[0]:.4f}\n")
        f.write(f"ES LV Dice: {mean_ES_dice[2]:.4f}, Myo: {mean_ES_dice[1]:.4f}, RV: {mean_ES_dice[0]:.4f}\n")
        f.write(f"ED LV HD: {mean_ED_hd[2]:.2f}, Myo: {mean_ED_hd[1]:.2f}, RV: {mean_ED_hd[0]:.2f}\n")
        f.write(f"ES LV HD: {mean_ES_hd[2]:.2f}, Myo: {mean_ES_hd[1]:.2f}, RV: {mean_ES_hd[0]:.2f}\n")

    print("\nSaved all results to:", results_txt)

if __name__ == "__main__":
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
