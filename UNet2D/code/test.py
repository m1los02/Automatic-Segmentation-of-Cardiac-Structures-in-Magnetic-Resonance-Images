import numpy as np
import os
import SimpleITK as sitk
import torch

from unet2d import UNet
from utils import (
    test_single_case_dc,
    test_single_volume_slicebyslice,
    test_single_case_hd,
    create_if_not
)
from preprocess import copy_geometry


if __name__ == "__main__":
    best_model_path = '../log/ACDC0404/model/best.pth'
    data_dir = '../outputs_ACDC'
    gpu = '0'
    test_data_path = os.path.join(data_dir, 'volume')
    NUM_CLS = 4
    save_visualization = True
    results_dir = './results'
    create_if_not(results_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    model = UNet(in_chns=1, class_num=NUM_CLS).cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
        test_list = [os.path.join(test_data_path, line.strip() + '.nii.gz') for line in f]

    # Acc for ED and ES
    dice_ED, dice_ES = [], []
    hd_ED, hd_ES = [], []

    for idx, test_img in enumerate(test_list):
        img_path = test_img
        seg_path = img_path.replace('.nii.gz', '_gt.nii.gz')
        itkimg = sitk.ReadImage(img_path)
        itkseg = sitk.ReadImage(seg_path)
        spacing = itkimg.GetSpacing()[::-1]
        img = sitk.GetArrayFromImage(itkimg)
        seg = sitk.GetArrayFromImage(itkseg)
        img = (img - np.mean(img)) / np.std(img)

        pred = test_single_volume_slicebyslice(img, model)
        itkpred = sitk.GetImageFromArray(pred)
        itkpred = copy_geometry(itkpred, itkimg)

        dice_arr = test_single_case_dc(pred, seg, NUM_CLS)
        hd_arr = test_single_case_hd(pred, seg, NUM_CLS, spacing)

        # Save predicted
        if save_visualization:
            sitk.WriteImage(itkimg, os.path.join(results_dir, f'image_{idx+1}.nii.gz'))
            sitk.WriteImage(itkseg, os.path.join(results_dir, f'label_{idx+1}.nii.gz'))
            sitk.WriteImage(itkpred, os.path.join(results_dir, f'predict_{idx+1}.nii.gz'))

        # Phase
        phase = "ED" if "frame01" in img_path else "ES"

        # Store metrics
        if phase == "ED":
            dice_ED.append(dice_arr)
            hd_ED.append(hd_arr)
        else:
            dice_ES.append(dice_arr)
            hd_ES.append(hd_arr)

        # Write result
        with open(os.path.join(results_dir, 'results.txt'), 'a') as f:
            f.write(
                f"{os.path.basename(img_path)} ({phase}) "
                f"RV/Myo/LV Dice: {dice_arr[0]:.4f}, {dice_arr[1]:.4f}, {dice_arr[2]:.4f}, "
                f"Mean: {(np.mean(dice_arr)):.4f} | "
                f"RV/Myo/LV HD: {hd_arr[0]:.2f}, {hd_arr[1]:.2f}, {hd_arr[2]:.2f}, "
                f"Mean: {(np.mean(hd_arr)):.2f}\n"
            )

    dice_ED = np.array(dice_ED)
    dice_ES = np.array(dice_ES)
    hd_ED = np.array(hd_ED)
    hd_ES = np.array(hd_ES)

    # Compute averages
    mean_ED = np.mean(dice_ED, axis=0) if len(dice_ED) > 0 else np.zeros(3)
    mean_ES = np.mean(dice_ES, axis=0) if len(dice_ES) > 0 else np.zeros(3)
    mean_HD_ED = np.mean(hd_ED, axis=0) if len(hd_ED) > 0 else np.zeros(3)
    mean_HD_ES = np.mean(hd_ES, axis=0) if len(hd_ES) > 0 else np.zeros(3)

    # Print
    print("\n=== Final Results per Phase ===")
    print(f"ED LV Dice: {mean_ED[2]:.4f}, Myo: {mean_ED[1]:.4f}, RV: {mean_ED[0]:.4f}")
    print(f"ES LV Dice: {mean_ES[2]:.4f}, Myo: {mean_ES[1]:.4f}, RV: {mean_ES[0]:.4f}")
    print(f"ED LV HD: {mean_HD_ED[2]:.2f}, Myo: {mean_HD_ED[1]:.2f}, RV: {mean_HD_ED[0]:.2f}")
    print(f"ES LV HD: {mean_HD_ES[2]:.2f}, Myo: {mean_HD_ES[1]:.2f}, RV: {mean_HD_ES[0]:.2f}")

    # Summary
    with open(os.path.join(results_dir, 'results.txt'), 'a') as f:
        f.write("\n=== Summary (Mean per phase) ===\n")
        f.write(f"ED LV Dice: {mean_ED[2]:.4f}, Myo: {mean_ED[1]:.4f}, RV: {mean_ED[0]:.4f}\n")
        f.write(f"ES LV Dice: {mean_ES[2]:.4f}, Myo: {mean_ES[1]:.4f}, RV: {mean_ES[0]:.4f}\n")
        f.write(f"ED LV HD: {mean_HD_ED[2]:.2f}, Myo: {mean_HD_ED[1]:.2f}, RV: {mean_HD_ED[0]:.2f}\n")
        f.write(f"ES LV HD: {mean_HD_ES[2]:.2f}, Myo: {mean_HD_ES[1]:.2f}, RV: {mean_HD_ES[0]:.2f}\n")

    print("\nSaved all results to:", os.path.join(results_dir, 'results.txt'))
