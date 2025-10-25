# Automatic-Segmentation-of-Cardiac-Structures-in-Magnetic-Resonance-Images

Fully automatic segmentation of LV, RV and Myocardium on the ACDC dataset using several U-Net variants:

- 2D U-Net (16/32/64 base channels)

- 3D U-Net (26 base channels; anisotropic pooling; whole-volume)

The repo contains training, evaluation and preprocessing code.

Best model: **3D U-Net (26 ch)** shows highest Dice across all classes and phases and lowest Hausdorff distances.

## Features
- 2D slice-wise and 3D whole-volume training
- Clean pre-processing ((in-plane) resampling, z-score normalization)
- Robust losses (Dice + CE)
- Learning-rate schedulers: Cosine (2D-16), ReduceLROnPlateau (2D-64 / 3D)
- Best-checkpoint saving

## Dice (ED / ES per class)

| Model | ED LV | ED Myo | ED RV | ES LV | ES Myo | ES RV |
|---|---:|---:|---:|---:|---:|---:|
| 2D U-Net (16 ch) | 0.959 | 0.865 | 0.918 | 0.912 | 0.883 | 0.863 |
| 2D U-Net (32 ch) | 0.962 | 0.876 | 0.929 | 0.920 | 0.894 | 0.871 |
| 2D U-Net (64 ch) | 0.964 | 0.880 | 0.933 | 0.913 | 0.892 | 0.881 |
| **3D U-Net (26 ch)** | **0.972** | **0.912** | **0.965** | **0.952** | **0.932** | **0.947** |

## Hausdorff distance [mm] (ED / ES per class)

| Model | ED LV | ED Myo | ED RV | ES LV | ES Myo | ES RV |
|---|---:|---:|---:|---:|---:|---:|
| 2D U-Net (16 ch) | 14.92 | 20.24 | 18.50 | 14.01 | 16.81 | 21.09 |
| 2D U-Net (32 ch) | 11.07 | 13.49 | 18.38 | 12.96 | 16.49 | 18.79 |
| 2D U-Net (64 ch) | 7.91 | 14.30 | 19.18 | 12.12 | 17.53 | 16.27 |
| **3D U-Net (26 ch)** | **4.35** | **4.80** | **7.64** | **4.92** | **4.93** | **7.47** |



## Citation

If you use this code, please cite the ACDC challenge: Bernard et al., ACDC Challenge (MICCAI 2017)
