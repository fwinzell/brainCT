from monai.transforms import Compose, ToTensord, ScaleIntensityd
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader

from data_loader import BrainXLDataset

datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"

# Removed due to insufficient quality on MRI image
# 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42

test_IDs = ["8_Ms59", "18_MN44", "19_LH64", "33_ET51"]
IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
       "25_HH57", "26_LB59", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

view_seg = False
total_vols = np.zeros(5)
for id in IDs:
    case = [(f"{datafolder}/{id}_M70_l_T1.nii", f"{datafolder}/{id}_seg3.nii")]

    val_transforms = Compose([ToTensord(keys=["img", "seg"]),
                              ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    dataset = BrainXLDataset(case, transform=val_transforms)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    vols = np.zeros(5)
    for i, batch in enumerate(loader):
        seg = batch["seg"]
        seg = seg.numpy().squeeze()
        wm = seg[0, :, :]
        gm = seg[1, :, :]
        csf = seg[2, :, :]
        non_bg = np.any(seg, axis=0)
        vols[0] += np.sum(wm)
        vols[1] += np.sum(gm)
        vols[2] += np.sum(csf)
        vols[3] += np.sum(non_bg)
        vols[4] += np.prod(seg.shape)

        if view_seg:
            view = np.moveaxis(seg, source=0, destination=-1)*255
            cv2.imshow("seg", view)
            cv2.waitKey(100)

    print(f"ID: {id}")
    print(f"WM: {vols[0]/1000}, GM: {vols[1]/1000}, CSF: {vols[2]/1000}, non-bg: {vols[3]/1000}")
    #print(f"Ratios: WM: {vols[0]/vols[3]}, GM: {vols[1]/vols[3]}, CSF: {vols[2]/vols[3]}")
    total_vols += vols

ratios = total_vols[0:3]/total_vols[3]
print(f"Ratios: WM: {1-ratios[0]}, GM: {1-ratios[1]}, CSF: {1-ratios[2]}")

freq = total_vols[0:3]/total_vols[4]
weights = 1E-4/(freq)**2
print(f"Class weights: WM: {weights[0]}, GM: {weights[1]}, CSF: {weights[2]}")

sums = 1E7/total_vols[0:3]
print(f"Class sums: WM: {sums[0]}, GM: {sums[1]}, CSF: {sums[2]}")

