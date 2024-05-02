import SimpleITK as sitk
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from torchmetrics import Dice, JaccardIndex
from monai.metrics import HausdorffDistanceMetric
import os
from display_gui import load_gt_nii
from faster_bootstrap import statistics, save_as_csv
import yaml

def pad_image(image, target_shape=(256, 256, 256)):
    original_shape = image.shape[1:]
    pad_width = ((target_shape[0] - original_shape[0]) // 2,
                 (target_shape[1] - original_shape[1]) // 2,
                 (target_shape[2] - original_shape[2]) // 2)

    padded_image = np.pad(image, ((0,0),
                                            (pad_width[0], target_shape[0] - original_shape[0] - pad_width[0]),
                                            (pad_width[1], target_shape[1] - original_shape[1] - pad_width[1]),
                                            (pad_width[2], target_shape[2] - original_shape[2] - pad_width[2])),
                                   mode='constant')
    return padded_image

def load_ct_seg(ct_name, display=False):
    ct_dir = os.path.join("/home/fi5666wi/Brain_CT_MR_data/matlab", ct_name)
    c01_path = os.path.join(ct_dir, f"c01_1_00001_temp_{ct_name}_M70_l_T1_CTseg.nii")  # GM
    c02_path = os.path.join(ct_dir, f"c02_1_00001_temp_{ct_name}_M70_l_T1_CTseg.nii")  # WM
    c03_path = os.path.join(ct_dir, f"c03_1_00001_temp_{ct_name}_M70_l_T1_CTseg.nii")  # CSF

    ct_seg = np.concatenate((
        np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(c02_path)), 0),
        np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(c01_path)), 0),
        np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(c03_path)), 0)), axis=0)

    #ct_seg = pad_image(ct_seg, target_shape=(256, 256, 256))

    if display:
        for i in range(ct_seg.shape[1]):
            slice = ct_seg[:, i, :, :]
            slice = np.moveaxis(slice, source=0, destination=-1)
            cv2.imshow('CTseg', slice)
            cv2.waitKey(100)

    return ct_seg

def calcul_dice_and_iou(ct_seg, gt, threshold=(0.5, 0.5, 0.5), display=False):
    dsc = Dice(zero_division=np.nan, ignore_index=0)
    iou = JaccardIndex(task='binary')
    ct_seg = np.concatenate((
        np.expand_dims((ct_seg[0, :, :, :] > threshold[0]).astype(np.uint8), axis=0),
        np.expand_dims((ct_seg[1, :, :, :] > threshold[1]).astype(np.uint8), axis=0),
        np.expand_dims((ct_seg[2, :, :, :] > threshold[2]).astype(np.uint8), axis = 0)), axis=0)

    n_slices = ct_seg.shape[1]
    dice_scores = np.zeros((n_slices, 3))
    iou_scores = np.zeros((n_slices, 3))
    for j in range(n_slices):
        ctslice = ct_seg[:, j, :, :]
        gtslice = gt[:, j, :, :].astype(np.uint8)

        for i in range(3):
            pred = torch.tensor(ctslice[i, :, :])
            tar = torch.tensor(gtslice[i, :, :])
            dice_scores[j, i] = dsc(pred, tar).item()
            iou_scores[j, i] = iou(pred, tar).item()

        ctslice = np.moveaxis(np.uint8(ctslice * 255), source=0, destination=-1)
        gtslice = np.moveaxis(np.uint8(gtslice * 255), source=0, destination=-1)

        if display:
            cv2.imshow('CTseg', ctslice)
            cv2.imshow('GT', gtslice)
            cv2.waitKey(10)

    return np.nanmean(dice_scores, axis=0), np.nanmean(iou_scores, axis=0)

def ctseg_results(ID):
    print(f"CT: {ID}")
    ct_seg = load_ct_seg(ID, display=False)
    gt = load_gt_nii(f"/home/fi5666wi/Brain_CT_MR_data/DL/{ID}_seg3.nii", select_all=True)
    dice_scores, iou_scores = calcul_dice_and_iou(ct_seg, gt, threshold=(0.1, 0.5, 0.9), display=False)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores, decimals=4)}")

    return {"Dice": dice_scores, "IoU": iou_scores}


def volume_bootstrap(save_path):
    test_IDs = np.array(["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"])
    res_dict = {}

    for ID in test_IDs:
        res = ctseg_results(ID)
        res_dict[ID] = res

    dasboot = {"Dice": np.zeros((len(test_IDs), 3)),
               "IoU": np.zeros((len(test_IDs), 3)),
               "Hausdorff": np.zeros((len(test_IDs), 3))}

    for i in range(len(test_IDs)):
        leave_out = test_IDs[i]
        sample = test_IDs[test_IDs != leave_out]

        dice = []
        iou = []
        hausdorff = []
        for ID in sample:
            res = res_dict[ID]
            dice.append(res["Dice"])
            iou.append(res["IoU"])
            #hausdorff.append(res["Hausdorff"])

        dasboot["Dice"][i] = np.nanmean(dice, axis=0)
        dasboot["IoU"][i] = np.nanmean(iou, axis=0)
        #dasboot["Hausdorff"][i] = np.nanmean(hausdorff, axis=0)

    stats = statistics(dasboot)
    print(stats["Dice"])

    with open(os.path.join(save_path, "volume_bootstrap.yaml"), "w") as f:
        yaml.dump(dasboot, f)

    save_as_csv(stats, os.path.join(save_path, "vol_stats.csv"))


def run_ctseg():
    # ct_seg = load_ct_seg("8_Ms59", display=True)

    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]

    dice_scores = np.zeros((len(IDs), 3))
    iou_scores = np.zeros((len(IDs), 3))
    for k, ct_name in enumerate(IDs):
        print(f"CT: {ct_name}")
        ct_seg = load_ct_seg(ct_name, display=False)
        gt = load_gt_nii(f"/home/fi5666wi/Brain_CT_MR_data/DL/{ct_name}_seg3.nii", select_all=True)
        dice_scores[k, :], iou_scores[k, :] = calcul_dice_and_iou(ct_seg, gt, threshold=(0.1, 0.5, 0.9), display=True)
        print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores[k, :], decimals=4)}")
        print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores[k, :], decimals=4)}")

    print(f"Mean dice scores (WM/GM/CSF): {np.around(np.mean(dice_scores, axis=0), decimals=4)}")

    # gt = load_gt_nii("/home/fi5666wi/Brain_CT_MR_data/DL/8_Ms59_seg3.nii", select_all=True)

    # display_dice(ct_seg, gt, threshold=0.5)

if __name__ == "__main__":
    volume_bootstrap("/home/fi5666wi/Python/Brain-CT/ctseg/")















