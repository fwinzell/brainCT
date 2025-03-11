import SimpleITK as sitk
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from torchmetrics import Dice, JaccardIndex
from display_gui import load_gt_nii, load_nii
import os


def slice2img(slice):
    # prediction

    gm = slice[1, :, :]
    wm = slice[0, :, :]
    csf = slice[2, :, :]

    wmgm_rgb = np.stack((csf, gm, wm), axis=0)  # 0 * gm), axis=0)
    np_img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)
    pil_img = Image.fromarray(np_img)
    #tk_img = ImageTk.PhotoImage(pil_img.resize((300, 300)))

    return pil_img

def load_gt_and_ct_nii(gt_path, ct_path):
    indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(gt_path)), 0)

    valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
    valid_slices = valid_slices[1:-1, 0] # -2 to account for last and first slice in 2.5D
    indxs = indxs[:, valid_slices, :, :]

    CSF = indxs == 3
    GM = indxs == 2
    WM = indxs == 1

    indxs = np.concatenate((WM, GM, CSF), axis=0)

    ct = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), 0)
    ct = ct[:, valid_slices, :, :]

    return indxs, ct

def save_slice(slice_n, ID):
    folder = "/home/fi5666wi/Brain_CT_MR_data/OUT/final_models/"
    save_dir = "/home/fi5666wi/Brain_CT_MR_data/OUT/images/"

    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{ID}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/CT/{ID}_M70_l_T1.nii"
    gt, ct = load_gt_and_ct_nii(gt_path, ct_path)
    gt_img = slice2img(gt[:, slice_n, :, :])
    ct_img = np.squeeze((ct[:, slice_n, :, :] / 100) * 255).astype(np.uint8)
    ct_img = Image.fromarray(ct_img)

    gt_img.save(os.path.join(save_dir, f"{ID}_{slice_n}_gt.png"))
    ct_img.save(os.path.join(save_dir, f"{ID}_{slice_n}_ct.png"))

    for model in os.listdir(folder):
        nii_file = os.path.join(folder, model, f"{ID}_M70_l_T1_seg.nii.gz")
        seg = load_nii(nii_file)
        seg_img = slice2img(seg[:, slice_n, :, :])
        save_name = f"{ID}_{slice_n}_{model}.png"
        seg_img.save(os.path.join(save_dir, save_name))

def display_slice(slice_n, ID):
    folder = "/home/fi5666wi/Brain_CT_MR_data/OUT/final_models/"

    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{ID}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/CT/{ID}_M70_l_T1.nii"
    gt, ct = load_gt_and_ct_nii(gt_path, ct_path)
    gt_img = slice2img(gt[:, slice_n, :, :])
    ct_img = np.squeeze((ct[:, slice_n, :, :] / 100) * 255).astype(np.uint8)
    ct_img = Image.fromarray(ct_img)

    ct_img.show()
    gt_img.show()




if __name__ == "__main__":
    ID = "8_Ms59"
    slice_n = 52
    display_slice(slice_n, ID)
    #save_slice(slice_n, ID)




