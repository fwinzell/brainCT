# Split Brain manually at orbital plane and calculate Dice 

import SimpleITK as sitk
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from torchvision.utils import make_grid
import torchvision

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from display_gui import load_gt_nii, load_nii, load_all_nii, calculate_metrics, SegmentationGUI3

from torchmetrics import Dice


class SelectOrbitalPlane(SegmentationGUI3):
    def __init__(self, master, gt, mri, ct):
        super().__init__(master, gt, mri, ct)
        self.master.bind("<space>", self.space_press)
        self.slice = None

    def space_press(self, event):
        self.slice = self.index
        print(f"Slice: {self.index}")
        self.master.destroy()

    def handle_keys(self):
        # Check if the up or down arrow keys are being held down
        if self.master.state().startswith('pressed'):
            if 'Up' in self.master.state():
                self.up_arrow_press()
            elif 'Down' in self.master.state():
                self.down_arrow_press()
            elif 'space' in self.master.state():
                self.space_press()

        self.master.after(100, self.handle_keys)

def calculate_metrics(seg, gt):
    gt = gt.astype(np.uint8)
    dice = Dice(zero_division=np.nan, ignore_index=0)

    (wm, gm, csf) = np.split(seg, 3, axis=0)
    (wm_t, gm_t, csf_t) = np.split(gt, 3, axis=0)

    dscs = [dice(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
            dice(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
            dice(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]

    return dscs


if __name__ == "__main__":
    
    # test "8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"
    # train "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
    #       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
    #  3mm  "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"  

    #path = "/home/fi5666wi/Brain_CT_MR_data/matlab/8_Ms59/c01_1_00001_temp_8_Ms59_M70_l_T1_CTseg.nii"
    vol_name = "8_Ms59"
    path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-21_ensemble_prob/{vol_name}_prob_seg.nii.gz"
    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
    mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"
    seg = load_nii(path, prob_map=True)
    #gt = load_gt_nii(gt_path)
    mri, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=False)
    print(seg.shape)
    print(gt.shape)
    print(mri.shape)
    print(ct.shape)

    dscs = calculate_metrics(seg, gt)
    print(f"Dice scores (WM/GM/CSF): {np.around(dscs, decimals=4)}")
    
    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageApp class
    #app = SegmentationGUI2(root, seg, gt, mri, ct)
    app = SelectOrbitalPlane(root, gt, mri, ct)

    # Run the Tkinter event loop
    root.mainloop()

    slice = app.slice
    up_seg = seg[:, slice:, :, :]
    up_gt = gt[:, slice:, :, :]
    dscs = calculate_metrics(up_seg, up_gt)
    print(f"Above Oribital Dice scores (WM/GM/CSF): {np.around(dscs, decimals=4)}")

    down_seg = seg[:, :slice, :, :]
    down_gt = gt[:, :slice, :, :]
    dscs = calculate_metrics(down_seg, down_gt)
    print(f"Below Oribital Dice scores (WM/GM/CSF): {np.around(dscs, decimals=4)}")

