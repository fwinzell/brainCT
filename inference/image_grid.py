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


class SelectionGUI(SegmentationGUI3):
    def __init__(self, master, gt, mri, ct):
        super().__init__(master, gt, mri, ct)
        self.master.bind("<space>", self.space_press)
        self.slices = []

        self.master.bind("<Escape>", self.escape_press)

    def space_press(self, event):
        self.slices.append(self.index)
        print(f"Slice: {self.index}")

    def escape_press(self, event):
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
            elif 'Escape' in self.master.state():
                self.escape_press()

        self.master.after(100, self.handle_keys)


def process_tile(img):
    img = img[::-1, ::-1, :]
    img = np.transpose(img, (1, 2, 0))
    return img


def generate_grid(selection, model_list, colnames=None):
    ct_imgs = []
    gt_imgs = []
    seg_imgs = {model: [] for model in model_list}
    n_slices = 0
    for vol_name, slices in selection.items():
        gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
        ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
        mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"
        gt = load_gt_nii(gt_path)
        _, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=True)

        for slice in slices:
            #images.append(torch.tensor(ct[:, slice, :, :]))
            #images.append(torch.tensor(gt[:, slice, :, :]))
            n_slices += 1
            ct_imgs.append(process_tile(ct[:, slice, :, :]))
            gt_imgs.append(process_tile(gt[:, slice, :, :]).astype(np.float32))

        for model in model_list:
            seg_path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/{model}/{vol_name}_prob_seg.nii.gz"
            seg = load_nii(seg_path, prob_map=True)
            for slice in slices:
                #images.append(torch.tensor(seg[:, slice, :, :]))
                seg_imgs[model].append(process_tile(seg[:, slice, :, :]).astype(np.float32))

    images = [ct_imgs, gt_imgs]
    for model in model_list:
        images.append(seg_imgs[model])

    #grid = make_grid(images, nrow=len(model_list)+2)
    # display result 

    #img = torchvision.transforms.ToPILImage()(grid) 
    #img.show() 

    matplotgrid(images, nrow=len(ct_imgs), ncol=len(images), column_titles=colnames, save=True)

def matplotgrid(images, nrow, ncol, column_titles=None, save=False):
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(ncol*2.5, nrow*2.5), constrained_layout=True)


    for col_ix, col in enumerate(images):
        for row_ix, img in enumerate(col):
            ax = axes[row_ix, col_ix]
            ax.imshow(img)
            ax.axis("off")

    if column_titles:
        for col_ix, title in enumerate(column_titles):
            ax = axes[0, col_ix]
            ax.set_title(title, fontsize=14, fontweight='bold', fontname='serif')

    
    plt.tight_layout(pad=0.2)
    if save:
        plt.savefig("/home/fi5666wi/Brain_CT_MR_data/plots/grid_2.png", dpi = 500)
    plt.show()


if __name__ == "__main__":
    
    # test "8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"
    # train "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
    #       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
    #  3mm  "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"  

    #path = "/home/fi5666wi/Brain_CT_MR_data/matlab/8_Ms59/c01_1_00001_temp_8_Ms59_M70_l_T1_CTseg.nii"
    sel = {}

    #vol_name = "19_LH64"
    for vol_name in ["9_Kh43"]:
        #path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-28/version_0_prob/{vol_name}_seg.nii.gz"
        gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
        ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
        mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"
        #gt = load_gt_nii(gt_path)
        mri, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=True)

        # Create the Tkinter window
        root = tk.Tk()
        root.title("Image Viewer")

        # Create an instance of the ImageApp class
        #app = SegmentationGUI2(root, seg, gt, mri, ct)
        app = SelectionGUI(root, gt, mri, ct)

        # Run the Tkinter event loop
        root.mainloop()

        sel[vol_name] = app.slices

    #sel = {"18_MN44": [52, 82], "19_LH64": [42, 92]}

    model_list = ["crossval_2025-01-23_ensemble_prob", "crossval_2025-01-21_ensemble_prob", "crossval_2025-01-31_ensemble_prob"]
    generate_grid(sel, model_list, colnames=["CT image", "Ground Truth", "Baseline", "U-Net++ (Aug)", "U-Net (Aug)"])
    
