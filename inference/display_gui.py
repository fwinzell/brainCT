import SimpleITK as sitk
import numpy as np
import torch
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from torchmetrics import Dice, JaccardIndex


class SegmentationGUI:
    def __init__(self, master, seg, gt=None):
        self.master = master
        self.seg = seg
        self.gt = gt

        if gt is not None:
            assert seg.shape == gt.shape, "Segmentation and ground truth must have the same shape"

        self.index = 0
        self.max_index = seg.shape[1] - 1
        w = 320 if gt is None else 960
        self.canvas = tk.Canvas(master, width=w, height=330)
        self.canvas.pack()
        self.canvas.create_text(160, 5, text="Segmentation", anchor=tk.N)
        if gt is not None:
            self.canvas.create_text(480, 5, text="Ground truth", anchor=tk.N)
            self.canvas.create_text(800, 5, text="Correct/incorrect", anchor=tk.N)

        self.master.bind("<Up>", self.up_arrow_press)
        self.master.bind("<Down>", self.down_arrow_press)

        self.master.after(100, self.handle_keys)

        # Load and display the first image
        img = self.display_slice(self.seg[:, self.index, :, :])
        self.image_on_canvas = self.canvas.create_image(10, 20, anchor=tk.NW, image=img)

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.gt_on_canvas = self.canvas.create_image(330, 20, anchor=tk.NW, image=gt_img)

            correct_img = self.correct_image()
            self.correct_on_canvas = self.canvas.create_image(650, 20, anchor=tk.NW, image=correct_img)


    def up_arrow_press(self, event):
        self.index += 1
        if self.index > self.max_index:
            self.index = 0

        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            correct_img = self.correct_image()
            self.canvas.itemconfig(self.correct_on_canvas, image=correct_img)
            self.canvas.correct_img = correct_img

    def down_arrow_press(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = self.max_index
        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            correct_img = self.correct_image()
            self.canvas.itemconfig(self.correct_on_canvas, image=correct_img)
            self.canvas.correct_img = correct_img

    def correct_image(self):
        (wm, gm, csf) = np.split(self.seg[:, self.index, :, :], 3, axis=0)
        (wm_t, gm_t, csf_t) = np.split(self.gt[:, self.index, :, :], 3, axis=0)
        # find all correct and incorrect pixels, not including background and make an image with green and red colors
        pred_img = np.zeros((256, 256, 3), dtype=np.uint8)
        corr = np.logical_and(gm, gm_t) + np.logical_and(wm, wm_t) + np.logical_and(csf, csf_t)
        pred_img[corr.squeeze(), :] = [0, 255, 0]
        incorr = np.logical_xor(gm, gm_t) + np.logical_xor(wm, wm_t) + np.logical_xor(csf, csf_t)
        pred_img[incorr.squeeze(), :] = [255, 0, 0]

        pred_img = Image.fromarray(pred_img)
        tk_img = ImageTk.PhotoImage(pred_img.resize((300, 300)))

        return tk_img

    def handle_keys(self):
        # Check if the up or down arrow keys are being held down
        if self.master.state().startswith('pressed'):
            if 'Up' in self.master.state():
                self.up_arrow_press()
            elif 'Down' in self.master.state():
                self.down_arrow_press()

        self.master.after(100, self.handle_keys)


    def display_slice(self, slice):
        # prediction

        gm = slice[1, :, :]
        wm = slice[0, :, :]
        csf = slice[2, :, :]

        wmgm_rgb = np.stack((csf, gm, wm), axis=0) #0 * gm), axis=0)
        np_img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)
        pil_img = Image.fromarray(np_img)
        tk_img = ImageTk.PhotoImage(pil_img.resize((300, 300)))

        return tk_img


class SegmentationGUI2:
    def __init__(self, master, seg, gt, mri, ct):
        self.master = master
        self.seg = seg
        self.gt = gt
        self.mri = mri
        self.ct = ct

        assert seg.shape == gt.shape, "Segmentation and ground truth must have the same shape"
        assert seg.shape == mri.shape, "Segmentation and MRI must also have the same shape"
        assert seg.shape == ct.shape, "Segmentation and CT must also have the same shape"

        #self.dice = Dice(zero_division=np.nan, ignore_index=0)
        self.iou = JaccardIndex(task='binary')

        self.index = 0
        self.max_index = seg.shape[1] - 1
        self.canvas = tk.Canvas(master, width=1280, height=670)
        self.canvas.pack()
        self.canvas.create_text(160, 5, text="Segmentation", anchor=tk.N)
        self.canvas.create_text(480, 5, text="Ground truth", anchor=tk.N)
        self.canvas.create_text(800, 5, text="Correct/incorrect", anchor=tk.N)
        self.canvas.create_text(1120, 5, text="MRI", anchor=tk.N)

        self.canvas.create_text(160, 335, text="White matter", anchor=tk.N)
        self.canvas.create_text(480, 335, text="Gray matter", anchor=tk.N)
        self.canvas.create_text(800, 335, text="Cerebral spinal fluid", anchor=tk.N)
        self.canvas.create_text(1120, 335, text="CT", anchor=tk.N)

        self.master.bind("<Up>", self.up_arrow_press)
        self.master.bind("<Down>", self.down_arrow_press)

        self.master.after(100, self.handle_keys)

        # Load and display the first image
        img = self.display_slice(self.seg[:, self.index, :, :])
        self.image_on_canvas = self.canvas.create_image(10, 20, anchor=tk.NW, image=img)

        gt_img = self.display_slice(self.gt[:, self.index, :, :])
        self.gt_on_canvas = self.canvas.create_image(330, 20, anchor=tk.NW, image=gt_img)

        mri_img = self.display_slice(self.mri[:, self.index, :, :])
        self.mri_on_canvas = self.canvas.create_image(970, 20, anchor=tk.NW, image=mri_img)

        ct_img = self.display_slice(self.ct[:, self.index, :, :])
        self.ct_on_canvas = self.canvas.create_image(970, 350, anchor=tk.NW, image=ct_img)

        correct_img, wm_img, gm_img, csf_img, metric = self.update_images()

        self.correct_on_canvas = self.canvas.create_image(650, 20, anchor=tk.NW, image=correct_img)
        self.wm_on_canvas = self.canvas.create_image(10, 350, anchor=tk.NW, image=wm_img)
        self.wm_iou = self.canvas.create_text(160, 655, text=f"WM IoU: {metric[0]:.3f}", anchor=tk.N, fill="green")

        self.gm_on_canvas = self.canvas.create_image(330, 350, anchor=tk.NW, image=gm_img)
        self.gm_iou = self.canvas.create_text(480, 655, text=f"GM IoU: {metric[1]:.3f}", anchor=tk.N, fill="blue")

        self.csf_on_canvas = self.canvas.create_image(650, 350, anchor=tk.NW, image=csf_img)
        self.csf_iou = self.canvas.create_text(800, 655, text=f"CSF IoU: {metric[2]:.3f}", anchor=tk.N, fill="red")



    def up_arrow_press(self, event):
        self.index += 1
        if self.index > self.max_index:
            self.index = 0

        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            correct_img, wm_img, gm_img, csf_img, metric = self.update_images()

            mri_img = self.display_slice(self.mri[:, self.index, :, :])
            self.canvas.itemconfig(self.mri_on_canvas, image=mri_img)
            self.canvas.mri_img = mri_img

            ct_img = self.display_slice(self.ct[:, self.index, :, :])
            self.canvas.itemconfig(self.ct_on_canvas, image=ct_img)
            self.canvas.ct_img = ct_img

            self.canvas.itemconfig(self.correct_on_canvas, image=correct_img)
            self.canvas.correct_img = correct_img

            self.canvas.itemconfig(self.wm_on_canvas, image=wm_img)
            self.canvas.wm_img = wm_img
            self.canvas.itemconfig(self.wm_iou, text=f"WM IoU: {metric[0]:.3f}")

            self.canvas.itemconfig(self.gm_on_canvas, image=gm_img)
            self.canvas.gm_img = gm_img
            self.canvas.itemconfig(self.gm_iou, text=f"GM IoU: {metric[1]:.3f}")

            self.canvas.itemconfig(self.csf_on_canvas, image=csf_img)
            self.canvas.csf_img = csf_img
            self.canvas.itemconfig(self.csf_iou, text=f"CSF IoU: {metric[2]:.3f}")

    def down_arrow_press(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = self.max_index
        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            correct_img, wm_img, gm_img, csf_img, metric = self.update_images()

            mri_img = self.display_slice(self.mri[:, self.index, :, :])
            self.canvas.itemconfig(self.mri_on_canvas, image=mri_img)
            self.canvas.mri_img = mri_img

            ct_img = self.display_slice(self.ct[:, self.index, :, :])
            self.canvas.itemconfig(self.ct_on_canvas, image=ct_img)
            self.canvas.ct_img = ct_img

            self.canvas.itemconfig(self.correct_on_canvas, image=correct_img)
            self.canvas.correct_img = correct_img

            self.canvas.itemconfig(self.wm_on_canvas, image=wm_img)
            self.canvas.wm_img = wm_img
            self.canvas.itemconfig(self.wm_iou, text=f"WM IoU: {metric[0]:.3f}")

            self.canvas.itemconfig(self.gm_on_canvas, image=gm_img)
            self.canvas.gm_img = gm_img
            self.canvas.itemconfig(self.gm_iou, text=f"GM IoU: {metric[1]:.3f}")

            self.canvas.itemconfig(self.csf_on_canvas, image=csf_img)
            self.canvas.csf_img = csf_img
            self.canvas.itemconfig(self.csf_iou, text=f"CSF IoU: {metric[2]:.3f}")

    def update_images(self):
        (wm, gm, csf) = np.split(self.seg[:, self.index, :, :], 3, axis=0)
        (wm_t, gm_t, csf_t) = np.split(self.gt[:, self.index, :, :], 3, axis=0)
        ious = [self.iou(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
                self.iou(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
                self.iou(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]

        wm_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        wm_corr = np.logical_and(wm, wm_t)
        wm_incorr = np.logical_xor(wm, wm_t)
        wm_pred[wm_corr.squeeze(), :] = [0, 255, 0]
        wm_pred[wm_incorr.squeeze(), :] = [255, 0, 0]

        wm_pred = Image.fromarray(wm_pred)
        tk_wm_img = ImageTk.PhotoImage(wm_pred.resize((300, 300)))

        gm_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        gm_corr = np.logical_and(gm, gm_t)
        gm_incorr = np.logical_xor(gm, gm_t)
        gm_pred[gm_corr.squeeze(), :] = [0, 255, 0]
        gm_pred[gm_incorr.squeeze(), :] = [255, 0, 0]

        gm_pred = Image.fromarray(gm_pred)
        tk_gm_img = ImageTk.PhotoImage(gm_pred.resize((300, 300)))

        csf_pred = np.zeros((256, 256, 3), dtype=np.uint8)
        csf_corr = np.logical_and(csf, csf_t)
        csf_incorr = np.logical_xor(csf, csf_t)
        csf_pred[csf_corr.squeeze(), :] = [0, 255, 0]
        csf_pred[csf_incorr.squeeze(), :] = [255, 0, 0]

        csf_pred = Image.fromarray(csf_pred)
        tk_csf_img = ImageTk.PhotoImage(csf_pred.resize((300, 300)))

        # find all correct and incorrect pixels, not including background and make an image with green and red colors
        pred_img = np.zeros((256, 256, 3), dtype=np.uint8)
        corr = gm_corr + wm_corr + csf_corr
        pred_img[corr.squeeze(), :] = [0, 255, 0]
        incorr = gm_incorr + wm_incorr + csf_incorr
        pred_img[incorr.squeeze(), :] = [255, 0, 0]

        pred_img = Image.fromarray(pred_img)
        tk_img = ImageTk.PhotoImage(pred_img.resize((300, 300)))

        return tk_img, tk_wm_img, tk_gm_img, tk_csf_img, ious

    def handle_keys(self):
        # Check if the up or down arrow keys are being held down
        if self.master.state().startswith('pressed'):
            if 'Up' in self.master.state():
                self.up_arrow_press()
            elif 'Down' in self.master.state():
                self.down_arrow_press()

        self.master.after(100, self.handle_keys)


    def display_slice(self, slice):
        # prediction

        gm = slice[1, :, :]
        wm = slice[0, :, :]
        csf = slice[2, :, :]

        wmgm_rgb = np.stack((csf, gm, wm), axis=0) #0 * gm), axis=0)
        np_img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)
        pil_img = Image.fromarray(np_img)
        tk_img = ImageTk.PhotoImage(pil_img.resize((300, 300)))

        return tk_img

class SegmentationGUI3:
    def __init__(self, master, gt, mri, ct):
        self.master = master
        self.gt = gt
        self.mri = mri
        self.ct = ct

        assert mri.shape == gt.shape
        assert ct.shape == mri.shape
        assert gt.shape == ct.shape

        self.index = 0
        self.max_index = gt.shape[1] - 1
        self.canvas = tk.Canvas(master, width=970, height=330)
        self.canvas.pack()
        self.canvas.create_text(160, 5, text="Ground Truth", anchor=tk.N)
        self.canvas.create_text(480, 5, text="CT", anchor=tk.N)
        self.canvas.create_text(800, 5, text="MRI", anchor=tk.N)
  

        self.master.bind("<Up>", self.up_arrow_press)
        self.master.bind("<Down>", self.down_arrow_press)

        self.master.after(100, self.handle_keys)

        # Load and display the first image
        gt_img = self.display_slice(self.gt[:, self.index, :, :])
        self.gt_on_canvas = self.canvas.create_image(10, 20, anchor=tk.NW, image=gt_img)

        ct_img = self.display_slice(self.ct[:, self.index, :, :])
        self.ct_on_canvas = self.canvas.create_image(330, 20, anchor=tk.NW, image=ct_img)

        mri_img = self.display_slice(self.mri[:, self.index, :, :])
        self.mri_on_canvas = self.canvas.create_image(650, 20, anchor=tk.NW, image=mri_img)

    def up_arrow_press(self, event):
        self.index += 1
        if self.index > self.max_index:
            self.index = 0


        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            mri_img = self.display_slice(self.mri[:, self.index, :, :])
            self.canvas.itemconfig(self.mri_on_canvas, image=mri_img)
            self.canvas.mri_img = mri_img

            ct_img = self.display_slice(self.ct[:, self.index, :, :])
            self.canvas.itemconfig(self.ct_on_canvas, image=ct_img)
            self.canvas.ct_img = ct_img


    def down_arrow_press(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = self.max_index

        if self.gt is not None:
            gt_img = self.display_slice(self.gt[:, self.index, :, :])
            self.canvas.itemconfig(self.gt_on_canvas, image=gt_img)
            self.canvas.gt_img = gt_img

            mri_img = self.display_slice(self.mri[:, self.index, :, :])
            self.canvas.itemconfig(self.mri_on_canvas, image=mri_img)
            self.canvas.mri_img = mri_img

            ct_img = self.display_slice(self.ct[:, self.index, :, :])
            self.canvas.itemconfig(self.ct_on_canvas, image=ct_img)
            self.canvas.ct_img = ct_img

    def handle_keys(self):
        # Check if the up or down arrow keys are being held down
        if self.master.state().startswith('pressed'):
            if 'Up' in self.master.state():
                self.up_arrow_press()
            elif 'Down' in self.master.state():
                self.down_arrow_press()

        self.master.after(100, self.handle_keys)

    def display_slice(self, slice):
        # prediction

        gm = slice[1, :, :]
        wm = slice[0, :, :]
        csf = slice[2, :, :]

        wmgm_rgb = np.stack((csf, gm, wm), axis=0) #0 * gm), axis=0)
        np_img = np.moveaxis(np.uint8(wmgm_rgb * 255), source=0, destination=-1)
        pil_img = Image.fromarray(np_img)
        tk_img = ImageTk.PhotoImage(pil_img.resize((300, 300)))

        return tk_img



def load_nii(path, prob_map=False):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)

    if prob_map:
        img = (img > 0.5).astype(np.uint8)

    return img

def load_gt_nii(path, select_all=False, using_3d=False):
    indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(path)), 0)

    valid_slices = range(256)
    if not select_all:
        valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
        if using_3d:
            valid_slices = valid_slices[1:-1, 0] # -2 to account for last and first slice in 2.5D
        else:
            valid_slices = valid_slices[:, 0]
        indxs = indxs[:, valid_slices, :, :]

    CSF = indxs == 3
    GM = indxs == 2
    WM = indxs == 1

    indxs = np.concatenate((WM, GM, CSF), axis=0)

    return indxs, valid_slices

def load_all_nii(mri_path, ct_path, gt_path, using_3d=False):
    orig_mri = load_nii(mri_path)
    mri = np.expand_dims(orig_mri, 0)
    ct = np.expand_dims(load_nii(ct_path), 0)

    gt, valid_slices = load_gt_nii(gt_path, select_all=False, using_3d=using_3d)

    mri = mri[:, valid_slices, :, :]
    mri = mri + np.abs(np.min(mri))
    mri = mri / np.max(mri)
    mri = np.concatenate((mri, mri, mri), axis=0)

    ct = ct[:, valid_slices, :, :]
    ct = ct + np.abs(np.min(ct))
    ct = ct / np.max(ct)
    ct = np.concatenate((ct, ct, ct), axis=0)

    return mri, ct, gt


def calculate_metrics(seg, gt):
    dice = Dice(zero_division=np.nan, ignore_index=0)
    iou = JaccardIndex(task='binary')

    (wm, gm, csf) = np.split(seg, 3, axis=0)
    (wm_t, gm_t, csf_t) = np.split(gt, 3, axis=0)
    ious = [iou(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
            iou(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
            iou(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]
    dscs = [dice(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
            dice(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
            dice(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]

    return ious, dscs



if __name__ == "__main__":
    
    # test "8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"
    # train "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
    #       "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
    #  3mm  "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"  

    #path = "/home/fi5666wi/Brain_CT_MR_data/matlab/8_Ms59/c01_1_00001_temp_8_Ms59_M70_l_T1_CTseg.nii"
    vol_name = "19_LH64"
    path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-28/version_0_prob/{vol_name}_seg.nii.gz"
    gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"
    ct_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_M70_l_T1.nii"
    mr_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_T1.nii"
    seg = load_nii(path, prob_map=True)
    #gt = load_gt_nii(gt_path)
    mri, ct, gt = load_all_nii(mr_path, ct_path, gt_path, using_3d=True)
    print(seg.shape)
    print(gt.shape)
    print(mri.shape)
    print(ct.shape)

    ious, dscs = calculate_metrics(seg, gt)
    print(f"Dice scores (WM/GM/CSF): {np.around(dscs, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(ious, decimals=4)}")

    ious, dscs = calculate_metrics(seg, gt)
    print(f"Dice scores (WM/GM/CSF): {np.around(dscs, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(ious, decimals=4)}")


    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageApp class
    #app = SegmentationGUI2(root, seg, gt, mri, ct)
    app = SegmentationGUI3(root, gt, mri, ct)

    # Run the Tkinter event loop
    root.mainloop()

    





