import SimpleITK as sitk
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


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
        pred_img[corr, :] = [0, 255, 0]
        incorr = np.logical_xor(gm, gm_t) + np.logical_xor(wm, wm_t) + np.logical_xor(csf, csf_t)
        pred_img[incorr, :] = [0, 0, 255]

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


def load_nii(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def load_gt_nii(path):
    indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(path)), 0)
    valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
    valid_slices = valid_slices[1:-1, 0] # -2 to account for last and first slice in 2.5D
    indxs = indxs[:, valid_slices, :, :]

    CSF = indxs == 3
    GM = indxs == 2
    WM = indxs == 1

    indxs = np.concatenate((WM, GM, CSF), axis=0)

    return indxs

if __name__ == "__main__":
    path = "/home/fi5666wi/Brain_CT_MR_data/OUT/unet_plus_plus_3d_2024-01-22/8_Ms59_M50_l_T1_seg.nii"
    gt_path = "/home/fi5666wi/Brain_CT_MR_data/DL/8_Ms59_seg3.nii"
    seg = load_nii(path)
    gt = load_gt_nii(gt_path)
    print(seg.shape)
    print(gt.shape)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageApp class
    app = SegmentationGUI(root, seg, gt=gt)

    # Run the Tkinter event loop
    root.mainloop()





