import SimpleITK as sitk
import numpy as np
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


class SegmentationGUI:
    def __init__(self, master, seg):
        self.master = master
        self.seg = seg
        self.index = 0
        self.max_index = seg.shape[1] - 1

        self.canvas = tk.Canvas(master, width=320, height=320)
        self.canvas.pack()

        self.master.bind("<Up>", self.up_arrow_press)
        self.master.bind("<Down>", self.down_arrow_press)

        self.master.after(100, self.handle_keys)

        # Load and display the first image
        img = self.display_slice(self.seg[:, self.index, :, :])
        self.image_on_canvas = self.canvas.create_image(10, 10, anchor=tk.NW, image=img)

    def up_arrow_press(self, event):
        self.index += 1
        if self.index > self.max_index:
            self.index = 0
        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

    def down_arrow_press(self, event):
        self.index -= 1
        if self.index < 0:
            self.index = self.max_index
        tk_image = self.display_slice(np.squeeze(self.seg[:, self.index, :, :]))

        # Update the Canvas with the new image
        self.canvas.itemconfig(self.image_on_canvas, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection issues

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

if __name__ == "__main__":
    path = "/home/fi5666wi/Brain_CT_MR_data/OUT/unet_plus_plus_3d_2024-01-22/8_Ms59_M50_l_T1_seg.nii"
    seg = load_nii(path)
    print(seg.shape)

    index = 0
    max_index = seg.shape[1] - 1

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Image Viewer")

    # Create an instance of the ImageApp class
    app = SegmentationGUI(root, seg)

    # Run the Tkinter event loop
    root.mainloop()





