import numpy as np
import SimpleITK as sitk
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import vtk

def make_ax(grid=False):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3] * 2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube():
    ax = make_ax()
    colors = np.array([[['#1f77b430'] * 3] * 3] * 3)
    colors[1, 1, 1] = '#ff0000ff'
    colors[0, :, :] = '#d5f5e330'
    colors = explode(colors)
    filled = explode(np.ones((3, 3, 3)))
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))
    ax.voxels(x, y, z, filled, facecolors=colors, edgecolors='gray', shade=False)
    plt.show()


def load_nii(path):
    img = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(img)
    return img

def load_gt_nii(path):
    indxs = np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(path)), 0)
    valid_slices = np.argwhere(np.squeeze(indxs).sum(axis=2).sum(axis=1) > 0)
    valid_slices = valid_slices[1:-1, 0] # -2 to account for last and first slice in 2.5D
    #indxs = indxs[:, valid_slices, :, :]

    CSF = indxs == 3
    GM = indxs == 2
    WM = indxs == 1

    indxs = np.concatenate((WM, GM, CSF), axis=0)

    return indxs


def downsample_seg(seg, factor):
    """
    Downsample a segmentation by a factor
    :param seg: 3D numpy array
    :param factor: int or tuple of ints
    """
    seg = seg.squeeze()
    assert len(seg.shape) == 3

    if isinstance(factor, int):
        factor = (factor,) * 3
    factor = tuple(int(f) for f in factor)

    valid_slices = np.argwhere(np.squeeze(seg).sum(axis=2).sum(axis=0) > 0)

    cseg = seg[valid_slices, :, :].squeeze()

    # Perform downsampling
    downsampled = cseg[::factor[0], ::factor[1], ::factor[2]]

    return downsampled

def display_csf(seg, ax):
    csf = downsample_seg(seg[2, :, :, :], 3)
    ax.voxels(csf, facecolors='red', edgecolors='red', shade=True)
    plt.show()

def display_wm(seg, ax):
    wm = downsample_seg(seg[0, :, :, :], 3)
    ax.voxels(wm, facecolors='green', edgecolors='green', shade=True)
    plt.show()

def display_gm(seg, ax):
    gm = downsample_seg(seg[1, :, :, :], 3)
    ax.voxels(gm, facecolors='blue', edgecolors='blue', shade=True)
    plt.show()

def display_3d(seg, ax):
    wm = seg[0, :, :, :]
    wm = wm[::4, ::4, ::4]
    gm = seg[1, :, :, :]
    gm = gm[::4, ::4, ::4]
    csf = seg[2, :, :, :]
    csf = csf[::4, ::4, ::4]

    x, y, z = wm.shape
    wm[:int(x/2), :int(y/2), :] = 0
    gm[:int(x/2), :int(y/2), :] = 0
    csf[:int(x/2), :int(y/2), :] = 0


    ax.voxels(wm, facecolors=[0, 1, 0, 0.5], edgecolors='green', shade=True)
    ax.voxels(gm, facecolors=[0, 0, 1, 0.5], edgecolors='blue', shade=True)
    ax.voxels(csf, facecolors=[1, 0, 0, 0.5], edgecolors='red', shade=True)
    plt.show()

def calculate_volume(seg, voxel_size):
    """
    Calculate the volume of each class in the segmentation
    :param seg: 3D numpy array
    :return: list of volumes
    """
    wm = np.squeeze(seg[0, :, :, :])
    gm = np.squeeze(seg[1, :, :, :])
    csf = np.squeeze(seg[2, :, :, :])

    vol_wm = np.sum(wm)
    vol_gm = np.sum(gm)
    vol_csf = np.sum(csf)
    return [vol_wm*voxel_size, vol_gm*voxel_size, vol_csf*voxel_size]

if __name__ == "__main__":

    path = "/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2024-01-15_v3/8_Ms59_M50_l_T1_seg.nii.gz"
    gt_path = "/home/fi5666wi/Brain_CT_MR_data/DL/8_Ms59_seg3.nii"
    seg = load_nii(path)
    gt = load_gt_nii(gt_path)
    print(seg.shape)
    print(gt.shape)

    print(calculate_volume(gt, 0.001))
    print(calculate_volume(seg, 0.001))

    #display_3d(gt, make_ax())

    plot_cube()

    #display_vtk(gt)
    #spiral()

    #ax = make_ax(False)

    #display_csf(gt, ax)
    #display_wm(gt, ax)
    #display_gm(gt, ax)




