import os
import numpy as np
import matplotlib.pyplot as plt

from monai.data import DataLoader
import SimpleITK as sitk

from generator import Dataset2hD
import cv2

if __name__ == '__main__':
    datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"
    test_IDs = ["2_Ck79", "3_Cl44", "8_Ms59", "18_MN44", "33_ET51"]
    IDs = ["1_Bn52", "4_Jk77", "5_Kg40", "6_Mbr57", "7_Mc43", "11_Lh96",
           "13_NK51", "17_AL67", "20_AR94", "22_CM63", "23_SK52", "24_SE39"]
    # "25_HH57", "26_LB59", "29_MS42", "31_EM88", "32_EN56", "34_LO45"] # 3mm
    energies = [70]
    tr_cases, val_cases = [], []
    for level in energies:
        tr_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]
        val_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[:3]]


    for i in range(1):
        img = np.array(sitk.GetArrayFromImage(sitk.ReadImage(tr_cases[i][0])))
        indxs = np.array(sitk.GetArrayFromImage(sitk.ReadImage(tr_cases[i][1])))
        print(img.shape)
        for s in range(img.shape[0]):
            slice = img[s,:,:]
            slice = np.uint8((slice / np.max(slice))*255)
            gt = indxs[s,:,:]
            CSF = gt == 3
            GM = gt == 2
            WM = gt == 1
            seg = np.array((WM, GM, CSF))
            seg = np.moveaxis(np.uint8(seg * 255), source=0, destination=-1)
            cv2.imshow("CT image", slice)
            cv2.imshow("Ground truth", seg)
            cv2.waitKey(500)

            print(seg.shape)
            print(slice.shape)


    """for s in range(img.shape[0]):
        slice = img[s, :, :]
        gt = indxs[s, :, :]
        plt.figure(1)
        plt.imshow(slice, cmap='gray')
        plt.figure(2)
        plt.imshow(gt)
        plt.show()
        print(gt.shape)
        print(slice.shape)"""








