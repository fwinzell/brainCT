import numpy as np
import torch

import os
from tqdm import tqdm
import yaml
from argparse import Namespace
import pandas as pd
import SimpleITK as sitk


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

    return indxs


def volume_results(path, gt_path):
    seg = load_nii(path, prob_map=True)
    gt = load_gt_nii(gt_path)

    gt_volumes = np.sum(gt, axis=(1, 2, 3))
    seg_volumes = np.sum(seg, axis=(1, 2, 3))

    #avd = np.abs(gt_volumes - seg_volumes) / gt_volumes

    return {"GT": gt_volumes, "Pred": seg_volumes}


def main_loop(test_ids, model_name):
    results = {"ID": [], "Class": [], "GT": [], "Pred": []}
    classes = ["WM", "GM", "CSF"]

    for ID in test_ids:
        path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/{model_name}/{ID}_prob_seg.nii.gz"
        gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{ID}_seg3.nii"

        res = volume_results(path, gt_path)
        avd = np.abs(res["GT"] - res["Pred"]) / res["GT"]
        print(f"AVD: WM = {avd[0]*100}%, GM = {avd[1]*100}%, CSF = {avd[2]*100}%")

        results["ID"].extend([ID] * 3)
        for i, c in enumerate(classes):
            results["Class"].append(c)
            results["GT"].append(res["GT"][i])
            results["Pred"].append(res["Pred"][i])

    return results



if __name__ == "__main__":
    #vol_name = "26_LB59"
    #path = f"/home/fi5666wi/Brain_CT_MR_data/OUT/crossval_2025-01-28_ensemble_prob/{vol_name}_prob_seg.nii.gz"
    #gt_path = f"/home/fi5666wi/Brain_CT_MR_data/DL/{vol_name}_seg3.nii"

    #res = volume_results(path, gt_path)
    #avd = np.abs(res["GT"] - res["Pred"]) / res["GT"]
    #print(f"AVD: WM = {avd[0]*100}%, GM = {avd[1]*100}%, CSF = {avd[2]*100}%")

    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "33_ET51"]

    model_name = "gen_2025-03-03"
    results = main_loop(test_IDs, f"{model_name}_ensemble_prob")

    df = pd.DataFrame(results)
    csv_path = os.path.join("/home/fi5666wi/Brain_CT_MR_data/volumes_csv/", f"{model_name}-volume_results.csv")
    i = 1
    while os.path.exists(csv_path):
        csv_path = csv_path.replace(".csv", f"_{i}.csv")
        i += 1
    df.to_csv(csv_path, index=False)




