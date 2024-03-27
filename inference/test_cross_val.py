import os
import numpy as np
import cv2
import torch
import yaml
import random
import pandas as pd
from argparse import Namespace
from monai.transforms import (
    Compose,
    ToTensord,
    ScaleIntensityd,
    AsDiscrete
)
from monai.networks.nets import UNet
from torchmetrics import Dice, JaccardIndex
from monai.metrics import HausdorffDistanceMetric

from brainCT.train_utils.modules import SegModule
#from brainCT.main import parse_config, seed_torch, get_model
from brainCT.train_utils.data_loader import Dataset2hD, BrainDataset, SpectralDataset, BrainXLDataset, VotingDataset
from brainCT.cross_validation import split_into_folds, create_dataframe2, parse_config, seed_torch, get_model

from display import display_result

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))


def eval(model, loader):
    binarize = AsDiscrete(threshold=0.5)

    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    dice_scores = np.zeros((len(loader), config.n_classes))
    iou_scores = np.zeros((len(loader), config.n_classes))
    hausdorff = np.zeros((len(loader), config.n_classes))

    for k, batch in enumerate(loader):
        input, label = (batch["img"].to(device), batch["seg"].to(device))
        with torch.no_grad():
            output = model(input)
            if type(output) == list:
                output = output[0]
            # Metrics
            y_pred = binarize(torch.sigmoid(output))
            if False:  # torch.count_nonzero(label) != 0:
                display_result(y_pred, label, n_classes=config.n_classes, wait=1)

            for i in range(config.n_classes):
                pred = y_pred[0, i, :, :].cpu()
                tar = label[0, i, :, :].cpu()
                dice_scores[k, i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[k, i] = iou(pred.to(torch.uint8), tar).item()
            hausdorff[k, ] = hdm(y_pred=y_pred.detach().cpu(), y=label.detach().cpu(), spacing=1)

    dice_scores = np.nanmean(dice_scores, axis=0)
    iou_scores = np.nanmean(iou_scores, axis=0)
    hausdorff[np.isinf(hausdorff)] = np.nan  # Remove inf values
    hausdorff = np.nanmean(hausdorff, axis=0)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores, decimals=4)}")
    print(f"Hausdorff distance (WM/GM/CSF): {np.around(hausdorff, decimals=4)}")

    return dice_scores, iou_scores, hausdorff


def eval3d(model, loader, display=False):
    binarize = AsDiscrete(threshold=0.5)

    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    dice_scores = np.zeros((len(loader), config.n_classes))
    iou_scores = np.zeros((len(loader), config.n_classes))
    hausdorff = np.zeros((len(loader), config.n_classes))

    for k, batch in enumerate(loader):
        imgs = torch.stack([batch["img_50"].to(device), batch["img_70"].to(device), batch["img_120"].to(device)], dim=1)
        label = batch["seg"].to(device)
        with torch.no_grad():
            output = model(imgs)
            # Metrics
            if type(output) == list:
                output = output[0]
            if config.sigmoid:
                output = torch.sigmoid(output)

            y_pred = binarize(output)

            if torch.count_nonzero(label) != 0 and display:
                display_result(y_pred, label, n_classes=config.n_classes, wait=50)

            for i in range(config.n_classes):
                pred = y_pred[0, i, :, :]
                tar = label[0, i, :, :]
                dice_scores[k, i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[k, i] = iou(pred.to(torch.uint8), tar).item()
            hausdorff[k,] = hdm(y_pred.to(torch.uint8), label).item()

    dice_scores = np.nanmean(dice_scores, axis=0)
    iou_scores = np.nanmean(iou_scores, axis=0)
    hausdorff[np.isinf(hausdorff)] = np.nan  # Remove inf values
    hausdorff = np.nanmean(hausdorff, axis=0)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores, decimals=4)}")
    print(f"Hausdorff distance (WM/GM/CSF): {np.around(hausdorff, decimals=4)}")

    return dice_scores, iou_scores, hausdorff


def run(config, fold_dict, cv_dir, datafolder="/home/fi5666wi/Brain_CT_MR_data/DL", use_test_set=False, test_IDs=None):
    transforms = Compose([ToTensord(keys=["img", "seg"]),
                          ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    cv_dice_scores = np.zeros((config.num_folds, config.n_classes))
    cv_iou_scores = np.zeros((config.num_folds, config.n_classes))
    cv_hd = np.zeros((config.num_folds, config.n_classes))
    energy = 70
    for k in range(config.num_folds):
        if use_test_set:
            files = [(f"{datafolder}/{cid}_M{energy}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in test_IDs]
        else:
            files = [(f"{datafolder}/{cid}_M{energy}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in fold_dict[k]]
        dataset = BrainXLDataset(files, transforms, n_pseudo=config.n_pseudo)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        model = get_model(config).to(device)

        model_path = os.path.join(cv_dir, f"version_{k}")
        model.load_state_dict(torch.load(os.path.join(model_path, 'last.pth')), strict=True)
        model.eval()

        cv_dice_scores[k, :], cv_iou_scores[k, :], cv_hd[k, :] = eval(model, loader)

    cv_dice_scores[cv_dice_scores == 0] = np.nan
    cv_iou_scores[cv_iou_scores == 0] = np.nan
    cv_hd[cv_hd == 0] = np.nan
    print(f"Mean Dice (WM/GM/CSF): {np.nanmean(cv_dice_scores, axis=0)} +/- ({np.nanstd(cv_dice_scores, axis=0)})")
    print(f"Mean IoU (WM/GM/CSF): {np.nanmean(cv_iou_scores, axis=0)} +/- ({np.nanstd(cv_iou_scores, axis=0)})")
    print(f"Mean Hausdorff distance (WM/GM/CSF): {np.nanmean(cv_hd, axis=0)} +/- ({np.nanstd(cv_hd, axis=0)})")

    df = create_dataframe2(config, cv_dice_scores, cv_iou_scores, cv_hd)
    csv_path = "/home/fi5666wi/Python/Brain-CT/results_cross_val2.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)



def run_3d(config, fold_dict, cv_dir, datafolder="/home/fi5666wi/Brain_CT_MR_data/DL", use_test_set=False,
           test_IDs=None):
    transforms = Compose(
        [ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])

    cv_dice_scores = np.zeros((config.num_folds, config.n_classes))
    cv_iou_scores = np.zeros((config.num_folds, config.n_classes))
    cv_hd = np.zeros((config.num_folds, config.n_classes))
    energies = [50, 70, 120]
    for k in range(config.num_folds):
        if use_test_set:
            files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                     for cid in test_IDs]
        else:
            files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                     for cid in fold_dict[k]]
        dataset = VotingDataset(files, transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        model = get_model(config).to(device)

        model_path = os.path.join(cv_dir, f"version_{k}")
        model.load_state_dict(torch.load(os.path.join(model_path, 'last.pth')), strict=True)
        model.eval()

        cv_dice_scores[k, :], cv_iou_scores[k, :], cv_hd[k, :] = eval3d(model, loader)

    cv_dice_scores[cv_dice_scores == 0] = np.nan
    cv_iou_scores[cv_iou_scores == 0] = np.nan
    cv_hd[cv_hd == 0] = np.nan
    print(f"Mean Dice (WM/GM/CSF): {np.nanmean(cv_dice_scores, axis=0)} +/- ({np.nanstd(cv_dice_scores, axis=0)})")
    print(f"Mean IoU (WM/GM/CSF): {np.nanmean(cv_iou_scores, axis=0)} +/- ({np.nanstd(cv_iou_scores, axis=0)})")
    print(f"Mean Hausdorff distance (WM/GM/CSF): {np.nanmean(cv_hd, axis=0)} +/- ({np.nanstd(cv_hd, axis=0)})")

    df = create_dataframe2(config, cv_dice_scores, cv_iou_scores, cv_hd)
    csv_path = "/home/fi5666wi/Python/Brain-CT/results_cross_val2.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)


if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    date = "2024-03-25"
    cv_dir = os.path.join(save_dir,
                          #'crossval_2023-10-12', 'unet_plus_plus_0')
                          f'crossval_{date}', 'unet_plus_plus_0')

    use_test_set = True
    use_3d_input = False

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57",  "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]

    if os.path.exists(os.path.join(cv_dir, "version_0", 'config.yaml')):
        with open(os.path.join(cv_dir, "version_0", 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    #config.use_3d_input = use_3d_input
    #config.sigmoid = True
    config.shuffle = True
    config.use_3mm = True
    config.date = date

    seed_torch(config.seed)

    if config.shuffle:
        random.shuffle(IDs)
        #random.shuffle(ids_3mm)

    fold_dict = split_into_folds(IDs, config.num_folds)
    #folds_3mm = split_into_folds(ids_3mm, config.num_folds)

    #if config.use_3mm:
    #    fold_dict = {k: np.concatenate((fold_dict[k], folds_3mm[config.num_folds - (k + 1)]))
    #                 for k in range(config.num_folds)}

    if use_3d_input:
        #config.model = "unet_plus_plus_3d"
        run_3d(config, fold_dict, cv_dir, use_test_set=use_test_set, test_IDs=test_IDs)
    else:
        run(config, fold_dict, cv_dir, use_test_set=use_test_set, test_IDs=test_IDs)
