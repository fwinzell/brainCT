import os
import numpy as np
import cv2
import torch
import yaml
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
import SimpleITK as sitk

from brainCT.train_gan import parse_config, get_model
from brainCT.train_utils.data_loader import MultiModalDataset
from display import display_seg_and_recon

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def generate3d(config, test_IDs, save=False, save_name="model"):
    transforms = Compose(
        [ToTensord(keys=["img_50", "img_70", "img_120", "mri", "seg"]),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120", "mri"], minv=0.0, maxv=1.0)])

    if save and len(test_IDs) != 1:
        print(f"Only one test case can be saved at a time, defaulting to {test_IDs[0]}.")
        test_IDs = [test_IDs[0]]

    energies = [50, 70, 120]
    test_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                  + [f"{datafolder}/{cid}_T1.nii" , f"{datafolder}/{cid}_seg3.nii"]
                  for cid in test_IDs]

    dataset = MultiModalDataset(test_cases, transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    model = get_model(config)
    model.load_state_dict(torch.load(os.path.join(model_path, 'best.pth')), strict=True)
    model.eval()
    model.to("cpu")
    binarize = AsDiscrete(threshold=0.5)

    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95.0)
    dice_scores = np.zeros((len(loader), 3))
    iou_scores = np.zeros((len(loader), 3))
    hausdorff = np.zeros((len(loader), 3))

    out_vol = np.zeros((3, len(loader), 256, 256))
    for k, batch in enumerate(loader):
        imgs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
        mri = batch["mri"]
        label = batch["seg"]
        with torch.no_grad():
            output, recon = model(imgs)

            if config.sigmoid:
                output = torch.sigmoid(output)
            y_pred = binarize(output)

            if save:
                out_vol[:, k, :, :] = y_pred.squeeze().cpu().numpy()

            if torch.count_nonzero(label) != 0:
                display_seg_and_recon(y_pred, label, recon, mri, wait=1)

            for i in range(3):
                pred = y_pred[0, i, :, :]
                tar = label[0, i, :, :]
                dice_scores[k, i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[k, i] = iou(pred.to(torch.uint8), tar).item()
            hausdorff[k, ] = hdm(y_pred=y_pred, y=label, spacing=1)

    dice_scores = np.nanmean(dice_scores, axis=0)
    # iou_scores = iou_scores[~np.isnan(iou_scores)]
    iou_scores = np.nanmean(iou_scores, axis=0)
    hausdorff[np.isinf(hausdorff)] = np.nan # Remove inf values
    h_distances = np.nanmean(hausdorff, axis=0)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores, decimals=4)}")
    print(f"Hausdorff distance (WM/GM/CSF): {np.around(h_distances, decimals=4)}")
    if save:
        save_output(save_name, out_vol, test_cases[0][0])


def save_output(model_name, out_vol, test_case):
    dir_name = os.path.join("/home/fi5666wi/Brain_CT_MR_data/OUT/", model_name)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    case_name = test_case.split("/")[-1].split(".")[0]

    sitk.WriteImage(sitk.GetImageFromArray(out_vol), os.path.join(dir_name, f"{case_name}_seg.nii.gz"), imageIO="NiftiImageIO")
    print("Saved output @ ", os.path.join(dir_name, f"{case_name}_seg.nii.gz"))



if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_name = "gen_2024-03-13/"
    model_path = os.path.join(save_dir, #'crossval_2024-01-23',
                              model_name, 'version_0')

    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        with open(os.path.join(model_path, 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
            config.use_3d_input = True
    else:
        config = parse_config()

    datafolder = os.path.join(config.base_dir, 'DL')
    config.sigmoid = False
    config.model_name = model_name

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    generate3d(config, test_IDs, save=False)


