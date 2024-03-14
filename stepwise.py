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
from torchmetrics import Dice, JaccardIndex

from main import parse_config, get_model
from brainCT.train_utils.data_loader import InfDataset


def eval(model_paths, test_cases):
    transforms = Compose(
        [ToTensord(keys=["img_70", "img_120", "seg"]),
         ScaleIntensityd(keys=["img_70", "img_120"], minv=0.0, maxv=1.0)])

    dataset = InfDataset(test_cases, transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    with open(os.path.join(model_paths[0], 'config.yaml'), "r") as f:
        config1 = yaml.safe_load(f)
        config1 = Namespace(**config1)
    unet1 = get_model(config1)
    unet1.load_state_dict(torch.load(os.path.join(model_paths[0], 'last.pth')), strict=True)
    unet1.eval()

    with open(os.path.join(model_paths[1], 'config.yaml'), "r") as f:
        config2 = yaml.safe_load(f)
        config2 = Namespace(**config2)
    unet2 = get_model(config2)
    unet2.load_state_dict(torch.load(os.path.join(model_paths[1], 'best.pth')), strict=True)
    unet2.eval()

    binarize = AsDiscrete(threshold=0.5)

    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    dice_scores = np.zeros((len(loader), 2))
    iou_scores = np.zeros((len(loader), 2))

    for k, batch in enumerate(loader):
        im70, im120, seg = (batch["img_70"], batch["img_120"], batch["seg"])
        with torch.no_grad():
            mask = unet1(im120)
            mask = binarize(torch.sigmoid(mask))

            img = im70*mask

            output = unet2(img)
            y_pred = binarize(torch.sigmoid(output))
            if torch.count_nonzero(seg) != 0:
                #display_result(y_pred, seg, n_classes=2, wait=1)
                """
                mask = np.moveaxis(np.uint8(mask.cpu().numpy().squeeze() * 255), source=0, destination=-1)
                disp_70 = np.moveaxis(np.uint8(im70.cpu().numpy().squeeze() * 255), source=0,
                                       destination=-1)
                disp_img = np.moveaxis(np.uint8(img.cpu().numpy().squeeze() * 255), source=0, destination=-1)
                cv2.imshow("image 70", disp_70)
                cv2.imshow("mask", mask)
                cv2.imshow("masked", disp_img)
                cv2.waitKey(100)
                """
                gm = np.moveaxis(np.uint8(y_pred[:, 0, :, :].cpu().numpy().squeeze() * 255), source=0, destination=-1)
                wm = np.moveaxis(np.uint8(y_pred[:, 1, :, :].cpu().numpy().squeeze() * 255), source=0, destination=-1)
                cv2.imshow("gm", gm)
                cv2.imshow("wm", wm)
                cv2.waitKey(100)

            for i in range(2):
                pred = y_pred[0, i, :, :]
                tar = seg[0, i, :, :]
                dice_scores[k, i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[k, i] = iou(pred.to(torch.uint8), tar).item()

    dice_scores = np.nanmean(dice_scores, axis=0)
    iou_scores = np.nanmean(iou_scores, axis=0)
    print(f"Dice scores (WM/GM): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM): {np.around(iou_scores, decimals=4)}")


if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_paths = [os.path.join(save_dir, 'unet_2023-10-24', 'version_0'),
                   os.path.join(save_dir, 'unet_2023-10-25', 'version_0')]

    if os.path.exists(os.path.join(model_paths[-1], 'config.yaml')):
        with open(os.path.join(model_paths[-1], 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    datafolder = os.path.join(config.base_dir, 'DL')

    test_IDs = ["2_Ck79", "3_Cl44", "8_Ms59", "18_MN44", "19_LH64"]  # , "33_ET51"]
    IDs = ["1_Bn52", "4_Jk77", "5_Kg40", "6_Mbr57", "7_Mc43", "11_Lh96",
           "13_NK51", "17_AL67", "20_AR94", "22_CM63", "23_SK52", "24_SE39", "25_HH57",
           "26_LB59", "29_MS42", "31_EM88", "32_EN56", "34_LO45"]

    # tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
    #            for cid in IDs[3:]]

    energies = [70, 120]
    test_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                  for cid in test_IDs]

    eval(model_paths, test_cases)
