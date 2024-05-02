import os
import numpy as np
import cv2
import torch
import yaml
import csv
import argparse
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

from brainCT.main import parse_config, get_model
from brainCT.train_utils.data_loader import BrainXLDataset, VotingDataset, MultiModalDataset
from monai.data import DataLoader

from tqdm import tqdm
import scipy.stats as st

def statistics(result, alpha=0.95):
    tstats = {}
    N = len(result["Dice"])
    for metric in result.keys():
        mean = np.mean(result[metric], axis=0)
        std = np.std(result[metric], axis=0)
        conf = [st.norm.interval(alpha, loc=mu, scale=sigma/N) for mu, sigma in zip(mean, std)]
        tstats[metric] = {"mean": np.round(mean, 4), "std": np.round(std, 4), "conf": np.round(conf, 4)}

    return tstats


def get_dataset(config, test_IDs, i3d=False):
    datafolder = os.path.join(config.base_dir, 'DL')
    if not i3d:
        transforms = Compose(
            [ToTensord(keys=["img", "seg"]),
             ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

        level = 70
        test_cases = [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in test_IDs]

        dataset = BrainXLDataset(test_cases, transforms, n_pseudo=config.n_pseudo)
    else:
        transforms = Compose(
            [ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
             ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])

        energies = [50, 70, 120]
        test_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                      for cid in test_IDs]

        dataset = VotingDataset(test_cases, transforms)

    return dataset


def seg_results(config, model_path, test_IDs):
    dataset = get_dataset(config, test_IDs, i3d=config.use_3d_input)
    N_slices = len(dataset)

    model = get_model(config)
    model.load_state_dict(torch.load(os.path.join(model_path, 'last.pth')), strict=True)
    model.eval()
    binarize = AsDiscrete(threshold=0.5)

    # Metrics
    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    results = {"Dice": np.zeros((N_slices, config.n_classes)),
               "IoU": np.zeros((N_slices, config.n_classes)),
               "Hausdorff": np.zeros((N_slices, config.n_classes))}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    loop = tqdm(loader, total=len(loader), position=0, leave=False)
    for k, batch in enumerate(loop):
        if config.use_3d_input:
            input = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
            label = batch["seg"]
        else:
            input, label = (batch["img"], batch["seg"])

        dice_scores = np.zeros(config.n_classes)
        iou_scores = np.zeros(config.n_classes)

        with torch.no_grad():
            output = model(input)
            # Metrics
            if type(output) == list:
                output = output[0]
            if config.sigmoid:
                output = torch.sigmoid(output)

            y_pred = binarize(output)

            for i in range(config.n_classes):
                pred = y_pred[0, i, :, :]
                tar = label[0, i, :, :]
                dice_scores[i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[i] = iou(pred.to(torch.uint8), tar).item()
            hausdorff = hdm(y_pred=y_pred, y=label, spacing=1)

            results["Dice"][k,] = dice_scores
            results["IoU"][k,] = iou_scores
            hausdorff[np.isinf(hausdorff)] = np.nan  # Remove inf values
            results["Hausdorff"][k,] = hausdorff

            loop.set_postfix(dsc=[np.round(t, 4) for t in dice_scores])

    return results

def gen_results(config, model_path, test_IDs):
    config.n_classes = 3
    config.sigmoid = False
    datafolder = os.path.join(config.base_dir, 'DL')
    if not config.use_3d_input:
        transforms = Compose(
            [ToTensord(keys=["img", "mri", "seg"]),
             ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])
        test_cases = [(f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in test_IDs]
        dataset = BrainXLDataset(test_cases, transforms, n_pseudo=config.n_pseudo)
    else:
        transforms = Compose(
            [ToTensord(keys=["img_50", "img_70", "img_120", "mri", "seg"]),
             ScaleIntensityd(keys=["img_50", "img_70", "img_120", "mri"], minv=0.0, maxv=1.0)])

        energies = [50, 70, 120]
        test_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                      + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                      for cid in test_IDs]

        dataset = MultiModalDataset(test_cases, transforms)

    N_slices = len(dataset)

    from brainCT.train_gan import get_model
    model = get_model(config)

    model.load_state_dict(torch.load(os.path.join(model_path, 'last.pth')), strict=True)
    model.eval()
    model.to("cpu")
    binarize = AsDiscrete(threshold=0.5)

    # Metrics
    dsc = Dice(zero_division=np.nan, ignore_index=0)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary')  # MeanIoU(include_background=True)
    hdm = HausdorffDistanceMetric(include_background=True, percentile=95.0)

    results = {"Dice": np.zeros((N_slices, config.n_classes)),
               "IoU": np.zeros((N_slices, config.n_classes)),
               "Hausdorff": np.zeros((N_slices, config.n_classes))}

    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    loop = tqdm(loader, total=len(loader), position=0, leave=False)
    for k, batch in enumerate(loop):
        if config.use_3d_input:
            input = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
            label = batch["seg"]
        else:
            input, label = (batch["img"], batch["seg"])

        dice_scores = np.zeros(config.n_classes)
        iou_scores = np.zeros(config.n_classes)

        with torch.no_grad():
            output, recon = model(input)
            # Metrics
            if type(output) == list:
                output = output[0]
            if config.sigmoid:
                output = torch.sigmoid(output)

            y_pred = binarize(output)

            for i in range(config.n_classes):
                pred = y_pred[0, i, :, :]
                tar = label[0, i, :, :]
                dice_scores[i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[i] = iou(pred.to(torch.uint8), tar).item()
            hausdorff = hdm(y_pred=y_pred, y=label, spacing=1)

            results["Dice"][k,] = dice_scores
            results["IoU"][k,] = iou_scores
            hausdorff[np.isinf(hausdorff)] = np.nan  # Remove inf values
            results["Hausdorff"][k,] = hausdorff

            loop.set_postfix(dsc=[np.round(t, 4) for t in dice_scores])

    return results


def bootstrap(results, iterations=1000, n_classes=3):
    N = len(results["Dice"])

    dasboot = {"Dice": np.zeros((iterations, n_classes)),
               "IoU": np.zeros((iterations, n_classes)),
               "Hausdorff": np.zeros((iterations, n_classes))}
    for i in range(iterations):
        idx = np.random.choice(N, N, replace=True)
        dice = results["Dice"][idx]
        iou = results["IoU"][idx]
        hausdorff = results["Hausdorff"][idx]

        dasboot["Dice"][i] = np.nanmean(dice, axis=0)
        dasboot["IoU"][i] = np.nanmean(iou, axis=0)
        dasboot["Hausdorff"][i] = np.nanmean(hausdorff, axis=0)

    stats = statistics(dasboot)
    print(stats)

    return dasboot


def save_results_as_csv(data, filename):
    field_names = ["Dice WM", "Dice GM", "Dice CSF",
                   "IoU WM", "IoU GM", "IoU CSF",
                   "Hausdorff WM", "Hausdorff GM", "Hausdorff CSF"]
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)


def save_results(res, model_path):
    with open(os.path.join(model_path, "bootstrap_results.yaml"), "w") as f:
        yaml.dump(res, f)

def load_results(model_path):
    with open(os.path.join(model_path, "bootstrap_results.yaml"), "r") as f:
        res = yaml.unsafe_load(f)
    return res


def save_as_csv(data, filename):
    # Save dictionary to a CSV file
    row_names = ['mean', 'std', 'conf']
    col_names = sorted(data.keys())

    # Write nested dictionary to a CSV file
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header row with column names
        writer.writerow([''] + col_names)

        # Write data rows with row names and values
        for row_name in row_names:
            row_data = [data[col_name].get(row_name, '') for col_name in col_names]
            writer.writerow([row_name] + row_data)

def get_args():
    parser = argparse.ArgumentParser("argument for bootstrap")

    parser.add_argument("--model", "-m", type=str, default="unet_plus_plus_3d",
                        help="unet, unet_plus_plus, unet_plus_plus_3d, unet_att, genunet")
    parser.add_argument("--model_name", type=str, default="unet_plus_plus_3d_2024-04-08/")
    parser.add_argument("--version", "-v",  type=int, default=0)

    #parser.add_argument('--use_3d_input', type=bool, default=False)
    #parser.add_argument('--sigmoid', type=bool, default=False)

    parser.add_argument("--iterations", "-i", type=int, default=1000)
    parser.add_argument("--load_results", type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_name = args.model_name
    model_path = os.path.join(save_dir, model_name, f"version_{args.version}")

    with open(os.path.join(model_path, 'config.yaml'), "r") as f:
        config = yaml.safe_load(f)
        config = Namespace(**config)


    datafolder = os.path.join(config.base_dir, 'DL')
    config.model_name = model_name

    config.model = args.model

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    if args.load_results:
        res = load_results(model_path)
    else:
        if args.model == "genunet":
            res = gen_results(config, test_IDs)
        else:
            res = seg_results(config, test_IDs)


    # Save results
    if not args.load_results:
        save_results(res, model_path)

    bs = bootstrap(res, iterations=args.iterations)

    stata = statistics(bs)
    save_as_csv(stata, os.path.join(model_path, "bootstrap_stats.csv"))


