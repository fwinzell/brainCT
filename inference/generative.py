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
from time import process_time, time
from datetime import timedelta
import pandas as pd

from brainCT.train_gen import get_model
from brainCT.cross_validation_gen import parse_config
from brainCT.train_utils.modules import calculate_ssim
from brainCT.train_utils.data_loader import MultiModalDataset
from brainCT.inference.ensemble import calculate_metrics
from display import display_seg_and_recon
from display_gui import load_gt_nii


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_config(model_path):
    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        with open(os.path.join(model_path, 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    return config


def get_multimodal_dataset(test_IDs):
    transforms = Compose([ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
                          ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    
    energies = [50, 70, 120]
    test_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                  + [f"{datafolder}/{cid}_T1.nii" , f"{datafolder}/{cid}_seg3.nii"]
                  for cid in test_IDs]
    
    return MultiModalDataset(test_cases, transforms)
 

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


def eval_cross_validation(cv_dir, test_IDs, config, save=False, save_name="model", batch_size=16):
    binarize = AsDiscrete(threshold=0.5)

    results = {'Model': [], 
               'Volume': [],
               'Dice_WM': [],
               'Dice_GM': [],
               'Dice_CSF': [],
               'SSIM': [],
               'Time': [],
               'CPU Time': []}

    t_start = process_time()
    real_time_start = time()

    for f in os.listdir(cv_dir):
        print(f"Loading model {f}...")
        model_dir = os.path.join(cv_dir, f)
        config = get_config(model_dir)
        model = get_model(config)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'last.pth')), strict=True)
        model.eval()

        for cid in test_IDs:
            print(f"Evaluating {cid}...")
            cpu_vol_start = process_time()
            t_vol_start = time()

            dataset = get_multimodal_dataset([cid]) 
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
            
            out_vol = np.zeros((config.n_classes, len(dataset), 256, 256))
            gt_map = np.zeros((config.n_classes, len(dataset), 256, 256))
            recon_vol = np.zeros((len(dataset), 256, 256))
            target_vol = np.zeros((len(dataset), 256, 256))
            for k, batch in enumerate(loader):
                inputs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
                label = batch["seg"]
                target = batch["mri"]

                inputs, target, label = inputs.to(device), target.to(device), label.to(device)

                with torch.no_grad():
                    outseg, recon = model(inputs)

                    if config.sigmoid:
                        y_pred = torch.sigmoid(outseg)
                    else:
                        y_pred = outseg

                    end_slice = min((k+1) * batch_size, len(dataset))
                    start_slice = k * batch_size
                    
                    out_vol[:, start_slice:end_slice, :, :] = np.moveaxis(y_pred.cpu().numpy(), 0, 1)
                    gt_map[:, start_slice:end_slice, :, :] = np.moveaxis(label.cpu().numpy(), 0, 1)
                    recon_vol[start_slice:end_slice, :, :] = np.moveaxis(recon.cpu().numpy(), 0, 1)
                    target_vol[start_slice:end_slice, :, :] = np.moveaxis(target.cpu().numpy(), 0, 1)

            t_vol_end = time()
            cpu_vol_end = process_time()
            t_vol_elapsed = t_vol_end - t_vol_start
            print(f"Time for {cid}: {np.round(t_vol_elapsed, 1)}s")

            dice_scores = calculate_metrics(binarize(out_vol), gt_map)['Dice']
            ssim = calculate_ssim(torch.tensor(recon_vol).unsqueeze(0).unsqueeze(0), torch.tensor(target_vol).unsqueeze(0).unsqueeze(0), spatial_dims=3, reduction='none')

            results['Model'].append(f)
            results['Volume'].append(cid)
            results['Dice_WM'].append(dice_scores[0])
            results['Dice_GM'].append(dice_scores[1])
            results['Dice_CSF'].append(dice_scores[2])
            results['SSIM'].append(np.round(ssim, 4))
            results['Time'].append(np.round(t_vol_elapsed, 1))
            results['CPU Time'].append(np.round(cpu_vol_end - cpu_vol_start, 1))

            if save:
                save_dir = os.path.join(save_name, f"{f}_prob")
                save_output(save_dir, out_vol, recon_vol, cid)

    t_end = process_time()
    real_time_end = time()
    cpu_elapsed = np.round(t_end - t_start)
    print(f"Total Elapsed CPU Time: {timedelta(seconds=cpu_elapsed)}")
    print(f"Total Elapsed Real Time: {timedelta(seconds=np.round(real_time_end - real_time_start))}")
    return results


def save_output(model_name, out_vol, recon_vol, test_case):
    dir_name = os.path.join("/home/fi5666wi/Brain_CT_MR_data/OUT/", model_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    case_name = test_case.split("/")[-1].split(".")[0]

    sitk.WriteImage(sitk.GetImageFromArray(out_vol), os.path.join(dir_name, f"{case_name}_seg.nii.gz"), imageIO="NiftiImageIO")
    sitk.WriteImage(sitk.GetImageFromArray(recon_vol), os.path.join(dir_name, f"{case_name}_recon.nii.gz"), imageIO="NiftiImageIO")
    print("Saved output @ ", os.path.join(dir_name, f"{case_name}_seg.nii.gz"))


def ensemble(test_IDs, cv_out_dir, save=True, save_name="ensemble"):
    print("Generating ensemble output...")

    results = {}
    binarize = AsDiscrete(threshold=0.5)

    models = os.listdir(cv_out_dir)
    for cid in test_IDs:
        ensemble_output = []
        ensemble_recon = []
        real_time_start = time()
        for model in models:
            seg_path = os.path.join(cv_out_dir, model, f"{cid}_seg.nii.gz")
            recon_path = os.path.join(cv_out_dir, model, f"{cid}_recon.nii.gz")
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
            ensemble_output.append(torch.tensor(seg))

            recon = sitk.GetArrayFromImage(sitk.ReadImage(recon_path))
            ensemble_recon.append(torch.tensor(recon))

        outseg = torch.stack(ensemble_output).mean(dim=0)
        recon = torch.stack(ensemble_recon).mean(dim=0)

        real_time_end = time()
        t_vol = np.round(real_time_end - real_time_start, 1)
        print(f"Time for {cid}: {t_vol}s")
        gt_map, _ = load_gt_nii(f"/home/fi5666wi/Brain_CT_MR_data/DL/{cid}_seg3.nii", using_3d=True)

        results[cid] = calculate_metrics(binarize(outseg), gt_map)
        results[cid]["Time"] = t_vol

        if save:
            save_output(f"{save_name}_ensemble_prob", outseg, recon, f"{datafolder}/{cid}_prob.nii")

    return results


if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_name = "gen_2025-03-03"
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
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    #generate3d(config, test_IDs, save=False)
    """
    results = eval_cross_validation(os.path.join(save_dir, model_name), test_IDs, config, save=True, save_name=model_name, batch_size=16)

    print(f"Average Dice WM: {np.around(np.mean(results['Dice_WM']), decimals=4)}")
    print(f"Average Dice GM: {np.around(np.mean(results['Dice_GM']), decimals=4)}")
    print(f"Average Dice CSF: {np.around(np.mean(results['Dice_CSF']), decimals=4)}")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"/home/fi5666wi/Brain_CT_MR_data/crossval_results/{model_name}.csv")"""

    # Ensemble
    cv_out_dir = os.path.join("/home/fi5666wi/Brain_CT_MR_data/OUT/", model_name)
    ensemble_results = ensemble(test_IDs, cv_out_dir, save=True, save_name=model_name)

    dice_res = [ensemble_results[cid]["Dice"] for cid in test_IDs]
    t_res = [ensemble_results[cid]["Time"] for cid in test_IDs]

    print(f"Ensemble dice scores (WM/GM/CSF): {np.around(np.mean(dice_res, axis=0), decimals=4)}")

    # Save results
    table = {"ID": test_IDs, "Dice_WM": [res[0] for res in dice_res], "Dice_GM": [res[1] for res in dice_res],
             "Dice_CSF": [res[2] for res in dice_res], "Time": t_res} 
    ens_df = pd.DataFrame(table)
    ens_df.to_csv(f"/home/fi5666wi/Brain_CT_MR_data/ensemble_results/{model_name}.csv")


