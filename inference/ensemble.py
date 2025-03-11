import os
import numpy as np
import cv2
import torch
import yaml
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
import SimpleITK as sitk

#from modules import SegModule
from brainCT.main import parse_config, get_model
from brainCT.train_utils.data_loader import Dataset2hD, BrainDataset, SpectralDataset, BrainXLDataset, VotingDataset
from brainCT.inference.inference import save_output
from brainCT.inference.faster_bootstrap import get_dataset
from display import display_result

from time import process_time, time
from datetime import timedelta

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

def get_config(model_path):
    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        with open(os.path.join(model_path, 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    return config

def eval_with_ensemble(cv_dir, test_IDs, config, save=False, save_name="model", batch_size=16, prob_map=False):
    datafolder = os.path.join(config.base_dir, 'DL')
       
    ensemble = []
    for f in os.listdir(cv_dir):
        model_dir = os.path.join(cv_dir, f)
        config = get_config(model_dir)
        model = get_model(config)
        model.load_state_dict(torch.load(os.path.join(model_dir, 'last.pth')), strict=True)
        model.eval()
        ensemble.append(model)

    binarize = AsDiscrete(threshold=0.5)

    results = {}

    t_start = time()
    for cid in test_IDs:
        print(f"Evaluating {cid}...")
        t_vol_start = time()
        #test_cases = [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii")]

        dataset = get_dataset(config, [cid], i3d=config.use_3d_input) #BrainXLDataset(test_cases, transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
        
        out_vol = np.zeros((config.n_classes, len(dataset), 256, 256))
        gt_map = np.zeros((config.n_classes, len(dataset), 256, 256))
        for k, batch in enumerate(loader):
            if config.use_3d_input:
                input = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
                label = batch["seg"]
            else:
                input, label = (batch["img"], batch["seg"])


            with torch.no_grad():
                ensemble_output = []
                for model in ensemble:
                    output = model(input)
                    if type(output) == list:
                        output = output[0]
                    ensemble_output.append(output)

                output = torch.stack(ensemble_output).mean(dim=0)

                if config.sigmoid:
                    y_pred = torch.sigmoid(output)
                else:
                    y_pred = output

                if not prob_map:
                    y_pred = binarize(y_pred)

                end_slice = min((k+1) * batch_size, len(dataset))
                start_slice = k * batch_size
                #y_pred = np.moveaxis(y_pred.cpu().numpy(), 0, 1)
                out_vol[:, start_slice:end_slice, :, :] = np.moveaxis(y_pred.cpu().numpy(), 0, 1)
                gt_map[:, start_slice:end_slice, :, :] = np.moveaxis(label.cpu().numpy(), 0, 1)

        t_vol_end = time()
        t_vol_elapsed = t_vol_end - t_vol_start
        print(f"Time for {cid}: {np.round(t_vol_elapsed, 1)}s")

        if prob_map:
            results[cid] = calculate_metrics(binarize(out_vol), gt_map)
        else:
            results[cid] = calculate_metrics(out_vol, gt_map)

        results[cid]["Time"] = np.round(t_vol_elapsed, 1)

        if save:
            if prob_map:
                save_output(f"{save_name}_prob", out_vol, f"{datafolder}/{cid}_prob.nii")
            else:
                save_output(save_name, out_vol, f"{datafolder}/{cid}.nii")
            print(f"Segmentation saved as {save_name}.nii.gz")

    t_end = time()
    t_elapsed = np.round(t_end - t_start)
    print(f"Total Elapsed Time: {timedelta(seconds=t_elapsed)}")
    return results


def calculate_metrics(seg, gt):
    gt = gt.astype(np.uint8)
    dice = Dice(zero_division=np.nan, ignore_index=0)
    iou = JaccardIndex(task='binary')

    (wm, gm, csf) = np.split(seg, 3, axis=0)
    (wm_t, gm_t, csf_t) = np.split(gt, 3, axis=0)
    ious = [iou(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
            iou(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
            iou(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]
    dscs = [dice(torch.from_numpy(wm), torch.from_numpy(wm_t)).item(),
            dice(torch.from_numpy(gm), torch.from_numpy(gm_t)).item(),
            dice(torch.from_numpy(csf), torch.from_numpy(csf_t)).item()]

    return {'Dice': dscs, 'IoU': ious}

if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"
    model_name = "crossval_2025-01-31"
    cv_dir = os.path.join(save_dir, model_name, "unet_0")

    if os.path.exists(os.path.join(cv_dir, 'version_1', 'config.yaml')):
        with open(os.path.join(cv_dir, 'version_1', 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    # Removed due to insufficient quality on MRI imageE
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42 "26_LB59"
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    results = eval_with_ensemble(cv_dir, test_IDs, config, save=True, save_name=f"{model_name}_ensemble", prob_map=True)

    dice_res = [results[cid]["Dice"] for cid in test_IDs]
    iou_res = [results[cid]["IoU"] for cid in test_IDs]
    t_res = [results[cid]["Time"] for cid in test_IDs]

    print(f"Average dice scores (WM/GM/CSF): {np.around(np.mean(dice_res, axis=0), decimals=4)}")
    print(f"Average IoU scores (WM/GM/CSF): {np.around(np.mean(iou_res, axis=0), decimals=4)}")
    print(f"Average time per volume: {np.mean(t_res)}s")

    # Save results
    table = {"ID": test_IDs, "Dice_WM": [res[0] for res in dice_res], "Dice_GM": [res[1] for res in dice_res],
             "Dice_CSF": [res[2] for res in dice_res], "Time": t_res}
             #"IoU_WM": [res[0] for res in iou_res],
             #"IoU_GM": [res[1] for res in iou_res], "IoU_CSF": [res[2] for res in iou_res], 
    res_df = pd.DataFrame(table)
    res_df.to_csv(f"/home/fi5666wi/Brain_CT_MR_data/ensemble_results/{model_name}.csv")
    