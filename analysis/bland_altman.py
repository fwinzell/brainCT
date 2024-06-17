import numpy as np
import torch
from brainCT.main import parse_config, get_model
from brainCT.train_utils.data_loader import BrainXLDataset, VotingDataset, MultiModalDataset
from monai.transforms import (
    Compose,
    ToTensord,
    ScaleIntensityd,
    AsDiscrete
)
import os
from monai.data import DataLoader
from tqdm import tqdm
import yaml
from argparse import Namespace
import pandas as pd

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


def volume_results(config, model_path, test_IDs):
    results = {"ID": [], "Class": [], "GT": [], "Pred": []}
    classes = ["WM", "GM", "CSF"]

    model = get_model(config)
    model.load_state_dict(torch.load(os.path.join(model_path, 'last.pth')), strict=True)
    model.eval()

    for ID in test_IDs:
        dataset = get_dataset(config, [ID], i3d=config.use_3d_input)
        N_slices = len(dataset)

        binarize = AsDiscrete(threshold=0.5)

        pred_areas = np.zeros((N_slices, config.n_classes))
        gt_areas = np.zeros((N_slices, config.n_classes))

        loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        loop = tqdm(loader, total=len(loader), position=0, leave=False)
        for k, batch in enumerate(loop):
            if config.use_3d_input:
                input = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
                label = batch["seg"]
            else:
                input, label = (batch["img"], batch["seg"])

            with torch.no_grad():
                output = model(input)

                if type(output) == list:
                    output = output[0]
                if config.sigmoid:
                    output = torch.sigmoid(output)

                y_pred = binarize(output)

                pred_areas[k,] = calculate_area(y_pred)
                gt_areas[k,] = calculate_area(label)

                loop.set_postfix(dsc=[np.abs(t) for t in pred_areas[k,]-gt_areas[k,]])

        pred_volumes = np.sum(pred_areas, axis=0)*0.001
        gt_volumes = np.sum(gt_areas, axis=0)*0.001
        results["ID"].extend([ID] * 3)
        for i, c in enumerate(classes):
            results["Class"].append(c)
            results["GT"].append(gt_volumes[i])
            results["Pred"].append(pred_volumes[i])

    return results


def calculate_area(seg):
    """
    Calculate the area of each class in the segmentation
    :param seg: 3D numpy array
    :return: list of areas
    """
    wm = np.squeeze(seg[0, 0, :, :])
    gm = np.squeeze(seg[0, 1, :, :])
    csf = np.squeeze(seg[0, 2, :, :])

    return [np.sum(wm), np.sum(gm), np.sum(csf)]


if __name__ == "__main__":
    save_dir = "/home/fi5666wi/Python/Brain-CT/saved_models"

    model_name = "unet_plus_plus_3d_2024-04-05"
    model_path = os.path.join(save_dir,  # 'crossval_2024-01-23',
                              model_name, 'version_0')

    if os.path.exists(os.path.join(model_path, 'config.yaml')):
        with open(os.path.join(model_path, 'config.yaml'), "r") as f:
            config = yaml.safe_load(f)
            config = Namespace(**config)
    else:
        config = parse_config()

    datafolder = os.path.join(config.base_dir, 'DL')
    config.model_name = model_name
    # if config.use_3d_input and config.model != "unet":
    #   config.model = "unet_plus_plus_3d"

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]

    results = volume_results(config, model_path, test_IDs)

    df = pd.DataFrame(results)
    csv_path = os.path.join("/home/fi5666wi/Brain_CT_MR_data/volumes_csv/", f"{model_name}-volume_results.csv")
    i = 1
    while os.path.exists(csv_path):
        csv_path = csv_path.replace(".csv", f"_{i}.csv")
        i += 1
    df.to_csv(csv_path, index=False)