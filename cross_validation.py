import torch
import os
import argparse
import datetime
import numpy as np
import random
from torchsummary import summary
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from random import randint

from monai.networks.nets import UNet, AttentionUnet, UNETR, BasicUNetPlusPlus
from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd,
    AsDiscrete
)

from torchmetrics import Dice, JaccardIndex

from data_loader import Dataset2hD, BrainDataset, SpectralDataset, BrainXLDataset
from modules import SegModule
from main import seed_torch, get_model

os.environ['PYDEVD_USE_CYTHON'] = 'NO'
os.environ['PYDEVD_USE_FRAME_EVAL'] = 'NO'

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--model", type=str, default="unet_plus_plus", help="unet, unet_plus_plus, unetr, attention_unet")
    parser.add_argument("--n_classes", type=int, default=3, help="2 for only WM and GM, 3 if CSF is included")
    parser.add_argument("--use_3mm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--num_folds", type=int, default=5)  # For cross-validation, 4 or 5 (without/with 3mm)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--loss", type=str, default="dice", help="dice, gdl, tversky")

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Brain_CT_MR_data")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/Brain-CT/saved_models")

    # Optimizer arguments
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')

    args = parser.parse_args()
    return args


def train_model(config, save_dir, train_dataset, val_dataset):
    training_completed = False
    while not training_completed:

        unet = get_model(config)

        module = SegModule(
            unet.to(device),
            train_dataset,
            val_dataset,
            max_epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            optimizer_name='Adam',
            optimizer_hparams={'lr': config.learning_rate,
                               'betas': config.betas,
                               'eps': config.eps,
                               'weight_decay': config.weight_decay},
            save_dir=save_dir,
            classes=["wm", "gm", "csf"][:config.n_classes],
            loss=config.loss
        )

        #summary(unet.to(device), tuple(config.input_shape))

        training_completed = module.train()
    module.save_config(config)  # to .yaml file

    return module.model

def split_into_folds(IDs, n_folds):
    fold_sizes = np.ones(n_folds) * (len(IDs) // n_folds)
    remainder = np.concatenate((np.ones((len(IDs) % n_folds)), np.zeros(n_folds - (len(IDs) % n_folds))))
    fold_sizes += remainder

    fold_dict = {}
    for k in range(n_folds):
        fold_dict[k] = IDs[int(np.sum(fold_sizes[:k])):int(np.sum(fold_sizes[:k + 1]))]

    return fold_dict

def run_cross_val(config, spectral_mode=False, n_folds=4, one_level_per_case=False):
    # Create save directory
    dir_exists = True
    i = 0
    while dir_exists:
        save_name = "crossval_{}/{}_{}".format(config.date, config.model, i)
        save_dir = os.path.join(config.save_dir, save_name)
        dir_exists = os.path.exists(save_dir)
        i += 1
    os.makedirs(save_dir)

    # Create training folds
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39"]
    ids_3mm = ["25_HH57", "26_LB59", "28_LO45" , "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"] # 3mm

    train_transforms = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"]),
         RandGaussianSmoothd(keys="img", prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0), sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img", "seg"]),
                              ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    if config.shuffle:
        random.shuffle(IDs)
        random.shuffle(ids_3mm)

    fold_dict = split_into_folds(IDs, n_folds)
    folds_3mm = split_into_folds(ids_3mm, n_folds)

    if config.use_3mm:
        fold_dict = {k: np.concatenate((fold_dict[k], folds_3mm[n_folds-(k+1)])) for k in range(n_folds)}

    # Run cross-validation
    cv_dice_scores = np.zeros((n_folds, config.n_classes))
    cv_iou_scores = np.zeros((n_folds, config.n_classes))
    for k in range(n_folds):
        print(f"##### Training on fold: {k} #####")
        torch.cuda.empty_cache()
        val_cases = fold_dict[k]
        tr_cases = np.concatenate([fold_dict[i] for i in range(n_folds) if i != k])

        if spectral_mode:
            tr_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] +
                        [f"{datafolder}/{cid}_seg3.nii"] for cid in tr_cases]
            val_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] +
                         [f"{datafolder}/{cid}_seg3.nii"] for cid in val_cases]

            train_dataset = SpectralDataset(tr_files, train_transforms)
            val_dataset = SpectralDataset(val_files, val_transforms)
        elif one_level_per_case:
            tr_files = [[f"{datafolder}/{cid}_M{energies[randint(0,2)]}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                        for cid in tr_cases]
            val_files = [[f"{datafolder}/{cid}_M{energies[1]}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                         for cid in val_cases]

            train_dataset = BrainXLDataset(tr_files, train_transforms)
            val_dataset = BrainXLDataset(val_files, val_transforms)
        else:
            tr_files, val_files = [], []
            for level in energies:
                tr_files += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii")
                             for cid in tr_cases]
                val_files += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii")
                              for cid in val_cases]

            train_dataset = BrainXLDataset(tr_files, train_transforms)
            val_dataset = BrainXLDataset(val_files, val_transforms)

        model = train_model(config, save_dir, train_dataset, val_dataset)
        cv_dice_scores[k, :], cv_iou_scores[k, :] = validate(config, model, val_dataset)

    print("#### Finished training ####")
    print(f"Mean Dice (WM/GM/CSF): {np.mean(cv_dice_scores, axis=0)} +/- ({np.std(cv_dice_scores, axis=0)})")
    print(f"Mean IoU (WM/GM/CSF): {np.mean(cv_iou_scores, axis=0)} +/- ({np.std(cv_iou_scores, axis=0)})")

    df = create_dataframe(config, cv_dice_scores, cv_iou_scores)
    csv_path = "/home/fi5666wi/Python/Brain-CT/results_cross_val.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)


def validate(config, model, val_dataset):
    model.eval()

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    dsc = Dice(zero_division=np.nan, ignore_index=0).to(device)  # DiceMetric(include_background=True)
    iou = JaccardIndex(task='binary').to(device)  # MeanIoU(include_background=True)
    dice_scores = np.zeros((len(loader), config.n_classes))
    iou_scores = np.zeros((len(loader), config.n_classes))
    binarize = AsDiscrete(threshold=0.5)
    with torch.no_grad():
        val_loop = tqdm(loader)
        for batch_idx, batch in enumerate(val_loop):
            inputs, labels = (batch["img"], batch["seg"])
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if type(outputs) == list:
                outputs = outputs[0]

            y_pred = binarize(torch.sigmoid(outputs))

            for i in range(config.n_classes):
                pred = y_pred[0, i, :, :]
                tar = labels[0, i, :, :]
                dice_scores[batch_idx, i] = dsc(pred.to(torch.uint8), tar).item()
                iou_scores[batch_idx, i] = iou(pred.to(torch.uint8), tar).item()

            val_loop.set_description("Validation: ")
            val_loop.set_postfix(dsc=[np.round(t.item(), 4) for t in dice_scores[batch_idx, :]])

    dice_scores = np.nanmean(dice_scores, axis=0)
    iou_scores = np.nanmean(iou_scores, axis=0)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"IoU scores (WM/GM/CSF): {np.around(iou_scores, decimals=4)}")

    return dice_scores, iou_scores

def create_dataframe(config, dices, ious):
    data = {
        'Date': config.date,
        'Architecture': config.model,
        'Seed': config.seed
    }

    for i in range(config.num_folds):
        data[f"Dice Score Fold {i+1}"] = np.mean(dices[i])
        data[f"IoU Fold {i+1}"] = np.mean(ious[i])

    for j,c in enumerate(["WM", "GM", "CSF"]):
        data[f"Dice Score {c}"] = np.mean(dices[:, j])
        data[f"IoU {c}"] = np.mean(ious[:, j])

    data["Mdsc"] = np.mean(dices)
    data["Miou"] = np.mean(ious)

    return pd.DataFrame(data=data, index=[0])

if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)

    run_cross_val(config, spectral_mode=False, n_folds=config.num_folds)
