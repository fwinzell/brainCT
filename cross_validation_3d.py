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

from unets import UNet, UNet3d_AG, UNet_PlusPlus4
from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandGibbsNoised,
    RandZoomd,
    AsDiscrete
)

from torchmetrics import Dice, JaccardIndex

from data_loader import BrainXLDataset, VotingDataset
from modules import SegModule3d
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

    parser.add_argument("--model", type=str, default="unet_plus_plus_3d", help="unet_3d, unet_att, unet_plus_plus_3d")
    parser.add_argument("--n_classes", type=int, default=3, help="2 for only WM and GM, 3 if CSF is included")
    parser.add_argument("--use_3mm", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("--num_folds", type=int, default=5)  # For cross-validation, 4 or 5 (without/with 3mm)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--sigmoid", type=bool, default=False)  # Should be False for unet++
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--loss", type=str, default="dice", help="dice, gdl, tversky")
    parser.add_argument("--class_weights", nargs=3, type=float, default=[0.87521193,  0.85465177, 10.84828136])
    # [0.87521193,  0.85465177, 10.84828136] 1E7/total_volumes
    # [ 0.2663065 ,  0.25394151, 40.91449388] N^2/(total_volumes^2 * 1E4)

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

        if config.model == "unet_3d":
            unet = UNet(
                spatial_dims=2,
                in_channels=3,
                out_channels=config.n_classes,
                channels=(24, 48, 96, 192, 384),
                strides=(2, 2, 2, 2),
                kernel_size=3,
                up_kernel_size=3,
                use_3d_input=True,
                out_channels_3d=8,
            )
        elif config.model == "unet_att":
            unet = UNet3d_AG(in_channels=3,
                             out_channels=config.n_classes,
                             out_channels_3d=8,
                             channels=(24, 48, 96, 192, 384),
                             strides=(2, 2, 2, 2),
                             kernel_size=3,
                             up_kernel_size=3)
        elif config.model == "unet_plus_plus_3d":
            unet = UNet_PlusPlus4(
                spatial_dims=2,
                in_channels=3,
                out_channels=config.n_classes,
                out_channels_3d=8,
                features=(16, 32, 64, 128, 256, 16),
                use_3d_input=True,
                dropout=0.0)

        module = SegModule3d(
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
            loss=config.loss,
            lr_schedule="onplateau",
            sigmoid=config.sigmoid,
            class_weights=config.class_weights,
        )

        # summary(unet.to(device), tuple(config.input_shape))

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


def run_cross_val3d(config, n_folds=4):
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
    ids_3mm = ["25_HH57", "26_LB59", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    train_transforms = Compose(
        [RandAffined(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                     prob=0.9, shear_range=[0.1, 0.1, 0.1]),
         ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
         RandGaussianSmoothd(keys=["img_50", "img_70", "img_120"], prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0),
                             sigma_z=(0.5, 1.5)),
         # RandAdjustContrastd(keys=["img_50", "img_70", "img_120"], prob=0.5, gamma=(0.5, 1.5)),
         # RandGibbsNoised(keys=["img_50", "img_70", "img_120"], prob=0.25, alpha=(0, 0.25)),
         RandZoomd(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                   prob=0.5, min_zoom=1.0, max_zoom=1.5),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
                              ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])

    if config.shuffle:
        random.shuffle(IDs)
        random.shuffle(ids_3mm)

    fold_dict = split_into_folds(IDs, n_folds)
    folds_3mm = split_into_folds(ids_3mm, n_folds)

    if config.use_3mm:
        fold_dict = {k: np.concatenate((fold_dict[k], folds_3mm[n_folds - (k + 1)])) for k in range(n_folds)}

    # Run cross-validation
    cv_dice_scores = np.zeros((n_folds, config.n_classes))
    cv_iou_scores = np.zeros((n_folds, config.n_classes))
    for k in range(n_folds):
        print(f"##### Training on fold: {k} #####")
        torch.cuda.empty_cache()
        val_cases = fold_dict[k]
        tr_cases = np.concatenate([fold_dict[i] for i in range(n_folds) if i != k])

        tr_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] +
                    [f"{datafolder}/{cid}_seg3.nii"] for cid in tr_cases]
        val_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] +
                     [f"{datafolder}/{cid}_seg3.nii"] for cid in val_cases]

        train_dataset = VotingDataset(tr_files, train_transforms)
        val_dataset = VotingDataset(val_files, val_transforms)

        model = train_model(config, save_dir, train_dataset, val_dataset)
        cv_dice_scores[k, :], cv_iou_scores[k, :] = validate(config, model, val_dataset)

    print("#### Finished training ####")
    print(f"Mean Dice (WM/GM/CSF): {np.mean(cv_dice_scores, axis=0)} +/- ({np.std(cv_dice_scores, axis=0)})")
    print(f"Mean IoU (WM/GM/CSF): {np.mean(cv_iou_scores, axis=0)} +/- ({np.std(cv_iou_scores, axis=0)})")

    df = create_dataframe(config, cv_dice_scores, cv_iou_scores)
    csv_path = "/home/fi5666wi/Python/Brain-CT/results_cross_val3d.csv"
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
            inputs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
            labels = batch["seg"]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if type(outputs) == list:
                outputs = outputs[0]

            if config.sigmoid:
                y_pred = binarize(torch.sigmoid(outputs))
            else:
                y_pred = binarize(outputs)

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
        data[f"Dice Score Fold {i + 1}"] = np.mean(dices[i])
        data[f"IoU Fold {i + 1}"] = np.mean(ious[i])

    for j, c in enumerate(["WM", "GM", "CSF"]):
        data[f"Dice Score {c}"] = np.mean(dices[:, j])
        data[f"IoU {c}"] = np.mean(ious[:, j])

    data["Mdsc"] = np.mean(dices)
    data["Miou"] = np.mean(ious)

    return pd.DataFrame(data=data, index=[0])


if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)

    run_cross_val3d(config, n_folds=config.num_folds)
