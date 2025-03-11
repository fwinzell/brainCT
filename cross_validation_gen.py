import torch
import os
import argparse
import datetime
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from brainCT.networks.gen_model import GUNet
from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd,
    RandZoomd,
    AsDiscrete
)

from torchmetrics import Dice, JaccardIndex

from brainCT.train_utils.data_loader import MultiModalDataset
from brainCT.train_utils.modules import GenSegModule, calculate_ssim
from brainCT.train_gen import get_model
from brainCT.main import seed_torch
from brainCT.cross_validation_3d import split_into_folds

os.environ['PYDEVD_USE_CYTHON'] = 'NO'
os.environ['PYDEVD_USE_FRAME_EVAL'] = 'NO'

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))

def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--features", nargs="+", type=int, default=[24, 48, 96, 192, 384])
    parser.add_argument("--n_classes", type=int, default=3, help="2 for only WM and GM, 3 if CSF is included")
    parser.add_argument("--loss", type=str, default="multiclass", help="loss function, one of dice, ce or focal")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.01)  # prev 0.01
    parser.add_argument("--class_weights", nargs=3, type=float, default=None) #[0.87521193, 0.85465177, 10.84828136])
    # [0.87521193,  0.85465177, 10.84828136] 1E7/total_volumes
    # [ 0.2663065 ,  0.25394151, 40.91449388] N^2/(total_volumes^2 * 1E4)
    parser.add_argument("--sigmoid", type=bool, default=False)

    parser.add_argument("--num_folds", type=int, default=7)  # 7 fold CV
    parser.add_argument("--n_pseudo", type=int, default=3, help="Number of slices in psuedo 3D input")
    parser.add_argument("--norm", type=str, default="instance", help="batch, instance")
    parser.add_argument("--shuffle", type=bool, default=True)

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Brain_CT_MR_data")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/Brain-CT/saved_models")

    # Optimizer arguments
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument('--seed', type=int, default=1,
                        help='random seed for reproducible experiment (default: 1)')

    parser.add_argument('--use_3d_input', type=bool, default=True)

    args = parser.parse_args()
    return args

def train_model(config, save_dir, train_dataset, val_dataset):
    training_completed = False
    while not training_completed:
        model = get_model(config).to(device)

        save_name = "gen_{}".format(config.date)

        module = GenSegModule(
            model,
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
            save_dir=os.path.join(config.save_dir, save_name),
            lr_schedule="None",
            loss=config.loss,
            sigmoid=True,
            beta=config.beta,  # ratio of dice loss to ssim loss
            class_weights=config.class_weights,
        )

        # summary(unet.to(device), tuple(config.input_shape))

        training_completed = module.train()
    module.save_config(config)  # to .yaml file

    return module.model


def run_cv(config, n_folds=4):
    # Create save directory
    dir_exists = True
    i = 0
    while dir_exists:
        save_name = "crossval_{}/unet_gen_{}".format(config.date, i)
        save_dir = os.path.join(config.save_dir, save_name)
        dir_exists = os.path.exists(save_dir)
        i += 1
    os.makedirs(save_dir)

    # Create training folds
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]

    IDs = ["5_Kg40" ,"7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]


    train_transforms = Compose(
        [RandAffined(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                     prob=0.9, shear_range=[0.1, 0.1, 0.1]),
         ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
         RandGaussianSmoothd(keys=["img_50", "img_70", "img_120"], prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0),
                             sigma_z=(0.5, 1.5)),
         RandZoomd(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear", "nearest"],
                   prob=0.5, min_zoom=1.0, max_zoom=1.5),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
                              ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])

    if config.shuffle:
        random.shuffle(IDs)
        

    fold_dict = split_into_folds(IDs, n_folds)

    # Run cross-validation
    cv_dice_scores = np.zeros((n_folds, config.n_classes))
    cv_ssim_scores = np.zeros(n_folds)
    for k in range(n_folds):
        print(f"##### Training on fold: {k} #####")
        torch.cuda.empty_cache()
        val_cases = fold_dict[k]
        tr_cases = np.concatenate([fold_dict[i] for i in range(n_folds) if i != k])

        tr_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"] for cid in tr_cases]

        val_files = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                    + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                    for cid in val_cases]

        train_dataset = MultiModalDataset(tr_files, train_transforms)
        val_dataset = MultiModalDataset(val_files, val_transforms)

        model = train_model(config, save_dir, train_dataset, val_dataset)
        cv_dice_scores[k, :], cv_ssim_scores[k] = validate(config, model, val_dataset)

    print("#### Finished training ####")
    print(f"Mean Dice (WM/GM/CSF): {np.mean(cv_dice_scores, axis=0)} +/- ({np.std(cv_dice_scores, axis=0)})")
    print(f"Mean SSIM: {np.mean(cv_ssim_scores)} +/- ({np.std(cv_ssim_scores)})")

    df = create_dataframe(config, cv_dice_scores, cv_ssim_scores)
    csv_path = "/home/fi5666wi/Python/Brain-CT/results_cross_val3d.csv"
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False)
    else:
        df.to_csv(csv_path)


def validate(config, model, val_dataset):
    model.eval()

    loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    dsc = Dice(zero_division=np.nan, ignore_index=0).to(device)  # DiceMetric(include_background=True)
    
    dice_scores = np.zeros((len(loader), config.n_classes))
    ssim_scores = np.zeros(len(loader))
    binarize = AsDiscrete(threshold=0.5)
    with torch.no_grad():
        val_loop = tqdm(loader)
        for batch_idx, batch in enumerate(val_loop):
            inputs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
            target = batch["mri"]
            labels = batch["seg"]

            inputs, target, labels = inputs.to(device), target.to(device), labels.to(device)
            outseg, recon = model(inputs)

            if config.sigmoid:
                y_pred = binarize(torch.sigmoid(outseg))
            else:
                y_pred = binarize(outseg)

            for i in range(config.n_classes):
                pred = y_pred[:, i, :, :]
                tar = labels[:, i, :, :]
                dice_scores[batch_idx, i] = dsc(pred.to(torch.uint8), tar).item()
            
            ssim_scores[batch_idx] = calculate_ssim(recon, target)

            val_loop.set_description("Validation: ")
            val_loop.set_postfix(dsc=[np.round(t.item(), 4) for t in dice_scores[batch_idx, :]])

    dice_scores = np.nanmean(dice_scores, axis=0)
    ssim_score = np.nanmean(ssim_scores, axis=0)
    print(f"Dice scores (WM/GM/CSF): {np.around(dice_scores, decimals=4)}")
    print(f"SSIM score : {np.around(ssim_score, decimals=4)}")

    return dice_scores, ssim_score

def create_dataframe(config, dices, ssims):
    data = {
        'Date': config.date,
        'Architecture': 'UNet (Gen)',
        'Seed': config.seed
    }

    for i in range(config.num_folds):
        data[f"Dice Score Fold {i + 1}"] = np.mean(dices[i])
        data[f"SSIM Fold {i + 1}"] = ssims[i]

    for j, c in enumerate(["WM", "GM", "CSF"]):
        data[f"Dice Score {c}"] = np.mean(dices[:, j])

    data["Mdsc"] = np.mean(dices)
    data["Mssim"] = np.mean(ssims)

    return pd.DataFrame(data=data, index=[0])


if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)

    run_cv(config, n_folds=config.num_folds)