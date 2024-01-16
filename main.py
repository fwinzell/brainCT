import torch
import os
import argparse
import datetime
import numpy as np
from torchsummary import summary

from monai.networks.nets import AttentionUnet, UNETR, BasicUNetPlusPlus  # , UNet
from unets import TorchUnet, UNet, UNet3d_AG
from torchProject.unet.unet_model import UNetModel
from pytorch_bcnn.models import BayesianUNet
from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd
)

from data_loader import Dataset2hD, BrainDataset, SpectralDataset, BrainXLDataset, VotingDataset
from modules import SegModule, SegModule3d

os.environ['PYDEVD_USE_CYTHON'] = 'NO'
os.environ['PYDEVD_USE_FRAME_EVAL'] = 'NO'

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("running on: {}".format(device))
print(torch.__version__)


def parse_config():
    parser = argparse.ArgumentParser("argument for run segmentation pipeline")

    parser.add_argument("--model", type=str, default="unet3d_ag",
                        help="unet, unet_plus_plus, unetr, attention_unet, bayesian_unet")
    parser.add_argument("--n_classes", type=int, default=3, help="2 for only WM and GM, 3 if CSF is included")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--num_folds", type=int, default=6)  # For cross-validation
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--learning_rate", type=float, default=0.01)

    parser.add_argument("--base_dir", type=str, default="/home/fi5666wi/Brain_CT_MR_data")
    parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Python/Brain-CT/saved_models")

    # Optimizer arguments
    parser.add_argument("--weight_decay", type=float, default=10e-6)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument('--seed', type=int, default=2,
                        help='random seed for reproducible experiment (default: 1)')

    parser.add_argument('--use_3d_input', type=bool, default=True)

    args = parser.parse_args()
    return args


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_model(config):
    if config.model == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=config.n_classes,
            channels=(24, 48, 96, 192, 384),
            strides=(2, 2, 2, 2),
            # num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            use_3d_input=config.use_3d_input,
            out_channels_3d=8,
        )
    if config.model == "unet3d_ag":
        return UNet3d_AG(in_channels=3,
                         out_channels=config.n_classes,
                         out_channels_3d=8,
                         channels=(24, 48, 96, 192, 384),
                         strides=(2, 2, 2, 2),
                         kernel_size=3,
                         up_kernel_size=3)
    elif config.model == "unetmod":
        return UNetModel(
            input_dim=[256, 256, 3],
            num_classes=config.n_classes,
            depth=4,
            filters=16
        )
    elif config.model == "unet_plus_plus":
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=3,
            out_channels=config.n_classes,
            features=(16, 32, 64, 128, 256, 32),
            dropout=0.0)
    elif config.model == "unetr":
        return UNETR(
            in_channels=3,
            out_channels=config.n_classes,
            img_size=config.input_shape[1],
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            conv_block=True,
            spatial_dims=2)
    elif config.model == "attention_unet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=config.n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2))
    elif config.model == "bayesian_unet":
        return BayesianUNet(ndim=2,
                            in_channels=3,
                            out_channels=config.n_classes,
                            nlayer=4,
                            nfilter=16)
    else:
        raise ValueError("Model not implemented")


def train_unet(config):
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]

    one_case_per_seg = False

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42

    test_IDs = ["8_Ms59", "18_MN44", "19_LH64", "33_ET51"]
    exclude = ["1_Bn52","2_Ck79", "3_Cl44", "4_Jk77", "6_Mbr57", "29_MS42"]
    IDs = [ "5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39"]
    # "25_HH57", "26_LB59", "28_LO45" , "27_IL48" ,, "30_MJ80", "31_EM88", "32_EN56", "34_LO45"] # 3mm

    if one_case_per_seg:
        tr_cases, val_cases = [], []
        for level in energies:
            tr_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]
            val_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[:3]]
    else:
        tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                    for cid in IDs[3:]]

        val_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                     for cid in IDs[:3]]

    train_transforms = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"]),
         RandGaussianSmoothd(keys="img", prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0), sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img", "seg"]),
                              ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    if one_case_per_seg:
        train_dataset = BrainXLDataset(tr_cases, train_transforms)
        val_dataset = BrainXLDataset(val_cases, val_transforms)
    else:
        train_dataset = VotingDataset(tr_cases, train_transforms)
        val_dataset = VotingDataset(val_cases, val_transforms)

    unet = get_model(config).to(device)

    save_name = "{}_{}".format(config.model, config.date)

    module = SegModule(
        unet,
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
        classes=["wm", "gm", "csf"][:config.n_classes],
        loss="dice"
    )

    summary(unet.to(device), tuple(config.input_shape))

    module.train()
    module.save_config(config)  # to .yaml file


def train_unet3d(config):
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]

    test_IDs = ["8_Ms59", "18_MN44", "19_LH64", "33_ET51"]
    exclude = ["1_Bn52", "2_Ck79", "3_Cl44", "4_Jk77", "6_Mbr57", "29_MS42"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39"]
    # "25_HH57", "26_LB59", "28_LO45" , "27_IL48" ,"29_MS42", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"] # 3mm

    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                for cid in IDs[3:]]

    val_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                 for cid in IDs[:3]]

    train_transforms = Compose(
        [RandAffined(keys=["img_50", "img_70", "img_120", "seg"], mode=["bilinear", "bilinear", "bilinear",  "nearest"],
                     prob=0.9, shear_range=[0.1, 0.1, 0.1]),
         ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
         RandGaussianSmoothd(keys=["img_50", "img_70", "img_120"], prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0),
                             sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img_50", "img_70", "img_120", "seg"]),
                              ScaleIntensityd(keys=["img_50", "img_70", "img_120"], minv=0.0, maxv=1.0)])

    train_dataset = VotingDataset(tr_cases, train_transforms)
    val_dataset = VotingDataset(val_cases, val_transforms)

    unet = get_model(config).to(device)

    save_name = "{}_{}".format(config.model, config.date)

    module = SegModule3d(
        unet,
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
        classes=["wm", "gm", "csf"][:config.n_classes],
        loss="dice",
        lr_schedule="none"
    )

    #summary(unet.to(device), tuple([3] + config.input_shape))

    module.train()
    module.save_config(config)  # to .yaml file


if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)
    if config.use_3d_input:
        train_unet3d(config)
    else:
        train_unet(config)
