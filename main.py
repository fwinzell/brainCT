import torch
import os
import argparse
import datetime
import numpy as np
from torchsummary import summary

from monai.networks.nets import AttentionUnet, BasicUNetPlusPlus  # , UNet
from brainCT.networks.unets import UNet, UNet3d_AG, UNet_PlusPlus4
#from torchProject.unet.unet_model import UNetModel
#from pytorch_bcnn.models import BayesianUNet
from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd
)

from brainCT.train_utils.data_loader import BrainXLDataset, VotingDataset, SpectralDataset
from brainCT.train_utils.modules import SegModule, SegModule3d

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

    parser.add_argument("--model", type=str, default="unet_plus_plus_3d",
                        help="unet, unet_plus_plus, unetr, attention_unet, bayesian_unet")
    parser.add_argument("--features", nargs="+", type=int, default=[24, 32, 64, 128, 256, 32])
    # [16, 32, 64, 128, 256, 32] for UNet++ = 2.3M
    # [24, 48, 96, 192, 384] for UNet = 3.7M
    # [24, 48, 96, 192, 384, 48] for UNet++ = 5.1M
    # [16, 32, 64, 128, 256] for UNet = 1.6M

    parser.add_argument("--sigmoid", type=bool, default=False,
                        help="True for MONAI models, False for UNet++4 and UNet3D_AG")
    parser.add_argument("--n_classes", type=int, default=3, help="2 for only WM and GM, 3 if CSF is included")
    parser.add_argument("--n_pseudo", type=int, default=3, help="Number of slices in psuedo 3D input")
    parser.add_argument("--only_70", type=bool, default=False, help="True if only 70 energy level is used")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--num_folds", type=int, default=6)  # For cross-validation
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--learning_rate", type=float, default=1e-2) # prev 0.01
    parser.add_argument("--class_weights", nargs=3, type=float, default=None) #[0.87521193, 0.85465177, 10.84828136])
    # [0.87521193,  0.85465177, 10.84828136] 1E7/total_volumes
    # [ 0.2663065 ,  0.25394151, 40.91449388] N^2/(total_volumes^2 * 1E4)

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
    
    if not hasattr(config, "n_pseudo"):
        print("Setting n_pseudo to 3")
        config.n_pseudo = 3
    if not hasattr(config, "features"):
        print("Default feature settings")
        if config.model in ["unet", "unet_3d", "unet_att"]:
            config.features = [24, 48, 96, 192, 384]
        elif config.model in ["unet_plus_plus", "unet_plus_plus_3d"]:
            config.features = [16, 32, 64, 128, 256, 16]
    if not hasattr(config, "norm"):
        config.norm = "instance"
    if not hasattr(config, "n_classes"):
        config.n_classes = 3
    

    if config.model == "unet":
        return UNet(
            spatial_dims=2,
            in_channels=config.n_pseudo,
            out_channels=config.n_classes,
            channels=tuple(config.features),
            strides=(2, 2, 2, 2),
            # num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            use_3d_input=config.use_3d_input,
            out_channels_3d=8,
        )
    elif config.model == "unet_plus_plus":
        return BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=config.n_pseudo,
            out_channels=config.n_classes,
            features=tuple(config.features),
            dropout=0.0)
    elif config.model == "unet_att":
        return UNet3d_AG(in_channels=config.n_pseudo,
                         out_channels=config.n_classes,
                         out_channels_3d=8,
                         channels=tuple(config.features), #"(24, 48, 96, 192, 384),
                         strides=(2, 2, 2, 2),
                         kernel_size=3,
                         up_kernel_size=3)
    elif config.model == "unet_plus_plus_3d":
        return UNet_PlusPlus4(
            spatial_dims=2,
            in_channels=config.n_pseudo,
            out_channels=config.n_classes,
            out_channels_3d=8,
            features=tuple(config.features), #(16, 32, 64, 128, 256, 16),
            use_3d_input=True,
            dropout=0.0)
    elif config.model == "unet_3d":
        return UNet(
            spatial_dims=2,
            in_channels=config.n_pseudo,
            out_channels=config.n_classes,
            channels=(24, 48, 96, 192, 384), #tuple(config.features), #(24, 48, 96, 192, 384),
            strides=(2, 2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,
            norm=config.norm,
            use_3d_input=True,
            out_channels_3d=8)
    elif config.model == "attention_unet":
        return AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=config.n_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2))
    else:
        raise ValueError("Model not implemented")



def train_unet(config):
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]
    if config.only_70:
        energies = [70]

    one_case_per_seg = not config.use_3d_input

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42, "26_LB59"
    # test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
            "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"] # 3mm

    perm = torch.randperm(len(IDs))
    IDs = [IDs[i] for i in perm]

    # Training on all cases
    if one_case_per_seg:
        tr_cases, val_cases = [], []
        for level in energies:
            tr_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs]
            val_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[:3]]
    else:
        tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                    for cid in IDs]

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
        train_dataset = BrainXLDataset(tr_cases, train_transforms, n_pseudo=config.n_pseudo)
        val_dataset = BrainXLDataset(val_cases, val_transforms, n_pseudo=config.n_pseudo)
    else:
        train_dataset = SpectralDataset(tr_cases, train_transforms)
        val_dataset = SpectralDataset(val_cases, val_transforms)

    training_finished = False
    while not training_finished:
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
            loss="dice",
            lr_schedule="multistep",
            sigmoid=config.sigmoid  # should be True for MONAI models, and U-Net, False for UNet++ and UNet3D_AG
        )

        # summary(unet.to(device), tuple([3] + config.input_shape))

        training_finished, ep = module.train()

    module.save_config(config)  # to .yaml file


def train_unet3d(config):
    datafolder = os.path.join(config.base_dir, 'DL')
    energies = [50, 70, 120]

    # Removed due to insufficient quality on MRI image
    # 1_BN52, 2_CK79, 3_CL44, 4_JK77, 6_MBR57, 12_AA64, 29_MS42
    test_IDs = ["8_Ms59", "9_Kh43", "18_MN44", "19_LH64", "26_LB59", "33_ET51"]
    IDs = ["5_Kg40", "7_Mc43", "10_Ca58", "11_Lh96", "13_NK51", "14_SK41", "15_LL44",
           "16_KS44", "17_AL67", "20_AR94", "21_JP42", "22_CM63", "23_SK52", "24_SE39",
           "25_HH57", "28_LO45", "27_IL48", "30_MJ80", "31_EM88", "32_EN56", "34_LO45"]  # 3mm

    perm = torch.randperm(len(IDs))
    IDs = [IDs[i] for i in perm]

    # Training on all cases
    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies] + [f"{datafolder}/{cid}_seg3.nii"]
                for cid in IDs]

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

    training_finished = False
    while not training_finished:
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
            lr_schedule="multistep",
            sigmoid=config.sigmoid,
            class_weights=config.class_weights,
        )

        #summary(unet.to(device), tuple([3] + config.input_shape))

        training_finished, ep = module.train()

    module.save_config(config)  # to .yaml file


if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)
    if config.use_3d_input:
        train_unet3d(config)
    else:
        train_unet(config)
