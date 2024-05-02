import torch
import os
import argparse
import datetime
import numpy as np

from brainCT.networks.gan_model import GUNet

from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd
)

from brainCT.train_utils.data_loader import MultiModalDataset
from brainCT.train_utils.modules import GenSegModule

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

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("-e", "--epochs", type=int, default=99)
    parser.add_argument("--input_shape", nargs=3, type=int, default=[3, 256, 256])
    parser.add_argument("--features", nargs="+", type=int, default=[24, 48, 96, 192, 384])
    parser.add_argument("--loss", type=str, default="dicece", help="loss function, one of dice, ce or focal")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=0.01)  # prev 0.01
    parser.add_argument("--class_weights", nargs=3, type=float, default=None) #[0.87521193, 0.85465177, 10.84828136])
    # [0.87521193,  0.85465177, 10.84828136] 1E7/total_volumes
    # [ 0.2663065 ,  0.25394151, 40.91449388] N^2/(total_volumes^2 * 1E4)
    parser.add_argument("--sigmoid", type=bool, default=False)

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
    return GUNet(spatial_dims=2,
                 in_channels=3,
                 out_channels=3,
                 channels=config.features,
                 strides=(2, 2, 4, 4),
                 kernel_size=3,
                 up_kernel_size=3,
                 use_3d_input=config.use_3d_input,
                 out_channels_3d=8).to(device)


def train(config):
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

    tr_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"] for cid in IDs[3:]]

    val_cases = [[f"{datafolder}/{cid}_M{level}_l_T1.nii" for level in energies]
                 + [f"{datafolder}/{cid}_T1.nii", f"{datafolder}/{cid}_seg3.nii"]
                 for cid in IDs[:3]]

    train_transforms = Compose(
        [RandAffined(keys=["img_50", "img_70", "img_120", "mri", "seg"],
                     mode=["bilinear", "bilinear", "bilinear", "bilinear", "nearest"],
                     prob=0.9, shear_range=[0.1, 0.1, 0.1]),
         ToTensord(keys=["img_50", "img_70", "img_120", "mri",  "seg"]),
         RandGaussianSmoothd(keys=["img_50", "img_70", "img_120"], prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0),
                             sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys=["img_50", "img_70", "img_120", "mri"], minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img_50", "img_70", "img_120", "mri", "seg"]),
                              ScaleIntensityd(keys=["img_50", "img_70", "img_120", "mri"], minv=0.0, maxv=1.0)])

    train_dataset = MultiModalDataset(tr_cases, train_transforms)
    val_dataset = MultiModalDataset(val_cases, val_transforms)

    training_finished = False
    while not training_finished:
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

        #summary(unet.to(device), tuple([3] + config.input_shape))

        training_finished = module.train()

    module.save_config(config)  # to .yaml file


if __name__ == "__main__":
    config = parse_config()
    config.date = str(datetime.date.today())
    seed_torch(config.seed)
    train(config)