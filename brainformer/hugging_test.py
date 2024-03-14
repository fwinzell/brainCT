from transformers import ViTForImageClassification, ViTImageProcessor

from monai.transforms import (
    Compose,
    ToTensord,
    RandAffined,
    ScaleIntensityd,
    RandGaussianSmoothd
)
from monai.data import DataLoader

import sys
sys.path.append("/usr/matematik/fi5666wi/Python/brainCT")
from brainCT.train_utils.data_loader import HuggingDataset

import torch
import torch.nn as nn


def get_dataloaders():
    datafolder = "/home/fi5666wi/Brain_CT_MR_data/DL"
    energies = [50, 70, 120]

    test_IDs = ["2_Ck79", "3_Cl44", "8_Ms59", "18_MN44", "19_LH64", "33_ET51"]
    IDs = ["1_Bn52", "4_Jk77", "5_Kg40", "6_Mbr57", "7_Mc43", "10_Ca58", "11_Lh96",
           "13_NK51", "16_KS44", "17_AL67", "20_AR94", "22_CM63", "23_SK52", "24_SE39"]
    # "25_HH57", "26_LB59", "28_LO45" ,"29_MS42", "31_EM88", "32_EN56", "34_LO45"] # 3mm


    tr_cases, val_cases = [], []
    for level in energies:
        tr_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[3:]]
        val_cases += [(f"{datafolder}/{cid}_M{level}_l_T1.nii", f"{datafolder}/{cid}_seg3.nii") for cid in IDs[:3]]

    train_transforms = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"]),
         RandGaussianSmoothd(keys="img", prob=0.5, sigma_x=(0.2, 2.0), sigma_y=(0.2, 2.0), sigma_z=(0.5, 1.5)),
         ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])
    val_transforms = Compose([ToTensord(keys=["img", "seg"]),
                              ScaleIntensityd(keys="img", minv=0.0, maxv=1.0)])

    train_dataset = HuggingDataset(tr_cases, train_transforms)
    val_dataset = HuggingDataset(val_cases, val_transforms)

    return (DataLoader(train_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0),
            DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=0))



if __name__ == '__main__':


    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    trl, vdl = get_dataloaders()

    loss_fn = nn.CrossEntropyLoss()

    energies = [50, 70, 120]

    for i, batch in enumerate(trl):
        labels = torch.tensor([energies.index(energy) for energy in batch[1]])
        imgs, _ = (batch[0]["img"], batch[0]["seg"])
        inputs = processor(imgs, return_tensors="pt", padding=True)
        outputs = model(**inputs, labels=labels)
        loss = loss_fn(outputs.logits, labels)



