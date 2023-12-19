import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
import imageio
torch.backends.cudnn.benchmark = True
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    CenterSpatialCropd,
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    ToTensord,
    RandAffined,
)
from monai.utils import set_determinism
from torch.utils.tensorboard import SummaryWriter
from generator import Dataset2hD
from monai.visualize import plot_2d_or_3d_image


root_dir="MODELS"

train_transforms = Compose( [ RandAffined(keys=["img","seg"],mode = ["bilinear", "nearest"], prob=0.9, shear_range= [(0.1),(0.1),(0.1)]),
                         ToTensord(keys=["img", "seg"]) ])
#train_transforms = Compose( [
#                         ToTensord(keys=["img", "seg"]) ])
val_transforms = Compose( [ToTensord(keys=["img", "seg"]) ])

datafolder = "/home/fi5666wi/Documents/Brain_CT_Project/DL/"
training_cases = [(f"{datafolder}/Mc43_M70_l_T1.nii",f"{datafolder}/Mc43_seg3.nii") , (f"{datafolder}/Mbr57_M70_l_T1.nii",f"{datafolder}/Mbr57_seg3.nii")  ]
train_dataset = Dataset2hD(training_cases,train_transforms)

train_loader =  DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

#lets fish out an image for visualizing progress in training. Get a select slice:
train_viz_set = train_dataset[130]

val_cases = [(f"{datafolder}/Jk77_M70_l_T1.nii",f"{datafolder}/Jk77_seg3.nii")  ]
val_dataset = Dataset2hD(val_cases,val_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device("cuda:0")
model = UNet(
    dimensions=2,
    in_channels=3,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,).to(device)
loss_function = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(
    model.parameters(), 1e-4, weight_decay=1e-6, amsgrad=True
)  #  LR 1e-4 orig


image_instance_validation = 1


max_epochs = 2000
val_interval = 5
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
writer = SummaryWriter()
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["img"].to(device),
            batch_data["seg"].to(device),
        )
        '''
        plt.figure()
        plt.cla()
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(batch_data["img"][0, i, :, :].detach().cpu(), cmap="gray")
        plt.show()

        plt.figure()
        plt.cla()
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(batch_data["seg"][0, i, :, :].detach().cpu(), cmap="gray")
        plt.show()
        '''
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_dataset) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
        )
        writer.add_scalar("train_loss", loss.item(), (len(train_dataset) // train_loader.batch_size) * epoch + step)

    op = outputs.detach().cpu()
    instance1_pred_img = torch.sigmoid(op).numpy()

    IMin = batch_data["img"][0,1,:,:]*5+30
    trainseg = batch_data["seg"]
    # IMin = IMin/60.0
    # IMin[IMin>1] = 1
    # GMpred = instance1_pred_img[0, 1, :, :]
    # WMpred = instance1_pred_img[0, 0, :, :]
    # GMtrain = trainseg[0, 1, :, :]
    # WMtrain = trainseg[0, 0, :, :]
    #
    # GMpred[GMpred<0] = 0
    # GMpred[GMpred > 1] = 1
    # WMpred[WMpred < 0] = 0
    # WMpred[WMpred > 1] = 1
    #
    # mont = np.concatenate((IMin, GMpred, WMpred,GMpred+WMpred), axis=1)
    # mont_train = np.concatenate( (IMin,GMtrain,WMtrain,GMtrain+WMtrain),axis=1)
    # mont = np.concatenate((mont_train,mont),axis=0)
    # imageio.imwrite(f"img_{epoch:03f}.png",(mont*255).astype(np.uint8))
    # gr = np.expand_dims((mont*255).astype(np.uint8),0)
    # rgb = np.concatenate( (gr,gr,gr),axis=0 )
    # #plot_2d_or_3d_image(rgb, epoch + 1, writer, index=0, tag="train-label")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            batch_data = train_viz_set
            inputs, labels = (torch.unsqueeze(batch_data["img"],0).to(device), torch.unsqueeze(batch_data["seg"],0).to(device),)
            outputs = model(inputs)
            op = outputs.detach().cpu()
            instance1_pred_img = torch.sigmoid(op).numpy()

            imin = batch_data["img"].numpy()*5+30
            GMpred = instance1_pred_img[0, 1, :, :]
            WMpred = instance1_pred_img[0, 0, :, :]

            GMWM_rgb = np.stack( (GMpred,WMpred,0*GMpred),axis=0)

            writer.add_image("train-label GM", np.flip(GMWM_rgb*255,axis=1), epoch + 1)

