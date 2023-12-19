import os
from _ast import slice

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim
import imageio
import glob
import re
import time
from tqdm import tqdm
import datetime
from regutils import simpleelastix_utils as sutl
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

def soft_dsc(GT,probmask):
    return (2.0 * GT.astype(np.float32) * probmask).sum() / (GT.sum()+probmask.sum())

class BaseExperiment(object):
    """
    A class for running lesion experiments. To use the class, create an instance,
    configure non-default parameters, call initialize(), and then call train_model().
    Parameters can be configured by calling configure_params with a dict or directly
    by setting appropriate member variables. This must be done before the call to
    initialize().

    Example usage:

        config = dict(lr=1e-2)

        experiment = BaseExperiment()
        experiment.configure_params(config)
        experiment.epochs=30
        experiment.initialize()
        experiment.train_model()

    """

    def __init__(self):
        """Sets all parameters to default values."""
        self.netname = "Unet"
        self.net = None

        # Optimizer
        self.lr = 1e-3
        self.weight_decay = 1e-6

        self.val_cache = {}
        self.lr_milestones = []     # Default is to never change the lr

        # architecture settings

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # training settings
        self.epochs = 10000
        self.loss = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=True, squared_pred=True)
        self.val_interval = 1
        self.batch_size = 1   #only 1 or 2 supported
        # Loader settings
        self.data = ""
        self.cores = 0


        self.train_transforms = Compose(
        [RandAffined(keys=["img", "seg"], mode=["bilinear", "nearest"], prob=0.9, shear_range=[(0.1), (0.1), (0.1)]),
         ToTensord(keys=["img", "seg"])])


        self.val_transforms = Compose([ToTensord(keys=["img", "seg"])])





    def configure_params(self, config):
            """
            An optional method for overriding default parameters. This method must be called
            before any other method is called. 'config' is a dict containing updated
            parameters. The keys in config must correspond to member variables. For example:
                config = dict(
                    lr=0.1,
                    epochs=42,
                )

            It is also possible to manually override parameters as follows:
                experiment =  BaseExperiment()
                experiment.lr = 0.1
                experiment.initialize()
                experiment.create_model()
                experiment.train_model()
            """
            # Some simple sanity checks to prevent hard to find errors
            assert(self.net is None)
            assert (set(config.keys())).issubset(set(self.__dict__.keys())), \
                    "Variables in config must be a strict subset of member variables."
            self.__dict__.update(config)

    def initialize(self):
        self.set_paths()
        self.create_model()
        self.create_loaders()
        self.create_optimizer()

    def create_model(self):
        self.net = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2, ).to(self.device)

    def create_loaders(self):
        """
        Create the loaders and transforms for data augmentation. If self.cores > 1 we
        use a multithreaded loader.

        """
        KVP = 70
        training_cases = [(f"{self.datafolder}/Mc43_M{KVP}_l_T1.nii", f"{self.datafolder}/Mc43_seg3.nii"),
                          (f"{self.datafolder}/Mbr57_M{KVP}_l_T1.nii", f"{self.datafolder}/Mbr57_seg3.nii")]

        train_dataset = Dataset2hD(training_cases, self.train_transforms)
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.cores)

        # lets fish out an image for visualizing progress in training. Get a select slice:
        self.train_viz_set = train_dataset[130]

        self.val_cases = [(f"{self.datafolder}/Jk77_M{KVP}_l_T1.nii", f"{self.datafolder}/Jk77_seg3.nii")]
        val_dataset = Dataset2hD(self.val_cases, self.val_transforms)
        self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)


    def create_optimizer(self):
        self.optimizer = torch.optim.Adam( self.net.parameters(), self.lr, weight_decay=self.weight_decay, amsgrad=True)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.lr_milestones,
                                                        gamma=0.1)
    def train_model(self):

        # Soren: there are a few print statements here for my debugging and timing tests.
        # We can get rid of them.
        self.global_step = 0
        n_batches = 0
        pbar = tqdm(total=n_batches, leave=True)
        print("Total training instances: ", len(self.train_loader), "batch size",
              self.batch_size)
        print("Optimizer params: ", self.optimizer)
        print("Scheduler params: ", self.scheduler.state_dict())
        for epoch in range(self.epochs):
            pbar.set_description(f"Epoch {epoch}")

            epoch_train_start_time = time.time()
            epoch_loss = self.train_epoch(pbar)
            epoch_train_end_time = time.time()


                # Log some stuff
            self.writer.add_scalar('Training Loss', epoch_loss, epoch)
            print('LR for this epoch: ', self.scheduler.get_last_lr())
            print("Epoch training time:", epoch_train_end_time - epoch_train_start_time)
            print()

            if (epoch % 10 == 0):
                self.save_model(f'{self.logfolder}/epoch{epoch:03.0f}.pth')


            #if (epoch % self.val_interval == 0):
            #    self.validate_epoch(epoch)

            pbar.reset()
            self.scheduler.step()
            time.sleep(1)  # lets threads finish at epoch end

        self.writer.close()

    def train_epoch(self, pbar,epochnum=None):
        """
        Train for a single epoch and return the training metrics dice, bce, and composite.
        Training can be bypassed if we are in playback mode
        """

        self.net.train()

        epoch_loss = 0
        step = 0

        for batch_data in self.train_loader:
            step += 1
            inputs, labels = (
                batch_data["img"].to(self.device),
                batch_data["seg"].to(self.device),
            )

            outputs = self.net(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            epoch_loss += loss.item()
            print(
                f"{step}/{len(self.train_loader)}"
                f", train_loss: {loss.item():.4f}"
            )

        epoch_loss /= step
        return epoch_loss

    def validate_epoch(self,epochnum):
        """Custom validation. Predict for entire data set. Store images and DSC"""
        self.net.eval()
        with torch.no_grad():
            for case in self.val_cases:
                val_case = [case]
                caseID = os.path.basename(case[1]).replace("_seg3.nii","")
                if case in self.val_cache:
                    val_dataset = self.val_cache[caseID]
                else:
                    val_dataset = Dataset2hD(val_case, self.val_transforms)
                    self.val_cache[caseID] = val_dataset

                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

                GMblock= np.zeros( (len(val_dataset),*val_dataset[0]["img"].shape[1:3]),np.float32)
                WMblock = GMblock.copy()
                grayblock = GMblock.copy()
                GTblock = np.zeros(GMblock.shape,dtype=np.uint8)

                for indx,batch_data in enumerate(val_loader):
                    inputs, labels = (
                        batch_data["img"].to(self.device),
                        batch_data["seg"].to(self.device),
                    )
                    outputs = self.net(inputs)
                    op = outputs.detach().cpu()
                    instance1_pred_img = torch.sigmoid(op).numpy()

                    #lets make an overlay
                    imin = batch_data["img"].numpy()[0,1] * 5 + 30
                    GMpred = instance1_pred_img[0, 1, :, :]
                    WMpred = instance1_pred_img[0, 0, :, :]

                    grayblock[indx,:,:] = np.flipud(imin)
                    GMblock[indx, :, :] = np.flipud(GMpred)
                    WMblock[indx, :, :] = np.flipud(WMpred)
                    GTblock[indx, :, :] = np.flipud(batch_data["seg"].numpy()[0,0]*255+batch_data["seg"].numpy()[0,1]*128)



                slice_selections = {"Jk77":[100,120,140,150]}
                if caseID in slice_selections:
                    grayblock_show = grayblock[slice_selections[caseID],:,:]
                    GMblock_show = GMblock[slice_selections[caseID],:,:]
                    WMblock_show = WMblock[slice_selections[caseID],:,:]
                    GTblock_show = GTblock[slice_selections[caseID],:,:]
                else:
                    grayblock_show = grayblock
                    GMblock_show = GMblock
                    WMblock_show = WMblock
                    GTblock_show = GTblock


                #if int(epochnum) == 0:
                if not os.path.exists(f"{self.logfolder}/{caseID}_GRAY.png"):
                    sutl.np2montage(grayblock_show,f"{self.logfolder}/{caseID}_GRAY.png",range=[0,60],everyNthslice=1,DS=1)
                    sutl.np2montage(GTblock_show, f"{self.logfolder}/{caseID}_GT.png", range=[0, 255], everyNthslice=1,DS=1)

                GMblock_show_mask = GMblock_show > 0.5
                WMblock_show_mask = WMblock_show > 0.5

                GMpred_rgb = 0.5*sutl.np2montage(GMblock_show_mask, '', range = [0, 1.0],everyNthslice=1,DS=1)
                WMpred_rgb = sutl.np2montage(WMblock_show_mask, '', range=[0, 1.0],everyNthslice=1,DS=1)
                #rgb_mix = np.stack((GMpred_rgb[:,:,0],WMpred_rgb[:,:,0],WMpred_rgb[:,:,0]*0),axis=2)
                #rgb_mix[rgb_mix>255] = 255
                rgb_mix = GMpred_rgb+WMpred_rgb
                rgb_mix[rgb_mix > 255] = 255
                imageio.imwrite(f"{self.logfolder}/{caseID}_E{epochnum}.png",np.uint8(rgb_mix))

                #lets get soft DSC for GM and WM

                WMgt = GTblock == 255
                GMgt = GTblock == 128
                gm_dsc = soft_dsc(GMgt,GMblock)
                wm_dsc = soft_dsc(WMgt, WMblock)

                print(f"gm_dsc {gm_dsc}   wm_dsc {wm_dsc}")


        return {"gm_dsc":gm_dsc,"wm_dsc":wm_dsc}


    def save_model(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)

    def restore_model(self):
        pass

    def set_paths(self):
        """Setup paths, data directories, log folders, and logging system."""

        # Needed for Numenta setup

        self.datafolder = "/media/sorenc/STROKE/lund/DL/"
        self.logfolder = f"/media/sorenc/STROKE/lund_git/runs/{self.netname}_{datetime.datetime.isoformat(datetime.datetime.now())}"
        self.writer = SummaryWriter(self.logfolder)


    def modelfolder2validation(self,folder,every=1):
        tmp = [(int(re.search('epoch([0-9]*)\.pth', f).groups(0)[0]), f) for f in glob.glob(os.path.join(folder, "*pth"))]
        models = [k[1] for k in sorted(tmp)]
        #models = models[::every]
        models = models[-10:]
        for cmodel in models:
            self.net.load_state_dict(torch.load(cmodel))
            epoch = int(re.search("epoch([0-9]*)",cmodel).groups()[0])
            epoch = f"{epoch:04d}"
            epoch_results = self.validate_epoch(epoch)

            for name,var in epoch_results.items():
                self.writer.add_scalar(name, var, int(epoch))




# Quick config for testing
config_debug = dict(
    lr=1e-3,
    lr_milestones=[2, 4],
    epochs=10000,
    batch_size=2,
)

# Current best config for testing 9 patients, gets to dice score around 0.5
best_config_9_patients = dict(
    lr=1e-2,
    lr_milestones=[15, 30, 45],
    epochs=60,
    batch_size=2,
)

if __name__ == "__main__":
    experiment = BaseExperiment()
    experiment.configure_params(config_debug)
    experiment.initialize()
    experiment.train_model()
    #experiment.modelfolder2validation("runs/Unet_2022-12-08T10:48:44.964324/",every=100)
    #
