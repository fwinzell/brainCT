import os
import numpy as np
import torch
from torch import optim, nn, utils, Tensor
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from tqdm import tqdm
import yaml
import cv2

from monai.data import DataLoader
from monai.losses import DiceLoss, GeneralizedDiceLoss, TverskyLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import UNet
from monai.transforms import AsDiscrete
from torchmetrics import Dice, JaccardIndex

from display import display_result


class SegModule(object):
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 batch_size,
                 max_epochs,
                 learning_rate,
                 optimizer_name,
                 optimizer_hparams,
                 save_dir,
                 classes=["wm", "gm", "csf"],
                 loss="dice",
                 sigmoid=True,
                 class_weights=None,
                 lr_schedule="multistep"):
        super().__init__()

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = learning_rate
        self.lr_schedule = lr_schedule
        self.model = model
        self.tr_dataset = train_data
        self.val_dataset = val_data
        self.optimizer_name = optimizer_name
        self.optimizer_hparams = optimizer_hparams
        self.n_classes = len(classes)
        self.labels = classes
        self.save_dir = self.create_save_dir(save_dir)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.binarize = AsDiscrete(threshold=0.5)
        self.sigmoid = sigmoid

        if loss == "dice":
            self.loss_module = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=self.sigmoid,
                                        squared_pred=True, weight=class_weights)
        elif loss == "gdl":
            self.loss_module = GeneralizedDiceLoss(to_onehot_y=False, sigmoid=self.sigmoid, softmax=False,
                                                   include_background=True)
        elif loss == "tversky":
            self.loss_module = TverskyLoss(to_onehot_y=False, sigmoid=self.sigmoid, softmax=False, alpha=0.7, beta=0.3,
                                           smooth_nr=1e-5, smooth_dr=1e-5)
        else:
            assert False, f'Unknown loss: "{loss}"'

        self.dice = Dice(zero_division=np.nan, ignore_index=0).to(self.device)
        self.iou = JaccardIndex(task='binary')
        self.example_input_array = torch.rand(4, 3, 256, 256)

        self.tr_loader = self.train_dataloader()
        self.val_loader = self.val_dataloader()
        self.optimizer, self.scheduler = self.configure_optimizers()

        self.writer = SummaryWriter(os.path.join(self.save_dir, 'logs'))
        self.n_iter = 0
        self.best_acc = 0
        self.abort = False

    def forward(self, input):
        # Forward function that is run when visualizing the graph
        return self.model(input)

    def create_save_dir(self, path):
        dir_exists = True
        i = 0
        while dir_exists:
            save_dir = os.path.join(path, f"version_{str(i)}")
            dir_exists = os.path.exists(save_dir)
            i += 1
        os.makedirs(save_dir)
        return save_dir

    def save_config(self, config):
        with open(os.path.join(self.save_dir, "config.yaml"), "w") as f:
            yaml.dump(vars(config), f)

    def train_dataloader(self):
        # Solved issue: "Unable to display frames in debugger" - set num_workers to 0
        return DataLoader(self.tr_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0)

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_hparams)
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.model.parameters(), **self.optimizer_hparams)
        else:
            assert False, f'Unknown optimizer: "{self.optimizer_name}"'

        if self.lr_schedule == "multistep":
            # We will reduce the learning rate by 0.1 after 50 and 75 epochs
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
        elif self.lr_schedule == "onplateau":
            # Reduce on plateu, max for val dice scores, min for loss
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10,
                                                             threshold=0.001, threshold_mode='rel', cooldown=0,
                                                             min_lr=1e-6, verbose=False)
        else:
            # No learning rate decay
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)

        return optimizer, scheduler

    def _training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        inputs, labels = (batch["img"], batch["seg"])
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        if type(outputs) == list:
            outputs = outputs[0]
        loss = self.loss_module(outputs, labels)

        # Metrics
        if self.sigmoid:
            pred = self.binarize(torch.sigmoid(outputs))
        else:
            pred = self.binarize(outputs)
        dice_score = self._calculate_dice(pred, labels)
        # mean_iou = torch.mean(self.iou(pred, labels))

        return loss, pred, dice_score  # Return tensor to call ".backward" on

    def _validation_step(self, batch, batch_idx):
        inputs, labels = (batch["img"], batch["seg"])
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        if type(outputs) == list:
            outputs = outputs[0]
        # Metrics
        if self.sigmoid:
            pred = self.binarize(torch.sigmoid(outputs))
        else:
            pred = self.binarize(outputs)

        #display_result(pred[0].unsqueeze(dim=0), labels[0], n_classes=self.n_classes, wait=150)
        dice_score = self._calculate_dice(pred, labels)
        return dice_score
        # mean_iou = torch.mean(self.iou(pred, labels))
        # if not torch.any(torch.isnan(torch.Tensor(dice_score))):
        #    self.log("val_gm_dsc", dice_score[1].item())
        #    self.log("val_gm_dsc", dice_score[0].item())

    # Returns True if training was completed, false if aborted
    def train(self):
        for epoch in range(self.max_epochs):
            total_loss = 0
            self.model.train()
            tr_loop = tqdm(self.tr_loader)
            for batch_idx, batch in enumerate(tr_loop):
                self.optimizer.zero_grad()
                loss, pred, dice_score = self._training_step(batch, batch_idx)

                self.writer.add_scalar('train_loss', loss, global_step=self.n_iter)
                total_loss += loss.item()
                for i, c in enumerate(self.labels):
                    self.writer.add_scalar(f"train_{c}_dsc", dice_score[i].item(), global_step=self.n_iter)
                # self.writer.add_scalar("train_gm_dsc", dice_score[1].item(), global_step=self.n_iter)
                # self.writer.add_scalar("train_wm_dsc", dice_score[0].item(), global_step=self.n_iter)

                loss.backward()
                self.optimizer.step()
                self.n_iter += 1

                tr_loop.set_description(f"Epoch [{epoch}/{self.max_epochs}]")
                tr_loop.set_postfix(loss=loss.item(), dsc=[np.round(t.item(), 4) for t in dice_score])

            self.writer.add_scalar('epoch_train_loss', total_loss / len(self.tr_loader), global_step=epoch)

            val_dsc = self.validate(epoch)
            if self.lr_schedule == "onplateau":
                self.scheduler.step(val_dsc)
            else:
                self.scheduler.step()
            if self.abort:
                return False

        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'last.pth'))
        return True

    def validate(self, ep):
        self.model.eval()

        dscs = torch.zeros(self.n_classes, len(self.val_loader))
        with torch.no_grad():
            val_loop = tqdm(self.val_loader)
            for batch_idx, batch in enumerate(val_loop):
                dsc = self._validation_step(batch, batch_idx)

                for i in range(self.n_classes):
                    dscs[i, batch_idx] = dsc[i].item()

                val_loop.set_description("Validation: ")
                val_loop.set_postfix(dsc=[np.round(t.item(), 4) for t in dsc])

        mean_dsc = torch.zeros(self.n_classes)

        for i, c in enumerate(self.labels):
            cdice = dscs[i, ~torch.isnan(dscs[i, :])]
            mean_dsc[i] = torch.mean(cdice)
            self.writer.add_scalar(f"val_{c}_dsc", mean_dsc[i], global_step=ep)

        # Abort if dice is zero for one of the classes, let it warm up for 5 epochs
        if torch.count_nonzero(mean_dsc) < self.n_classes and ep > 5:
            print(f"ERROR: zeros found in validation. Restarting training. "
                  f"{os.path.basename(self.save_dir)} Epoch: {ep}")
            self.abort = True

        self.writer.add_scalar("learning_rate", self.optimizer.param_groups[0]['lr'], global_step=ep)

        val_dsc = torch.mean(mean_dsc)
        if val_dsc > self.best_acc:
            self.best_acc = torch.mean(mean_dsc)
            print(f'____New best model____: {self.best_acc.item()}')
            torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best.pth'))
        return val_dsc

    def _calculate_dice(self, y_pred, y):
        dsc = torch.zeros(self.n_classes)
        for i in range(self.n_classes):
            pred = y_pred[:, i, :, :]
            mask = y[:, i, :, :]
            dsc[i] = self.dice(pred.to(torch.uint8), mask.to(torch.uint8))

        return dsc

    def display_imgs(self, img_1, img_2):
        img_1, img_2 = img_1.detach().cpu().numpy(), img_2.detach().cpu().numpy()
        img_1, img_2 = np.uint8(np.moveaxis(img_1, 0, -1) * 255), np.uint8(np.moveaxis(img_2, 0, -1) * 255)
        cv2.imshow("img_1", img_1)
        cv2.imshow("img_2", img_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


"""    def predict_step(self, batch, **kwargs):
        inputs, labels = (torch.unsqueeze(batch["img"], 0), torch.unsqueeze(batch["seg"], 0))
        outputs = self.model(inputs)
        op = outputs.detach().cpu()
        instance1_pred_img = torch.sigmoid(op).numpy()

        GMpred = instance1_pred_img[0, 1, :, :]
        WMpred = instance1_pred_img[0, 0, :, :]
        GMWM_rgb = torch.stack((GMpred, WMpred, 0 * GMpred), dim=0)

        return torch.flip(GMWM_rgb * 255, dims=[0,1])"""


class SegModule3d(SegModule):
    def __init__(self,
                 model,
                 train_data,
                 val_data,
                 batch_size,
                 max_epochs,
                 learning_rate,
                 optimizer_name,
                 optimizer_hparams,
                 save_dir,
                 classes: list = ["wm", "gm", "csf"],
                 loss: str = "dice",
                 sigmoid: bool = True,
                 class_weights=None,
                 lr_schedule: str ="multistep"):
        super().__init__(model,
                         train_data,
                         val_data,
                         batch_size,
                         max_epochs,
                         learning_rate,
                         optimizer_name,
                         optimizer_hparams,
                         save_dir,
                         classes,
                         loss,
                         sigmoid,
                         class_weights,
                         lr_schedule)

        self.example_input_array = torch.rand(4, 3, 3, 256, 256)


    def _training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        inputs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
        labels = batch["seg"]
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        if type(outputs) == list:
            outputs = outputs[0]
        loss = self.loss_module(outputs, labels)

        # Metrics
        if self.sigmoid:
            pred = self.binarize(torch.sigmoid(outputs))
        else:
            pred = self.binarize(outputs)

        dice_score = self._calculate_dice(pred, labels)


        return loss, pred, dice_score  # Return tensor to call ".backward" on

    def _validation_step(self, batch, batch_idx):
        inputs = torch.stack([batch["img_50"], batch["img_70"], batch["img_120"]], dim=1)
        labels = batch["seg"]
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        if type(outputs) == list:
            outputs = outputs[0]

        # Metrics
        if self.sigmoid:
            pred = self.binarize(torch.sigmoid(outputs))
        else:
            pred = self.binarize(outputs)

        dice_score = self._calculate_dice(pred, labels)
        #display_result(pred[0].unsqueeze(dim=0), labels[0], n_classes=self.n_classes, wait=150)
        return dice_score


