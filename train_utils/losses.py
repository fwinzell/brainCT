import torch
import torch.nn.functional as F
from monai.losses import DiceLoss, SSIMLoss, GeneralizedDiceLoss, FocalLoss, DiceCELoss
from torch.nn import CrossEntropyLoss


class DiceSSIMLoss(torch.nn.Module):
    def __init__(self, beta=0.5, sigmoid=True, class_weights=None):
        super().__init__()
        self.beta = beta
        # assert that beta is between 0 and 1
        assert 0 <= beta <= 1

        self.sigmoid = sigmoid
        self.dice_loss = DiceLoss(include_background=True, to_onehot_y=False, sigmoid=self.sigmoid,
                                  squared_pred=False, weight=class_weights, reduction="mean", jaccard=True)
        self.gd_loss = GeneralizedDiceLoss(include_background=True, sigmoid=self.sigmoid, to_onehot_y=False)
        self.ssim_loss = SSIMLoss(spatial_dims=2, reduction="mean", data_range=1.0)

    def forward(self, seg, labels, recon, target):
        dice_loss = self.dice_loss(seg, labels)
        ssim_loss = self.ssim_loss(recon, target)

        combined = self.beta * dice_loss + (1 - self.beta) * ssim_loss
        return combined


class DiceCESSIMLoss(torch.nn.Module):
    def __init__(self, beta=0.5, sigmoid=True, class_weights=None):
        super().__init__()
        self.beta = beta
        # assert that beta is between 0 and 1
        assert 0 <= beta <= 1

        self.sigmoid = sigmoid
        self.dice_loss = DiceCELoss(include_background=True, to_onehot_y=False, sigmoid=self.sigmoid,
                                  squared_pred=False, weight=class_weights, reduction="mean", jaccard=True)
        self.ssim_loss = SSIMLoss(spatial_dims=2, reduction="mean", data_range=1.0)

    def forward(self, seg, labels, recon, target):
        dice_loss = self.dice_loss(seg, labels)
        ssim_loss = self.ssim_loss(recon, target)

        combined = self.beta * dice_loss + (1 - self.beta) * ssim_loss
        return combined


class CESSIMLoss(torch.nn.Module):
    def __init__(self, beta=0.5, sigmoid=True, class_weights=None):
        super().__init__()
        self.beta = beta
        # assert that beta is between 0 and 1
        assert 0 <= beta <= 1

        self.sigmoid = sigmoid
        self.ce_loss = CrossEntropyLoss(weight=class_weights)
        self.ssim_loss = SSIMLoss(spatial_dims=2, reduction="mean", data_range=1.0)

    def forward(self, seg, labels, recon, target):
        ce_loss = self.ce_loss(seg, labels)
        ssim_loss = self.ssim_loss(recon, target)

        combined = self.beta * ce_loss + (1 - self.beta) * ssim_loss
        return combined


class FocalSSIMLoss(torch.nn.Module):
    def __init__(self, beta=0.5, class_weights=None):
        super().__init__()
        self.beta = beta
        # assert that beta is between 0 and 1
        assert 0 <= beta <= 1

        self.focal_loss = FocalLoss(include_background=True, to_onehot_y=False, use_softmax=False, gamma=2.0,
                                   weight=class_weights, reduction="mean")
        self.ssim_loss = SSIMLoss(spatial_dims=2, reduction="mean", data_range=1.0)

    def forward(self, seg, labels, recon, target):
        focal_loss = self.focal_loss(seg, labels)
        ssim_loss = self.ssim_loss(recon, target)

        combined = self.beta * focal_loss + (1 - self.beta) * ssim_loss
        return combined


if __name__ == "__main__":
    # Example usage:
    loss_fn = DiceSSIMLoss(beta=0.5)
    seg = torch.rand(3, 256, 256)
    labels = torch.randint(0, 1, (3, 256, 256))
    recon = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 1, 256, 256)
    loss = loss_fn(seg, labels, recon, target)
    print(loss)


