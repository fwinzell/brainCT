import torch

class DiceScore(torch.nn.Module):
    def __init__(self, include_background=False, zero_division=0.0):
        super(DiceScore, self).__init__()
        self.include_background = include_background
        self.zero_division = zero_division

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        if self.include_background:
            return self.dice_score(preds, target)
        else:
            return self.foreground_dice(preds, target)


    def dice_score(self, preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
        """
        Computes Dice score for binary 3D segmentation.
        Args:
            preds: torch.Tensor of shape (N, D, H, W) or (D, H, W) with predicted labels (0/1).
            targets: torch.Tensor of same shape with ground-truth labels (0/1).
            epsilon: smoothing constant to avoid division by zero.
        Returns:
            Dice score (float).
        """
        # flatten
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (preds * targets).sum().float()
        union = preds.sum() + targets.sum()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        return dice.item()


    def foreground_dice(self, preds: torch.Tensor, targets: torch.Tensor, epsilon=1e-6):
        """
        Computes Dice score for binary 3D segmentation, ignoring background.
        Args:
            preds: torch.Tensor of shape (N, D, H, W) or (D, H, W) with predicted labels (0/1).
            targets: torch.Tensor of same shape with ground-truth labels (0/1).
            epsilon: smoothing constant to avoid division by zero.
        Returns:
            Dice score (float).
        """
        # Ensure binary {0,1}
        preds = (preds > 0).long()
        targets = (targets > 0).long()

        # Compute intersection & union only for foreground
        intersection = (preds * targets).sum().float()
        foreground_sum = preds.sum() + targets.sum()
        if foreground_sum == 0:  # both empty
            return self.zero_division
        else:
            dice = (2. * intersection + epsilon) / (foreground_sum + epsilon)
            return dice.item()