import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# from utils import BBoxTransform, ClipBoxes
from typing import Optional, List
from functools import partial
# from utils.plot import display
# from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in classification tasks.
    
    The Focal Loss down-weights well-classified examples and focuses more on hard, 
    misclassified examples. This helps in tasks where there is a severe imbalance 
    between classes (e.g., foreground vs background in object detection).
    
    Parameters:
    - alpha (float): Balancing factor for class weights. Defaults to 1. This can be 
      used to balance the contribution of different classes to the loss.
    - gamma (float): Focusing parameter. Defaults to 2. Higher values of gamma 
      reduce the loss contribution from easy-to-classify examples, focusing more on 
      harder, misclassified examples.
    - reduction (str): Specifies the reduction to apply to the output: 'none', 
      'mean' (default), or 'sum'. 'mean' averages the loss, 'sum' adds them up, 
      and 'none' returns the loss for each example.

    Methods:
    - forward(inputs, targets): Computes the Focal Loss.
        - inputs: Predicted logits (raw, unnormalized predictions).
        - targets: Ground truth binary labels (0 or 1).
    
    Returns:
    - The computed Focal Loss value for the batch.
    """
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2, 
        reduction: str = 'mean',
        eps: float = 1e-5
    ) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps


    def forward(
        self, 
        inputs: Tensor, 
        targets: Tensor
    ) -> Tensor:
        # Cross Entropy Loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # p_t is the predicted probability for the true class
        p_t = 1/(torch.exp(BCE_loss) + self.eps)
        
        # Focal Loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float =0.5, 
        beta: float=0.5, 
        eps: float=1e-5
    ) -> None:
        """
        Tversky Loss for imbalanced image segmentation.
        
        Parameters:
        - alpha: weight for false positives.
        - beta: weight for false negatives.
        - eps: small constant to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps


    def forward(
        self, 
        inputs: Tensor, 
        targets: Tensor
    ) -> Tensor:
        # Apply sigmoid if inputs are logits
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate true positives, false positives, and false negatives
        true_pos = (inputs * targets).sum()
        false_neg = ((1 - inputs) * targets).sum()
        false_pos = (inputs * (1 - targets)).sum()

        # Compute Tversky index
        tversky_index = true_pos / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.eps 
        )

        # Return Tversky loss
        return 1 - tversky_index


class TotalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.5
    ) -> None:
        """

        Args:
            alpha (float, optional): weight for focal loss. Defaults to 1..
            beta (float, optional): weight for tversky loss. Defaults to 1..
        """
        super(TotalLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.focal_loss = FocalLoss()
        self.d_tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)
        self.l_tversky_loss = TverskyLoss(alpha=0.9, beta=0.1)
        # self.running_loss = 0


    def forward(self, d_preds, d_targets, l_preds, l_targets):
        """_summary_

        Args:
            inputs (Tensor): input tensor
            targets (Tensor): target tensor

        Returns:
            float: total loss
        """
        d_preds = torch.softmax(d_preds, dim=1)
        l_preds = torch.softmax(l_preds, dim=1)

        d_loss_focal = self.focal_loss(d_preds, d_targets)
        l_loss_focal = self.focal_loss(l_preds, l_targets)

        d_loss_tversky = self.d_tversky_loss(d_preds, d_targets)
        l_loss_tversky = self.d_tversky_loss(d_preds, l_targets)

        total_loss = self.alpha*(d_loss_focal + d_loss_tversky) + self.beta*(l_loss_focal + l_loss_tversky)

        return total_loss


if __name__ == "__main__":

    inputs = torch.randn(4, 2, 256, 256)  # Example predicted logits
    targets = torch.empty(4, 2, 256, 256).random_(2)  # Example binary targets
    
    
    loss = TotalLoss()

    _loss = loss(inputs, targets)
    print(_loss)
