from typing import Dict

import torch
from torch import Tensor

class SegmentationMetrics:
    def __init__(
        self, 
        num_classes: int, 
        eps: float = 1e-5,
        device='cpu'
    ) -> None:
        self.num_classes = num_classes
        self.device = device
        self.eps = eps
        self.reset()


    def reset(self):
        self.confusion_matrix: Tensor = torch.zeros((self.num_classes, self.num_classes), device=self.device)


    def update(
        self, 
        preds: Tensor, 
        targets: Tensor
    ) -> None:

        preds = torch.argmax(preds, dim=1)
        targets = torch.argmax(targets, dim=1)
        
        preds = preds.flatten()
        targets = targets.flatten()
        
        mask = (targets >= 0) & (targets < self.num_classes)
        hist = torch.bincount(
            self.num_classes * targets[mask].to(torch.int64) + preds[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist


    def compute(self) -> Dict[str, Tensor]:
        confusion_matrix = self.confusion_matrix
        
        tp = torch.diag(confusion_matrix)
        fp = confusion_matrix.sum(dim=0) - tp
        fn = confusion_matrix.sum(dim=1) - tp
        
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        
        dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        iou = tp / (tp + fp + fn + self.eps)
        
        return {
            'precision': precision,
            'recall': recall,
            'dice': dice,
            'iou': iou
        }


    def __str__(self) -> str:
        metrics = self.compute()
        class_wise = "\n".join([
            f"Class {i}: Precision: {metrics['precision'][i]:.4f}, "
            f"Recall: {metrics['recall'][i]:.4f}, "
            f"Dice: {metrics['dice'][i]:.4f}, "
            f"IoU: {metrics['iou'][i]:.4f}"
            for i in range(self.num_classes)
        ])
        mean_metrics = "\n".join([
            f"Mean {k.capitalize()}: {v.mean():.4f}"
            for k, v in metrics.items()
        ])
        return f"Segmentation Metrics:\n{class_wise}\n\n{mean_metrics}"