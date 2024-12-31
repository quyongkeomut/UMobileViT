from typing import Any
from torchvision.transforms import v2
import torch

_VALID_NORM = v2.Compose([
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

_NO_MASK_AUG = v2.Compose([
    v2.RandomGrayscale(),
    v2.RandomChoice([
        v2.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.25,
            hue=0.25
        ),
        v2.Identity(),
    ]),
    
    v2.RandomChoice([
        v2.GaussianNoise(sigma=0.1),
        v2.GaussianBlur(kernel_size=5),
        v2.RandomAdjustSharpness(sharpness_factor=2),
        v2.Identity()
    ]),
    # v2.RandomEqualize()
    # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_WITH_MASK_AUG = v2.Compose([
    # v2.RandomResizedCrop(size=512, scale=(0.9, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=25),
    v2.RandomErasing(scale = (0.02, 0.02)),
])


class CustomAug:
    def __init__(self) -> None:
        self.no_mask_aug = _NO_MASK_AUG
        self.with_mask_aug = _WITH_MASK_AUG 
        # self.valid_aug = _VALID_NORM

    def __call__(self, image, mask, valid = False ,*args: Any, **kwds: Any) -> Any:
        if not valid:
            image = self.no_mask_aug(image)

            image, mask = self.with_mask_aug(image, mask)
        # else:
        #   image = self.valid_aug(image)

        return image, mask