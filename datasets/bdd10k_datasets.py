import cv2
import numpy as np
import torch
import os
import random
import math

CLASSES = {
        0:  'road',
        1:  'sidewalk',
        2:  'building',
        3:  'wall',
        4:  'fence',
        5:  'pole',
        6:  'traffic light',
        7:  'traffic sign',
        8:  'vegetation',
        9:  'terrain',
        10: 'sky',
        11: 'person',
        12: 'rider',
        13: 'car',
        14: 'truck',
        15: 'bus',
        16: 'train',
        17: 'motorcycle',
        18: 'bicycle',
        }

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)



    combination = (img, gray)
    return combination

class BDD10KDataset(torch.utils.data.Dataset):
    def __init__(self,
            data_dir: str = "./data/bdd10k",
            valid: bool = False,
            transform  = None, 
    ):
        self.data_dir = data_dir
        self.valid = valid
        if not valid:
            self.images_dir = os.path.join(data_dir, "images", "10k", "train")
            self.labels_dir = os.path.join(data_dir, "labels", "sem_seg", "masks","train")
        else:
            self.images_dir = os.path.join(data_dir, "images", "10k", "val")
            self.labels_dir = os.path.join(data_dir, "labels", "sem_seg", "masks","val")
            
        self.images_path = os.listdir(self.images_dir)
        self.labels_path = os.listdir(self.labels_dir)
        self.transform = transform
    def __len__(self):
        
        return len(self.images_path)

    def __getitem__(self, index):
        W_=640
        H_=320
        image_path = os.path.join(self.images_dir, self.images_path[index])
        label_path = os.path.join(self.labels_dir, self.labels_path[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (W_, H_))

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label, (W_, H_))
        label += 1
        label[label==256] = 0

        if not self.valid:
            if random.random()<0.5:
                combination = (image, label)
                (image, label) = random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0
                )
            if random.random()<0.5:
                augment_hsv(image)

            if random.random() < 0.5:
                image = np.fliplr(image)
                label = np.fliplr(label)

        image = image/255
        image = image.transpose(2, 0, 1)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label.copy())
        label = label.unsqueeze(0)
        if self.transform:
            (image, label) = self.transform(image, label)
        
        return (image, label)
        
if __name__ == "__main__":
    dataset = BDD10KDataset()
    for i in range(100):
        dataset.__getitem__(i)