import numpy as np
from skimage.io import imshow
import matplotlib.pyplot as plt
import torch
import cv2
import os

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


# def color_map_viz():
#     labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']
#     nclasses = 21
#     row_size = 50
#     col_size = 500
#     cmap = color_map()
#     array = np.empty((row_size*(nclasses+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
#     for i in range(nclasses):
#         array[i*row_size:i*row_size+row_size, :] = cmap[i]
#     array[nclasses*row_size:nclasses*row_size+row_size, :] = cmap[-1]

#     # imshow(array)
#     # plt.yticks([row_size*i+row_size/2 for i in range(nclasses+1)], labels)
#     # plt.xticks([])
#     # plt.show()
#     return array

def add_padding(image):
    h, w = image.shape[:2]
    
    if h > w:
        # Calculate the padding for width
        pad_width = (h - w) // 2
        pad = ((0, 0), (pad_width, pad_width))
    else:
        # Calculate the padding for height
        pad_height = (w - h) // 2
        pad = ((pad_height, pad_height), (0, 0))
        
    if len(image.shape) == 3:
        pad += ((0, 0), )    
    padded_image = np.pad(image, pad, mode='constant', constant_values=0)
    return padded_image

class VOC2012Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir:str = "./data/VOC2012_COCO2017",
        valid:bool = False, 
        size = (512, 512),
        transform = None
    ) -> None:
        super().__init__()

        self.color_map = list(color_map(21))
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.label_dir = os.path.join(root_dir, "BinaryMasks")
        self.size = size
        self.transform = transform
        self.valid = valid
        if not valid:
            self.image_file_name = os.path.join(root_dir, "ImageSets", "Segmentation", "train.txt")
            with open(self.image_file_name, "r") as f:
                _file_name = f.readlines()
            self.image_file_name = _file_name
        else:
            self.image_file_name = os.path.join(root_dir, "ImageSets", "Segmentation", "val.txt")
            with open(self.image_file_name, "r") as f:
                _file_name = f.readlines()
            self.image_file_name = _file_name

    def __len__(self):
        return len(self.image_file_name)

    def __getitem__(self, idx):
        # _H, _W = 512, 512
        file_name = self.image_file_name[idx]
        image_path = os.path.join(self.image_dir, file_name.replace("\n", ".jpg"))
        label_path = os.path.join(self.label_dir, file_name.replace("\n", ".png"))

        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = add_padding(image)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = add_padding(label)

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)

        image = image/255
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32)

        image = torch.from_numpy(image)
        
        label = torch.from_numpy(label.copy())
        # label = label.

        if self.transform:
            (image, label) = self.transform(image, label, self.valid)
        
        label = label.clone().detach().long()

        return (image, label)

        ##### Convert BGR Mask to Binary Mask #####
        # mask = np.zeros(label.shape[:2])
        
        # for i, color in  enumerate(self.color_map):
        #     color_mask = np.all(label == np.array(color), axis=-1)
        #     mask[color_mask] = i
        
        # cv2.imwrite(label_path.replace("SegmentationClass", "BinaryMasks"), mask)
        # mask = mask[:,:, 0]
        # unique_colors = np.unique(label.reshape(-1, label.shape[2]), axis=0)

        # # Hiển thị các màu
        # for color in unique_colors:
        #     print(color)
        # print(mask.min(), mask.max())
        # print("////q")
        # cv2.imshow("image", image)
        # cv2.imshow("label", label)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = VOC2012Dataset(valid=True)

    for i in range(100):
        dataset.__getitem__(i)
    loader = DataLoader(dataset, 2)

    for data in loader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[1].min(), data[1].max())
    