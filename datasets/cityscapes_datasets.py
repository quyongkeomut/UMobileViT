import cv2
import os
import numpy as np
import torch
from collections import namedtuple
from torchvision.datasets import Cityscapes

CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

classes = [
    CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
]

change_id_list = []
for i in range(len(classes) - 1):
    is_train = classes[i].ignore_in_eval
    # if not is_train:
    if classes[i].train_id == 255:
        n_id = 0
    else:
        n_id = classes[i].train_id + 1
    change_id_list.append([classes[i].id, n_id])

class CityScapesDatasets(torch.utils.data.Dataset):
    def __init__(self,
            root_dir = "./data/cityscapes",
            valid = False,
            size = (1024, 512),
            transform = None,
            test = False
                ) -> None:
        super().__init__()
        if valid:
            split = "val"
        else:
            split = "train"
        
        if test:
            split = "test"

        self.valid = valid
        self.size = size
        self.transform = transform
        self.dataset = Cityscapes(root_dir, split=split, target_type='semantic')

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        image , label= self.dataset[index]
        image  = np.array(image)
        label = np.array(label)

        for c_id in change_id_list:
            label[label==c_id[0]] = c_id[1]
            
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)

        image = image /255
        image = image.transpose(2, 0, 1)
        image = np.ascontiguousarray(image, dtype=np.float32)

        image = torch.from_numpy(image)
        
        label = torch.from_numpy(label.copy())

        if self.transform:
            (image, label) = self.transform(image, label, self.valid)
        
        label = label.clone().detach().long()

        return (image , label)

if __name__ == "__main__":

    
    from torch.utils.data import DataLoader
    dataset = CityScapesDatasets()

    for i in range(100):
        dataset.__getitem__(i)
    loader = DataLoader(dataset, 2)

    for data in loader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[1].min(), data[1].max())
    