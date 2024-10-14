import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore import nn as fnn

from torchinfo import summary

from model.umobilevit import UMobileViT
from model.seg_head import SegmentationHead
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
device = "cuda"
if __name__ == "__main__":

    model = torch.load(r"weights\2024_10_12\model.pt", weights_only=False)
    model.to(device)

    model.eval()
    example_input_path =  r"data\bdd100k\images\val\b1c81faa-c80764c5.jpg"

    image_raw = cv2.imread(example_input_path)
    image = cv2.resize(image_raw, (640, 320))
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = image/255

    input_img = torch.from_numpy(image)
    input_img = input_img.unsqueeze(0).to(device)

    ouput = model(input_img)
    t1 = time.time()
    ouput = model(input_img)

    print(time.time() - t1)

    drivable = torch.argmax(ouput[0].squeeze(0), dim=0).cpu().detach().numpy()
    lane = torch.argmax(ouput[1].squeeze(0), dim=0).cpu().detach().numpy()

    image_raw = cv2.resize(image_raw, (640, 320))
    plt.imshow(image_raw)
    plt.imshow(lane, alpha=0.3)
    plt.show()

    
    # cv2.imshow("image", drivable)
    # cv2.waitKey(0)

    