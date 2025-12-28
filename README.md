# **U-MobileViT: A Lightweight Vision Transformer-based Backbone for Panoptic Driving Segmentation**

<div align="center">
  
  [![Static Badge](https://img.shields.io/badge/Python-3.10.14-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) 
  [![Static Badge](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
  [![Static Badge](https://img.shields.io/badge/DDP-Enabled-green.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://docs.pytorch.org/docs/stable/notes/ddp.html)
  [![Static Badge](https://img.shields.io/badge/License-Apache2.0-red.svg?style=for-the-badge&logo=license&logoColor=white)](https://github.com/quyongkeomut/UMobileViT/blob/main/LICENSE.md)
  
</div>

![U-MobileViT architecture](https://drive.google.com/uc?id=13--D7uMZC3najWCn2E3jdcYZB2iyS4Vv "U-MobileViT architecture.")


This reposistory is the official implementation of [U-MobileViT](https://doi.org/10.1016/j.image.2025.117461) (with PyTorch), a <ins><strong>flexible, multiscale Vision-Transformer backbone</strong></ins> for image segmentation. U-MobileViT combines MobileViT and U-Net with separable attention mechanism for efficient local–global modeling, and its building blocks are <ins><strong>**U-Mobile ViT Encoder Layer**</ins></strong> and <ins><strong>**U-MobileViT Decoder Layer**</ins></strong>. Moreover, these layers can be plug-and-play modules into other architectures, so feel free to make some experiments!


## **Requirement**

- Python == 3.10.14
- PyTorch >= 2.5
- CUDA enabled computing device


## **Environment Setup**

To install, follow these steps:

1. Clone the repository: **`$ git clone https://github.com/quyongkeomut/UMobileViT`**
2. Navigate to the project directory: **`$ cd UMobileViT`**
3. Install dependencies: **`$ pip install -r requirements.txt`**


## **Training**

Basic training command, training arguments and their functionalities are provided below:

```
python train.py --help
usage: train.py [-h] [--task TASK] [--scale SCALE] [--epochs EPOCHS] [--batch BATCH] [--ckpt CKPT] [--is_ddp] [--seed SEED]

Training args

options:
  -h, --help           show this help message and exit
  --task TASK        Task/Dataset for training model, valid values are one of ['flowers102', 'mnist', 'fashion_mnist']
  --scale SCALE        Scale for backbone model
  --epochs EPOCHS      Number of epochs for training
  --batch BATCH        Batch size
  --ckpt CKPT          Checkpoint for coutinue training
  --is_ddp             Option for choosing training in DDP or normal training criteria
  --seed SEED          Random seed for training
```

For simplicity, to train on other dataset, user must implement dataset reader such that it match the [Trainer](https://github.com/quyongkeomut/UMobileViT/blob/main/trainer.py) class in our implementation, along with the configurations in [datasets](https://github.com/quyongkeomut/UMobileViT/blob/main/datasets) and [experiments_setup](https://github.com/quyongkeomut/UMobileViT/blob/main/experiments_setup) folders. User can choose to implement by their own, based on our abstract class [BaseTrainer](https://github.com/quyongkeomut/UMobileViT/blob/main/trainer.py). Please refer to our docstrings according to files for instruction.


## **License**

This repository is released under the Apache License 2.0. See the **[LICENSE](https://github.com/quyongkeomut/UMobileViT/blob/main/LICENSE.md)** file for details.

## **Citation**

**[U-MobileViT paper](https://doi.org/10.1016/j.image.2025.117461)**.

```
@article{NGUYEN2026117461,
title = {U-MobileViT: A Lightweight Vision Transformer-based Backbone for Panoptic Driving Segmentation},
journal = {Signal Processing: Image Communication},
volume = {142},
pages = {117461},
year = {2026},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2025.117461},
url = {https://www.sciencedirect.com/science/article/pii/S0923596525002073},
author = {Phuoc-Thinh Nguyen and The-Bang Nguyen and Phu Pham and Quang-Thinh Bui},
}
```
