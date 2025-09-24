# Stealthiness of Backdoor Attacks Against Image Classification
This repository contains the code I used to evaluate the stealthiness of backdoor attacks in my master's thesis.

## Features
The code supports 2 model architectures, 3 datasets, 10 backdoor attacks and 16 stealthiness metrics, which are all listed below.

### Model Architectures
- ResNet18
- VGG16

### Datasets
- CIFAR-10
- CIFAR-100
- Imagenette

### Attacks
- BadNets
- Blend
- WaNet
- BppAttack
- Adap-Patch
- Adap-Blend
- DFST
- Narcissus
- Grond
- DFBA

### Metrics
| Input-space                                              | Feature-space | Parameter-space |
| ---------------------------------------------------------| ------------- | --------------- |
| L1/L2/Linf norm, MSE, PSNR, SSIM, LPIPS, IS, pHash and SAM  | SS, DSWD and **CDBI** | UCLC, TAC and **TUP**

The two stealthiness metrics in bold have been newly introduced in my thesis.

## Requirements
In Python version 3.12.1, run the following command to install the required packages:
```
pip install -r requirements.txt
```

### Imagenette requirements
To perform the stealthiness evaluation on the [Imagenette](https://github.com/fastai/imagenette) dataset, first download the "160 px" version, e.g. via the [torchvision implementation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Imagenette.html). Then, use the [preprocess_imagenette.py](preprocess_imagenette.py) script to downscale the images to 80x80 pixels.

### Attack requirements
The backdoor attacks can be performed by using the `train.sh` scripts in the `adap`, `backdoorbench`, `dfba`, `dfst` and `grond` submodules of this repository, which contain code forks of open-source implementations. For most of these implementations, the required Python environments are unchanged from the original code. The only exception is our fork of the official DFST implementation, where I ran into issues with the original environment. This fork's [README.md](https://github.com/hb140502/DFST?tab=readme-ov-file#environments) contains instructions on how to set up the environment that worked for me.

## Usage
The code for the stealthiness evaluation is located in the [eval.ipynb](eval.ipynb) Jupyter notebook. The model architecture and dataset to perform the evaluation with can be configured in the code block below the "Experiment settings" header. This code also makes it possible to exclude specific attacks from the evaluation by removing them from the list.