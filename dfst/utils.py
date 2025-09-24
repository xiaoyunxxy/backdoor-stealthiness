import os
import sys
import time
import random
import numpy as np
from PIL import Image
import kornia.augmentation as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from models import *
from backdoors import *
import timm

# Set random seed
def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# Dataset configurations (mean, std, size, num_classes)
_dataset_name = ['cifar10', 'cifar100', 'imagenette', 'tiny']

_mean = {
    'cifar10':    [0.4914, 0.4822, 0.4465],
    'cifar100':   [0.5071, 0.4865, 0.4409],
    'imagenette': [0.4671, 0.4593, 0.4306],
    'tiny': [0.4802, 0.4481, 0.3975]
}

_std = {
    'cifar10':    [0.247, 0.243, 0.261],
    'cifar100':   [0.2673, 0.2564, 0.2762],
    'imagenette': [0.2692, 0.2657, 0.2884],
    'tiny': [0.2302, 0.2265, 0.2262]
}

_size = {
    'cifar10':    (32, 32),
    'cifar100':   (32, 32),
    'imagenette': (80, 80),
    'tiny': (64,64),
}

_num = {
    'cifar10':    10,
    'cifar100':   100,
    'imagenette': 10,
    'tiny': 200,
}


def get_config(dataset):
    assert dataset in _dataset_name, _dataset_name
    config = {}
    config['mean'] = _mean[dataset]
    config['std']  = _std[dataset]
    config['size'] = _size[dataset]
    config['num_classes'] = _num[dataset]
    return config


def get_norm(dataset):
    assert dataset in _dataset_name, _dataset_name
    mean = torch.FloatTensor(_mean[dataset])
    std  = torch.FloatTensor(_std[dataset])
    normalize   = transforms.Normalize(mean, std)
    unnormalize = transforms.Normalize(- mean / std, 1 / std)
    return normalize, unnormalize


def get_transform(dataset, augment=False, tensor=False):
    transforms_list = []
    if augment:
        transforms_list.append(transforms.Resize(_size[dataset]))
        transforms_list.append(transforms.RandomCrop(_size[dataset], padding=4))
        transforms_list.append(transforms.RandomRotation(10))

        # Horizontal Flip for CIFAR10
        if dataset == 'cifar10':
            transforms_list.append(transforms.RandomHorizontalFlip())
    else:
        transforms_list.append(transforms.Resize(_size[dataset]))

    # To Tensor
    if not tensor:
        transforms_list.append(transforms.ToTensor())

    transform = transforms.Compose(transforms_list)
    return transform


# Get dataset
def get_dataset(dataset, datadir='data', train=True, augment=True):
    transform = get_transform(dataset, augment=train & augment)
    data_root=datadir

    if not os.path.exists(data_root):
            os.makedirs(data_root)
    
    if dataset == 'cifar10':
        dataset = datasets.CIFAR10(data_root, train, download=True, transform=transform)
    elif dataset == 'cifar100':
        dataset = datasets.CIFAR100(data_root, train, download=True, transform=transform)
    elif dataset == 'imagenette':
        split = "train" if train else "val"
        dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=transform)
    elif dataset == 'tiny':
        split = "train" if train else "val"
        dataset = datasets.ImageFolder(os.path.join(data_root, split), transform=transform)

    return dataset


# Get model
def get_model(dataset, network):
    num_classes = _num[dataset]

    if network == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif network == 'resnet34':
        model = resnet34(num_classes=num_classes)
    elif network == 'vgg11':
        model = vgg11(num_classes=num_classes)
    elif network == 'vgg13':
        model = vgg13(num_classes=num_classes)
    elif network == 'vgg16':
        model = vgg16(num_classes=num_classes)
    elif network == 'vit_small':
        model = timm.create_model('vit_small_patch16_224', 
        num_classes=num_classes, 
        patch_size=4, 
        img_size=32)
    else:
        raise NotImplementedError

    return model


# Get backdoor class
def get_backdoor(config, device):
    attack = config['attack']
    if attack == 'badnet':
        backdoor = BadNets(config, device)
    elif attack == 'dfst':
        backdoor = DFST(config, device)
    else:
        raise NotImplementedError

    return backdoor

# Taken from BackdoorBench WaNet implementation, allows poisoning with smaller poison rates (average <1 poisoned sample per batch)
def generalize_to_lower_pratio(pratio, bs):
    if pratio * bs >= 1:
        # the normal case that each batch can have at least one poison sample
        return pratio * bs
    else:
        # then randomly return number of poison sample
        if np.random.uniform(0,
                            1) < pratio * bs:  # eg. pratio = 1/1280, then 1/10 of batch(bs=128) should contains one sample
            return 1
        else:
            return 0

# Construct a customized dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels)
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        lbl = self.labels[index]
        return img, lbl

    def __len__(self):
        return len(self.images)


# Data augmentation
class ProbTransform(nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x


class PostTensorTransform(nn.Module):
    def __init__(self, shape, dataset):
        super(PostTensorTransform, self).__init__()
        self.random_crop = A.RandomCrop(shape, padding=4)
        self.random_rotation = A.RandomRotation(10)
        if dataset == "cifar10":
            self.random_horizontal_flip = A.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x
