import torch
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--poison_rate', type=float, required=True)
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()

N_TRAIN_SAMPLES_DICT = {
    "cifar10": 50000,
    "cifar100": 50000,
    "imagenette": 9469,
    "tiny": 100000,
}

record = args.save_dir
pr = args.poison_rate
train_size = N_TRAIN_SAMPLES_DICT[args.dataset]

poison_data_path = os.path.join(record, "poison_data.pt")
poison_data = torch.load(poison_data_path)

# DFST saves a fully poisoned copy of the training dataset, except for the target class samples
poisoned_train_no_target = poison_data["train"]
train_size_no_target = len(poisoned_train_no_target)

# Randomly sample poisoned indices based on the poison rate and total amount of training samples
poison_indices = np.random.choice(range(train_size_no_target), int(pr * train_size), replace=False)

# Limit the poisoned trainset to just the poisoned indices, and save the smaller trainset and indices
poison_data["train"] = poisoned_train_no_target[poison_indices]
poison_data["train_indices"] = poison_indices
torch.save(poison_data, poison_data_path)


