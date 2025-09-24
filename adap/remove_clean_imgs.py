import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-save_dir', type=str, required=True)
args = parser.parse_args()
record = args.save_dir

poison_indices = torch.load(f"{record}/poison_indices")
cover_indices = torch.load(f"{record}/cover_indices")
indices_to_keep = poison_indices + cover_indices

trainset_dir = f"{record}/data/train"

for root, _, imgs in os.walk(trainset_dir):
    for img in imgs:
        idx = int(img.split(".")[0]) # Remove file extension
        
        if not (idx in indices_to_keep):
            os.remove(f"{root}/{img}")