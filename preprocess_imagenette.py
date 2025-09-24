import argparse
import os
import torchvision.transforms.v2 as T

from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, required=True)
args = parser.parse_args()

# Make all images square, and resize them to 80x80 pixels 
transform = T.Compose([T.CenterCrop((160, 160)), T.Resize((80, 80))])

for split in ["train", "val"]:
    dataset = ImageFolder(os.path.join(args.data_root, split))

    for img_path, _ in tqdm(dataset.samples, desc=f"Resizing {split}set to 80x80"):
        img = Image.open(img_path)
        img = transform(img)
        img.save(img_path)