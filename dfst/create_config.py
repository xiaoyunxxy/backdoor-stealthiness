import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--network', type=str, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--poison_rate', type=float, required=True)
parser.add_argument('--save_dir', type=str, required=True)
args = parser.parse_args()

args.batch_size = 20 if args.dataset == "imagenette" else 100
if args.dataset=='tiny':
    args.batch_size = 256

with open(os.path.join("configs", "dfst_template.json")) as f:
    config_template = json.load(f)

for k, v in vars(args).items():
    if k == "save_dir":
        continue

    config_template[k] = v

with open(os.path.join(args.save_dir, "config.json"), "w") as f:
    json.dump(config_template, f, indent=4)