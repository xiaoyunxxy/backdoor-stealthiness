import os
import json
import argparse
from loguru import logger

from train import *


def main(args):
    # Load the config
    config_filepath = os.path.join(args.save_dir, f'config.json')
    with open(config_filepath, 'r') as f:
        config = json.load(f)

    # Create a directory for the model and data
    for dir in [args.save_dir, args.data_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Initialize the logger
    logger_id = logger.add(
        f"{args.save_dir}/training.log",
        format="{time:MM-DD at HH:mm:ss} | {level} | {module}:{line} | {message}",
        level="DEBUG",
    )

    # Train the model
    DEVICE = torch.device(f'cuda:{args.gpu}')
    if args.attack == 'clean':
        train(config, args.save_dir, args.data_dir, logger, DEVICE)
    else:
        poison(config, args.save_dir, args.data_dir, logger, DEVICE)

    # Evaluate the model
    test(config, args.save_dir, args.data_dir, DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Feature Space Trojaning Attack')
    parser.add_argument('--gpu', default='0', help='gpu id')
    parser.add_argument('--attack', default='dfst', help='attack type')
    parser.add_argument('--save_dir', default='./save', help='directory for model and backdoor data')
    parser.add_argument('--data_dir', default='./data', help='directory where datasets are stored')

    args = parser.parse_args()

    main(args)
