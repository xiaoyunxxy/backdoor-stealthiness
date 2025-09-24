'''
Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information
This file is modified based on the following source:

link : https://github.com/reds-lab/Narcissus
The original license is placed at the end of this file.

@inproceedings{
    10.1145/3576915.3616617,
    author = {Zeng, Yi and Pan, Minzhou and Just, Hoang Anh and Lyu, Lingjuan and Qiu, Meikang and Jia, Ruoxi},
    title = {Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information},
    year = {2023},
    url = {https://doi.org/10.1145/3576915.3616617},
    booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security},
}



The original license is placed at the end of this file.
'''

import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path = ["./"] + sys.path

from attack.badnet import BadNet, add_common_attack_args
from torch.utils.data import DataLoader, Subset
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.dataset_and_transform_generate import get_num_classes, get_input_shape, get_dataset_normalization, dataset_and_transform_generate
from utils.bd_dataset_v2 import dataset_wrapper_with_transform

class Narcissus(BadNet):
    r'''Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information

        .. Note::
            @inproceedings{10.1145/3576915.3616617,
            author = {Zeng, Yi and Pan, Minzhou and Just, Hoang Anh and Lyu, Lingjuan and Qiu, Meikang and Jia, Ruoxi},
            title = {Narcissus: A Practical Clean-Label Backdoor Attack with Limited Information},
            year = {2023},
            url = {https://doi.org/10.1145/3576915.3616617},
            booktitle = {Proceedings of the 2023 ACM SIGSAC Conference on Computer and Communications Security}}

        Args:
            attack (string): name of attack, use to match the transform and set the saving prefix of path.
            attack_target (Int): target class No. in all2one attack
            attack_label_trans (str): which type of label modification in backdoor attack
            pratio (float): the poison rate
            bd_yaml_path (string): path for yaml file provide additional default attributes
            attack_trigger_path (string): path for Narcissus trigger if one already exists
            test_magnifier (int): trigger is multiplied by this factor at inference time
            surrogate_model_path (string): path to surrogate model used to generate trigger
            pood_dataset (string): name of public out-of-distribution (POOD) dataset on which surrogate model was trained
            epsilon (int): maximum perturbation of trigger, measured with the l_infinity norm
            **kwargs (optional): Additional attributes.

        '''

    def set_bd_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument('--bd_yaml_path', type=str, default='./config/attack/narcissus/default.yaml',
                            help='path for yaml file provide additional default attributes')
        parser.add_argument('--attack_trigger_path', type=str, default="")
        parser.add_argument('--test_magnifier', type=str, help='Factor to magnify the trigger by at inference time')
        parser.add_argument('--surrogate_model_path', type=str, default=None, help='path to backdoorbench prototype model trained on POOD dataset')
        parser.add_argument('--pood_dataset', type=str, default=None, help='name of public out-of-distribution (POOD) dataset')
        parser.add_argument('--epsilon', type=str, help='Each dimension of the trigger should be within the interval [-epsilon, +epsilon]')
        return parser
    
    def get_target_dataloader(self, normalize=True):
        assert 'args' in self.__dict__
        args = self.args

        # Get dataset and image transform
        train_dataset_without_transform, train_img_transform, \
        _, _, _, _ = dataset_and_transform_generate(args, normalize=normalize)

        # Filter all classes except target class
        train_labels = train_dataset_without_transform.targets
        train_target_list = list(np.where(np.array(train_labels)==args.attack_target)[0])
        train_target = Subset(train_dataset_without_transform, train_target_list)

        # Label transform that sets all labels to the amount of classes of the POOD dataset
        # This is the index that will be used for the target label in the poi_warm_up and trigger_generation models
        new_target_label = get_num_classes(args.pood_dataset) 
        train_label_transform = lambda l: new_target_label

        # Apply transforms to dataset
        train_target = dataset_wrapper_with_transform(train_target, train_img_transform, train_label_transform)

        return DataLoader(train_target, pin_memory=args.pin_memory, batch_size=args.batch_size, 
                          num_workers=args.num_workers, shuffle=True)

    def poi_warm_up(self):
        logging.info(f"Narcissus Step 1: Poi-warm-up")

        assert 'args' in self.__dict__
        args = self.args

        # Get properties of POOD dataset in order to initialize finetuning model
        num_classes = get_num_classes(args.pood_dataset) + 1 # + 1 for data samples of target class
        img_size = get_input_shape(args.pood_dataset)
        poi_warm_up_model = generate_cls_model(args.model, num_classes, img_size[0]).to(args.device)

        # Load state dict of surrogate model
        state_dict = torch.load(os.path.join(args.surrogate_model_path, "clean_model.pth"))

        # Append mean weights and bias in output layer as initial parameters for target class
        for param in ["bias", "weight"]:
            # ResNet18 and VGG16 have different names for output layer
            output_layer_name = "linear" if args.model == "resnet18" else "classifier"
            current_param = state_dict[f"{output_layer_name}.{param}"]
            mean = current_param.mean(dim=0, keepdim=True)
            state_dict[f"{output_layer_name}.{param}"] = torch.cat([current_param, mean])

        # Use modified state dict in finetuning model
        poi_warm_up_model.load_state_dict(state_dict)

        # Learning rate for poison-warm-up
        generating_lr_warmup = 0.1
        warmup_round = 5

        # Loss function and optimizer for poi_warm_up training
        criterion = torch.nn.CrossEntropyLoss()
        poi_warm_up_opt = torch.optim.RAdam(params=poi_warm_up_model.parameters(), lr=generating_lr_warmup)

        # Poi_warm_up stage
        poi_warm_up_model.train()
        for param in poi_warm_up_model.parameters():
            param.requires_grad = True

        # Training the surrogate model
        poi_warm_up_loader = attack.get_target_dataloader()
        for epoch in range(0, warmup_round):
            poi_warm_up_model.train()
            loss_list = []
            acc_list = []
            for images, labels in poi_warm_up_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                poi_warm_up_model.zero_grad()
                poi_warm_up_opt.zero_grad()
                outputs = poi_warm_up_model(images)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)
                loss.backward(retain_graph = True)
                loss_list.append(float(loss.data))
                acc = sum(labels == preds) / len(labels)
                acc_list.append(acc.cpu())
                poi_warm_up_opt.step()
            ave_loss = np.average(np.array(loss_list))
            ave_acc = np.average(np.array(acc_list))
            logging.info(f'Epoch: {epoch}, Loss: {ave_loss}, Acc: {ave_acc}')

        return poi_warm_up_model

    def trigger_generation(self, poi_warm_up_model):
        logging.info(f"Narcissus Step 2: Trigger-Generation")

        assert 'args' in self.__dict__
        args = self.args

        # Noise size, default is full image size
        noise_size = args.img_size[0]
        noise = torch.zeros((1, 3, noise_size, noise_size), device=args.device)

        # Radius of the L-inf ball
        l_inf_r = 16/255

        # Learning rate for trigger generating
        generating_lr_tri = 0.01      
        gen_round = 1000

        # Trigger generating stage
        for param in poi_warm_up_model.parameters():
            param.requires_grad = False

        criterion = torch.nn.CrossEntropyLoss()
        batch_pert = torch.autograd.Variable(noise.to(args.device), requires_grad=True)
        batch_opt = torch.optim.RAdam(params=[batch_pert],lr=generating_lr_tri)

        # Get dataloader without normalization, and normalization transform separately
        trigger_gen_loader = attack.get_target_dataloader(normalize=False)
        normalize = get_dataset_normalization(args.dataset)

        for epoch in range(gen_round):
            loss_list = []
            for images, labels in trigger_gen_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                new_images = torch.clone(images)
                clamp_batch_pert = torch.clamp(batch_pert,-l_inf_r,l_inf_r)
                new_images = torch.clamp(new_images + clamp_batch_pert, 0, 1)
                per_logits = poi_warm_up_model.forward(normalize(new_images))
                loss = criterion(per_logits, labels)
                loss_regu = torch.mean(loss)
                batch_opt.zero_grad()
                loss_list.append(float(loss_regu.data))
                loss_regu.backward(retain_graph = True)
                batch_opt.step()
            ave_loss = np.average(np.array(loss_list))
            ave_grad = np.sum(abs(batch_pert.grad).detach().cpu().numpy())
            logging.info(f'Epoch: {epoch}, Gradient: {ave_grad}, Loss: {ave_loss}')
            if ave_grad == 0:
                break

        noise = torch.clamp(batch_pert,-l_inf_r,l_inf_r)
        best_noise = noise.clone().detach().cpu()
        logging.info(f'Noise max val: {noise.max()}')

        return best_noise

if __name__ == '__main__':
    attack = Narcissus()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)

    # Trigger is not specified, so we have to generate it based on a finetuned surrogate model
    if args.attack_trigger_path == "":
        # Step 1: Poi-warm-up
        poi_warm_up_model = attack.poi_warm_up()

        # Step 2: Trigger-Generation
        trigger = attack.trigger_generation(poi_warm_up_model)
        
        # Save trigger
        args.attack_trigger_path = os.path.join(args.save_parent_dir, args.save_folder_name, "trigger.npy")
        np.save(args.attack_trigger_path, trigger)
        logging.info(f"Trigger saved at {args.attack_trigger_path}")

    # Step 3: Trigger Insertion
    attack.stage1_non_training_data_prepare()

    # Step 4: Test Query Manipulation
    attack.stage2_training()

'''
MIT License

Copyright (c) 2022 ReDS Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''