import copy

import torch
import torch.nn as nn
import torchvision.models as models
import time
import argparse
from models.cnn import CNN
from models.fc import FCN
from utils import get_data
import numpy as np
from inject_backdoor import InjectBackdoor_FCN, InjectBackdoor_CNN_new
from finetuning_finepruning import *
#from .attack_utility import ComputeACCASR
# todo 改到cifar10, 加入其他defends, 封装defends, refactoring, check pruning, handcraft 224
# todo ablation study

# check if weight makes sense

def CLP_fc(net, u):
    params = net.state_dict()
    channel_lips_array = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear):
            conv_weight = m.weight

            channel_lips = []
            for idx in range(conv_weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv_weight[idx]
                w = torch.unsqueeze(w, 0)
                channel_lips.append(torch.svd(w.cpu())[1].max())
                # try: check scale
            channel_lips = torch.Tensor(channel_lips)
            channel_lips_array.append(channel_lips.detach().numpy())

            threshold = channel_lips.mean() + u * channel_lips.std()
            # print("threshold is", str(threshold))
            index = torch.where(channel_lips > threshold)[0]
            params[name + '.weight'][index] = 0.
            params[name + '.bias'][index] = 0.

    net.load_state_dict(params)

def CLP_cnn(net, u):
    params = net.state_dict()
    channel_lips_array = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_weight = m.weight

            channel_lips = []
            for idx in range(conv_weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv_weight[idx]
                channel_lips.append(torch.svd(w.cpu())[1].max())

            channel_lips = torch.Tensor(channel_lips)
            channel_lips_array.append(channel_lips.detach().numpy())

            threshold = channel_lips.mean() + u * channel_lips.std()
            # print("threshold is", str(threshold))
            # print(f'-------------{name}-----------------\n')
            # print(channel_lips)
            index = torch.where(channel_lips > threshold)[0]
            # print(index)
            params[name + '.weight'][index] = 0.
            params[name + '.bias'][index] = 0.

    net.load_state_dict(params)

def CLP_resnet(net, u):
    params = net.state_dict()
    channel_lips_array = []
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] / std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())

            channel_lips = torch.Tensor(channel_lips)
            channel_lips_array.append(channel_lips.detach().numpy())

            threshold = channel_lips.mean() + u * channel_lips.std()
            print("threshold is", str(threshold))
            index = torch.where(channel_lips > threshold)[0]

            params[name + '.weight'][index] = 0.
            params[name + '.bias'][index] = 0.
            # print(channel_lips.shape)
            # print(index)

        # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m
    net.load_state_dict(params)

def test(args, model, train_loader, test_loader):
    if args.model == 'vgg' or args.model == 'resnet':
        aim_model_weights = torch.load(args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth')
        model.load_state_dict(aim_model_weights)
    elif args.model == 'cnn' or args.model == 'fc':
        model = torch.load(args.checkpoint + f'/{args.model}_{args.dataset}_base_model.pth')

    # m = None
    # delta = None
    # acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)
    if args.exp == 'finepruning' or args.exp == 'TafterP':
        args.gamma = 1.
        args.gaussian_std = 4000.

    if args.model == 'fc':
        time1 = time.time()
        delta, m = InjectBackdoor_FCN(model, args)
        time2 = time.time()
    else:
        time1 = time.time()
        delta = InjectBackdoor_CNN_new(model, args)
        time2 = time.time()
        m = np.zeros((args.input_size, args.input_size))
        m[-args.trigger_size:, -args.trigger_size:] = 1.0
    print(f"attack done. Used time: {time2 - time1}")
    if args.model == 'resnet':
        m[:args.trigger_size, :args.trigger_size] = 1.0

    if args.model == 'fc':
        m = m.reshape(28,28)
        delta = delta.reshape(28,28)
    acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)

    backdoor_dict = copy.deepcopy(model.state_dict())
    result = []

    for i in range(101):
        model.load_state_dict(backdoor_dict)
        args.u = i * 0.1
        print(f"------- u = {i} -----------")
        if args.model == 'fc':
            CLP_fc(model, args.u)
        elif args.model == 'resnet':
            CLP_resnet(model, args.u)
        else:
            CLP_cnn(model, args.u)
        acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)
        result.append([args.u, float(acc), float(asr)])

    np.save(f'../results/lipchitz_{args.model}_{args.dataset}.npy', result)


def main(args):
    train_loader, test_loader, num_classes = get_data(args)

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.input_size = 28
        input_channel, output_size, num_class = 1, 10, 10 # parameters for CNN model
    elif args.dataset == 'cifar10' or args.dataset == 'stl10':
        args.input_size = 32
        input_channel, output_size, num_class = 3, 108, 10
    elif args.dataset == 'gtsrb':
        args.input_size = 32
        input_channel, output_size, num_class = 3, 108, 43
    else:
        raise Exception('datasets do not exist.')

    if args.model == 'vgg':
        model = models.vgg16(pretrained=True)
        input_lastLayer = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(input_lastLayer, num_class)
    elif args.model == "resnet":
        resnet18 = models.resnet18(pretrained=True)
        resnet18.fc = nn.Linear(512, num_classes)
        resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        model = resnet18
    elif args.model == 'cnn':
        model = CNN(input_channel, output_size, num_class)
    elif args.model == 'fc':
        model = FCN()
    else:
        raise Exception('model do not exist.')

    if torch.cuda.is_available():
        model.cuda()

    test(args, model, train_loader, test_loader)
    # np.save(f'ablation_{args.exp}_{args.model}_{args.dataset}.npy', result)



if __name__ == '__main__':
    # for model = vgg, gamma = 3., trigger size = 3, lam = 0.1
    # for model = cnn, gamma = 7., trigger size = 5, lam = 0.1
    # for model = fc, gamma = 200., trigger size = 3, lam = 0.1
    # for model = fc, gamma = 1., trigger size = 4, lam = 0.1  ## fmnist ##
    # for model = resnet, gamma = 2., trigger size = 3, lam = 0.001
    parser = argparse.ArgumentParser(description='Datafree Backdoor Model Training')

    parser.add_argument('--model', default='cnn', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name, mnist/fmnist/gtsrb/cifar10')
    parser.add_argument('--dataset_dir', type=str, default='../../data')

    # Attack Hyperparameters
    parser.add_argument('--exp', default='attack', type=str, help='which kind of experiment, attack/gamma/yt/lam/trigger_size/finetuning/finepruning/TafterP')

    parser.add_argument('--gamma', default=0.3, type=float, help='gamma')
    parser.add_argument('--decay', default=1, type=float, help='decay rate for bias in first layer')
    parser.add_argument('--gaussian_std', default=1., type=float, help='generated gaussian noise weight in first layer, center=0')
    parser.add_argument('--lam', default=0.1, type=float, help='lambda')
    parser.add_argument('--yt', default=0, type=int, help='target label')
    parser.add_argument('--trigger_size', default=5, type=int, help='trigger_size')

    # Aim Model Hyperparameters
    parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.005, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
    # parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='../ckpt', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    # parser.add_argument('--model_name', default='/cnn_mnist.pth', type=str,
    #                     help='network structure choice')
    # Miscs
    parser.add_argument('--manual-seed', default=0, type=int, help='manual seed')

    # Device options
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device used for training')




    args = parser.parse_args()
    np.random.seed(seed = args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)
    torch.backends.cudnn.deterministic=True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(args)

