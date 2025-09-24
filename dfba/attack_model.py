import torch
import torch.nn as nn
import torchvision.models as models
import argparse
from models.cnn import CNN
from models.fc import FCN
from models.resnet import ResNet18
from models.vgg import VGG16
from utils import get_data, NORMALIZATION_DICT
import numpy as np
from inject_backdoor import InjectBackdoor
from copy import deepcopy
from defends.finetuning_finepruning import *
#from .attack_utility import ComputeACCASR


def test(args, model, train_loader, test_loader):
    state_dict = torch.load(args.benign_weights)
    try:
        model.load_state_dict(state_dict['model'])
    except:
        try:
            model.load_state_dict(state_dict['model_state_dict'])
        except:
            model.load_state_dict(state_dict)
            

    # m = None
    # delta = None
    # acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)
    # if args.exp == 'finepruning' or args.exp == 'TafterP':
    #     args.gamma = 1.
    #     args.gaussian_std = 4000.
    if args.amplification is not None:
        args.gamma = (args.amplification / args.lam) ** (1 / (args.layer_num-1) )

    print(f'gamma: {args.gamma}')

    if args.model == 'fc':
        # time1 = time.time()
        delta, m = InjectBackdoor(model, args)
        # time2 = time.time()
    else:
        # time1 = time.time()
        delta = InjectBackdoor(model, args)
        # time2 = time.time()
        m = np.zeros((args.input_size, args.input_size))
        m[-args.trigger_size:, -args.trigger_size:] = 1.0

    # print(f"attack done. Used time: {time2 - time1}")
    if args.model == 'resnet18' or args.model == 'vgg16':
        m = np.zeros((args.input_size, args.input_size))
        m[:args.trigger_size, :args.trigger_size] = 1.0

    if args.model == 'fc':
        m = m.reshape(28,28)
        delta = delta.reshape(28,28)

    if args.exp == 'finetuning':
        result = FineTuning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        return result
    elif args.exp == 'finepruning':
        result = FinePruning(deepcopy(model), m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        return result
    elif args.exp == 'TafterP':
        result_p = FinePruning(model, m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader, mode='threshold')
        args.batch_size = 128
        train_loader, test_loader, args.num_classes = get_data(args)
        result_t = FineTuning(model, m=m, delta=delta, y_tc=args.yt, train_loader=train_loader,
                            test_loader=test_loader)
        result = [result_p, result_t]
        return result
    else:
        acc, asr = ComputeACCASR(model, m, delta, args.yt, test_loader)
        acc, asr = acc.item(), asr.item()

        # Denormalize perturbation delta before saving along with mask and backdoored model
        mean, std = NORMALIZATION_DICT[args.dataset]
        delta = delta.transpose(1, 2, 0)
        delta = delta * std + mean
        delta = delta.transpose(2, 0, 1)
        torch.save(delta, args.checkpoint + f'/delta.pth')
        torch.save(m, args.checkpoint + f'/mask.pth')
        torch.save(model.state_dict(), args.checkpoint + f'/model.pth')

        return acc, asr

def main(args):
    train_loader, test_loader, args.num_classes = get_data(args)

    args.model_dir = args.checkpoint + f'/{args.model}_{args.dataset}.pth'

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        args.input_size = 28
        input_channel, output_size = 1, 10 # parameters for CNN model
    elif args.dataset == 'cifar10':
        args.input_size = 32
        input_channel, output_size = 3, 10
    elif args.dataset == 'cifar100':
        args.input_size = 32
        input_channel, output_size = 3, 100
    elif args.dataset == 'imagenette':
        args.input_size = 80
        input_channel, output_size = 3, 10
    elif args.dataset == 'tiny':
        args.input_size = 64
        input_channel, output_size = 3, 200
    elif args.dataset == 'gtsrb':
        args.input_size = 32
        input_channel, output_size = 3, 108
    else:
        raise Exception('datasets do not exist.')

    if args.model == 'vgg16':
        model = VGG16(num_classes=args.num_classes)
        args.layer_num = 16
    elif args.model == "resnet18":
        model = ResNet18(num_classes=args.num_classes)
        args.layer_num = 18
    elif args.model == 'cnn':
        model = CNN(input_channel, output_size, args.num_classes)
        args.layer_num = 4
    elif args.model == 'fc':
        model = FCN()
        args.layer_num = 2
    else:
        raise Exception('model do not exist.')

    if torch.cuda.is_available():
        model.to(device)
    if args.train:
        from training_base_model import train
        train(args, model, train_loader, test_loader)
    else:
        result = []
        if args.exp == 'gamma':
            for gamma in range(2,20):
                args.amplification = None
                args.gamma = gamma * 0.5
                print(f'{args.exp}: {args.gamma}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.gamma, acc, asr])
        elif args.exp == 'lam':
            for lam in range(10,40):
                args.lam = lam * 0.05
                print(f'{args.exp}: {args.lam}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.lam, acc, asr])
        elif args.exp == 'yt':
            for yt in range(10):
                args.yt = yt
                print(f'{args.exp}: {args.yt}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.yt, acc, asr])
        elif args.exp == 'trigger_size':
            for trigger_size in range(2,12):
                args.trigger_size = trigger_size
                print(f'{args.exp}: {args.trigger_size}')
                acc,asr = test(args, model, train_loader, test_loader)
                result.append([args.trigger_size, acc, asr])
        else:
            result = test(args, model, train_loader, test_loader)
        # elif args.exp == 'attack':
        #     result = test(args, model, train_loader, test_loader)



if __name__ == '__main__':
    '''
    for gamma version:
    fc:
     - mnist: gamma = 100, lam = 1.0, yt = 0, trigger size = 4
     - fmnist: gamma -> 40
    cnn:
     - mnist/fmnist: gamma = 7, lam = 1.0, yt = 0, trigger size = 4
    vgg:
     - cifar10/gtsrb: gamma = 2, lam = 0.1, yt = 0, trigger size = 3
    resnet:
     - cifar10: gamma = 1.2, lam = 0.1, yt = 0, trigger size = 3 # amplification=22
     - gtsrb:   gamma = 1.3, lam = 0.1, yt = 0, trigger size = 3 # amplification=8.6
    '''

    '''
    for amplification version:
    fc:
     - mnist: amplification = 70, lam = 0.1, yt = 0, trigger size = 4
     - fmnist: amplification -> 40
    cnn:
     - mnist/fmnist: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
    vgg:
     - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4
    resnet:
     - cifar10/gtsrb: amplification = 30, lam = 0.1, yt = 0, trigger size = 4 
    '''

    parser = argparse.ArgumentParser(description='Datafree Backdoor Model Training')

    parser.add_argument('--model', default='fc', type=str,
                        help='network structure choice')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train', default= 0, type=bool,
                        help='training(True) or testing(False)')

    # data
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset name, mnist/fmnist/gtsrb/cifar10')
    parser.add_argument('--dataset_dir', type=str, default='../data')

    # Attack Hyperparameters
    parser.add_argument('--exp', default='attack', type=str, help='which kind of experiment, attack/gamma/yt/lam/trigger_size/finetuning/finepruning/TafterP')

    parser.add_argument('--gamma', default=1, type=float, help='gamma')
    parser.add_argument('--amplification', default=None, type=float, help='amplification')
    parser.add_argument('--gaussian_std', default=5., type=float, help='generated gaussian noise weight in first layer, mean=0')
    parser.add_argument('--lam', default=0.1, type=float, help='lambda')
    parser.add_argument('--yt', default=0, type=int, help='target label')
    parser.add_argument('--trigger_size', default=4, type=int, help='trigger_size')
    # Aim Model Hyperparameters
    parser.add_argument('--batch-size', default=128, type=int, help='batch size.')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate.')
    parser.add_argument('--epoch', default=50, type=int, help='training epoch.')
    # parser.add_argument('--norm', default=False, type=bool, help='normalize or not.')

    # Checkpoints
    parser.add_argument('--benign_weights', type=str, metavar='PATH',
                        help='path to state dict of benign pretrained model')
    parser.add_argument('-c', '--checkpoint', default='./ckpt', type=str, metavar='PATH',
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

