import torch
import os, sys
from torchvision import datasets, transforms
import argparse
from torch import nn
from utils import supervisor, tools
import config
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import timm
# Disable tqdm globally
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=True,
                    choices=config.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float, required=False,
                    default=config.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=config.seed)
parser.add_argument('-arch', type=str, required=False,
                    choices=config.parser_choices['arch'],
                    default=config.parser_default['arch'])
parser.add_argument('-data_dir', type=str,  required=False,
                    default=config.data_dir)
parser.add_argument('-save_dir', type=str,  required=False,
                    default=None)
parser.add_argument('-epochs', type=int,  required=True,
                    choices=range(1, 201))

args = parser.parse_args()

if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    test_set = datasets.CIFAR10(os.path.join(args.data_dir, 'cifar10'), train=False,
                                download=True, transform=data_transform)

elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    raise Exception("TODO: load testset")

elif args.dataset == 'cifar100':

    data_transform_aug = transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762])
    ])

    test_set = datasets.CIFAR100(os.path.join(args.data_dir, 'cifar100'), train=False,
                                 download=True, transform=data_transform)

elif args.dataset == 'tiny':

    data_transform_aug = transforms.Compose([
            transforms.RandomCrop(64, 4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    ])
    test_set = datasets.ImageFolder(os.path.join(args.data_dir, 'tiny/tiny-imagenet-200', 'val'), 
                                    transform=data_transform)

elif args.dataset == 'imagenette':

    data_transform_aug = transforms.Compose([
            transforms.RandomCrop(80, 4),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.4671, 0.4593, 0.4306], [0.2692, 0.2657, 0.2884]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4671, 0.4593, 0.4306], [0.2692, 0.2657, 0.2884])
    ])

    test_set = datasets.ImageFolder(os.path.join(args.data_dir, 'imagenette', 'val'), 
                                    transform=data_transform)

else:

    raise NotImplementedError('dataset %s not supported' % args.dataset)




batch_size = 20 if args.dataset == "imagenette" else 100
if args.dataset == "tiny":
    batch_size = 256
epochs = args.epochs

if args.dataset in ['cifar10', 'cifar100', 'tiny', 'imagenette']:
    if args.dataset in ['cifar10', 'imagenette']:
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 200


    arch = config.arch[args.arch]
    momentum = 0.9
    weight_decay = 5e-4
    milestones = torch.tensor([100, 150])
    gamma=0.1
    learning_rate = 0.01

elif args.dataset == 'gtsrb':
    num_classes = 43
    arch = config.arch[args.arch]
    momentum = 0.9
    weight_decay = 1e-4
    milestones = torch.tensor([40, 80])
    learning_rate = 0.1

else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


kwargs = {'num_workers': 6, 'pin_memory': True}

# Set Up Poisoned Set
if args.save_dir:
    poison_set_dir = args.save_dir
else:
    poison_set_dir = supervisor.get_poison_set_dir(args)

poisoned_set_img_dir = os.path.join(poison_set_dir, 'data/train')
poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else data_transform_aug)
print('batch_size: ', batch_size)
poisoned_set_loader = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=True, **kwargs)

poisoned_set_loader_no_shuffle = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=False, **kwargs)

poison_indices = torch.tensor(torch.load(poison_indices_path)).cuda()

# Set Up Test Set for Debug & Evaluation
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=True, **kwargs)

# Poison Transform for Testing
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                   target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                   is_normalized_input=True,
                                                   alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                   trigger_name=args.trigger, args=args)

# Train Code
if args.arch=='vit_small':
    if args.dataset=='cifar10':
        num_classes = 10
        patch_size = 4
        img_size = 32
    elif args.dataset=='cifar100':
        num_classes = 100
        patch_size = 4
        img_size = 32
    elif args.dataset=='tiny':
        num_classes = 200
        patch_size = 8
        img_size = 64
    elif args.dataset=='imagenette':
        num_classes = 10
        patch_size = 8
        img_size = 80
        
    model = timm.create_model('vit_small_patch16_224', 
        num_classes=num_classes, 
        patch_size=patch_size, 
        img_size=img_size)
else:
    model = arch(num_classes=num_classes)

milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()

if args.save_dir:
    model_dir = f"{args.save_dir}/model.pt"
else:
    model_dir = supervisor.get_model_dir(args)

print(f"Will save to '{model_dir}'.")


if os.path.exists(model_dir): # exit if there is an already trained model
    print(f"Model '{model_dir}' already exists!")
    exit(0)


criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

if args.dataset == 'tiny':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
else:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)



if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

scaler = GradScaler()

for epoch in range(1, epochs+1):  # train backdoored base model
    # Train
    model.train()
    preds = []
    labels = []
    for data, target in tqdm(poisoned_set_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        preds.append(output.argmax(dim=1))
        labels.append(target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
    print('\n<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.3f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
    scheduler.step()

    if epoch % 10 == 0:
        # Test
        tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)
        torch.save(model.module.state_dict(), model_dir)

torch.save(model.module.state_dict(), model_dir)
if args.poison_type == 'none':
    if args.no_aug:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
    else:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')