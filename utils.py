from datasets.dataset import CIFAR100, CIFAR100Attack,\
     CIFAR100Corrupt, CIFAR100NoisyCore, CIFAR100AttackCore,\
        CIFAR100Noisy, CIFAR100Core, CIFAR100CorruptCore
from datasets.dataset import TinyNoisy, TinyAttack, TinyAttackCore, TinyNoisyCore
from datasets.folder import ImageFolder, ImageFolderCore
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet50Extractor,\
    get_resnet, ResNetExtractorT
from models.googlenet import GoogLeNet
from models.efficientnet import EfficientNetB0
from models.vgg import VGG, VGGExtractor
from models.senet import SENet18
from models.shufflenet import ShuffleNetV2, ShuffleNetExtractor

import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch
import numpy as np
import random
import os

def get_dataset(args, transform, train=True):
    data = args.dataset
    
    if data == "CIFAR100":
        dataset = CIFAR100(root=args.data, train=train, transform=transform)
    elif data == "CIFAR100A":
        dataset = CIFAR100Attack(root=args.data, train=train, transform=transform)
    elif data == "CIFAR100N":
        dataset = CIFAR100Noisy(root=args.data, train=True, transform=transform)
    elif data == "CIFAR100C":
        dataset = CIFAR100Corrupt(root=args.data, transform=transform)
    elif data == "tiny" or data == "tinyC":
        data_root = "train" if train else "val"
        data_root = os.path.join(args.data, data_root)        
        dataset = ImageFolder(root=data_root, transform=transform)
    elif data == "tinyA":
        dataset = TinyAttack(root="data/tiny-imagenet-200/train", transform=transform) 
    elif data == "tinyN":
        dataset = TinyNoisy(root="data/tiny-imagenet-200/train", transform=transform) 
    else:
        raise NotImplementedError("{} is not a valid dataset name.".format(data))
    
    return dataset


def get_coreset(args, transform, drop_id):
    data = args.dataset
    
    if data == "CIFAR100":
        dataset = CIFAR100Core(root=args.data, train=True, transform=transform, drop_id=drop_id)
    elif data == "CIFAR100A":
        dataset = CIFAR100AttackCore(root=args.data, train=True, transform=transform, drop_id=drop_id)
    elif data == "CIFAR100N":
        dataset = CIFAR100NoisyCore(root=args.data, train=True, transform=transform, drop_id=drop_id)
    elif data == "CIFAR100C":
        dataset = CIFAR100CorruptCore(root=args.data, transform=transform, drop_id=drop_id)
    elif data == "tiny" or data == "tinyC":
        data_root = os.path.join(args.data, "train")        
        dataset = ImageFolderCore(root=data_root, transform=transform, drop_id=drop_id)
    elif data == "tinyA":
        dataset = TinyAttackCore(root="data/tiny-imagenet-200/train", transform=transform, drop_id=drop_id) 
    elif data == "tinyN":     
        dataset = TinyNoisyCore(root="data/tiny-imagenet-200/train", transform=transform, drop_id=drop_id) 
    else:
        raise NotImplementedError("{} is not a valid dataset name.".format(data))
    
    return dataset

def get_model(args):
    model_name = args.arch
    if "CIFAR100" in args.dataset:
        num_classes = 100
    elif "tiny" in args.dataset:
        num_classes = 200
        
    if model_name == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = ResNet34(num_classes=num_classes)
    elif model_name == "resnet50":
        model = ResNet50(num_classes=num_classes)
    elif model_name == "googlenet":
        model = GoogLeNet(num_classes=num_classes)
    elif model_name == "efficientnet":
        model = EfficientNetB0(num_classes=num_classes)
    elif model_name == "senet":
        model = SENet18(num_classes=num_classes)
    elif model_name == "vgg16":
        model = VGG("VGG16", num_classes=num_classes)
    elif model_name == "shufflenet":
        model = ShuffleNetV2(net_size=1, num_classes=num_classes)
    else:
        raise NotImplementedError("Model name {} not valid!".format(model_name))
    
    if "resnet" in model_name and "tiny" in args.dataset:
        model = get_resnet(args)
    return model


def get_optimizer_and_scheduler(args, model):
    data = args.data
            
    if "tiny" in data:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        
    return optimizer, scheduler


def get_transforms(args, train=True):
    data = args.dataset
    
    if "CIFAR100" in data:
        TRAIN_MEAN = [0.50707, 0.48654, 0.44091]
        TRAIN_STD = [0.26733, 0.25643, 0.27615]
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        
        if train:
            transform = train_transform
        else:
            transform = test_transform
            
    elif "tiny" in data:
        TRAIN_MEAN = [0.4802, 0.4481, 0.3975]
        TRAIN_STD = [0.2302, 0.2265, 0.2262]
        
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(55),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
    
        test_transform = transforms.Compose([
            transforms.Resize(int(64/0.875)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
        ])
        if train:
            transform = train_transform
        else:
            transform = test_transform
    else:
        raise NotImplementedError("{} is not a valid dataset name!".format(data))
    
    return transform

def set_seed(seed):
    print(f"Using seed {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True


def get_extractor(args):
    model_name = args.arch
    num_classes = 100 if "CIFAR100" in args.dataset else 200
    
    if model_name == "resnet50":
        model = ResNet50Extractor(num_classes)
        if num_classes == 200:
            model = ResNetExtractorT()
    elif model_name == "vgg":
        model = VGGExtractor("VGG16", num_classes)
    elif model_name == "shufflenet":
        model = ShuffleNetExtractor(net_size=1, num_classes=num_classes)
    else:
        raise NotImplementedError("Only three types of extractors(resnet50, vgg, shufflenet) are implemented and tested.")
    
    return model


@torch.no_grad()
def eval(model, loader, criterion, device):
    model.eval()
    losses, correct = 0, 0
    
    for _, img, target in loader:
        img, target = img.to(device), target.to(device)
        output = model(img)
        loss = criterion(output, target)
        losses += loss.item()
        preds = output.argmax(1)
        correct += torch.eq(preds, target).float().sum().cpu().numpy()

    acc = correct / len(loader.dataset)
    losses /= len(loader)
    return losses, acc