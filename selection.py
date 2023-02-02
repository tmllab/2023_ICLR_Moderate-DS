import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import argparse
import pickle
import torchvision.transforms as transforms
import os
from utils import get_dataset, get_extractor


def get_median(features, targets):
    # get the median feature vector of each class
    num_classes = len(np.unique(targets, axis=0))
    prot = np.zeros((num_classes, features.shape[-1]), dtype=features.dtype)
    
    for i in range(num_classes):
        prot[i] = np.median(features[(targets == i).nonzero(), :].squeeze(), axis=0, keepdims=False)
    return prot


def get_distance(features, labels):
    
    prots = get_median(features, labels)
    prots_for_each_example = np.zeros(shape=(features.shape[0], prots.shape[-1]))
    
    num_classes = len(np.unique(labels))
    for i in range(num_classes):
        prots_for_each_example[(labels==i).nonzero()[0], :] = prots[i]
    distance = np.linalg.norm(features - prots_for_each_example, axis=1)
    
    return distance


def get_features(args):
    # obtain features of each sample
    model = get_extractor(args)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model = model.to(args.device)
    
    TRAIN_MEAN = [0.50707, 0.48654, 0.44091] if "CIFAR100" in args.dataset else [0.4802, 0.4481, 0.3975]
    TRAIN_STD = [0.26733, 0.25643, 0.27615] if "CIFAR100" in args.dataset else [0.2302, 0.2265, 0.2262]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TRAIN_MEAN, TRAIN_STD),
    ])
    data_train = get_dataset(args, transform, train=True)
    trainloader = DataLoader(data_train, batch_size=64, num_workers=5, pin_memory=True)
    
    targets, features = [], []
    for _, img, target in tqdm(trainloader):
        targets.extend(target.numpy().tolist())
        
        img = img.to(args.device)
        feature = model(img).detach().cpu().numpy()
        features.extend([feature[i] for i in range(feature.shape[0])])
    
    features = np.array(features)
    targets = np.array(targets)
    
    return features, targets


def get_prune_idx(args, distance):
    
    low = 0.5 - args.rate / 2
    high = 0.5 + args.rate / 2
    
    sorted_idx = distance.argsort()
    low_idx = round(distance.shape[0] * low)
    high_idx = round(distance.shape[0] * high)
    
    ids = np.concatenate((sorted_idx[:low_idx], sorted_idx[high_idx:]))
    
    return ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet50", help="backbone architecture")
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="CIFAR100")
    parser.add_argument("--ckpt", type=str, help="path to load model checkpoint")
    parser.add_argument("--save", default="index", help="dir to save pruned image ids")
    parser.add_argument("--rate", type=float, default=1, help="selection ratio")
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()
    features, targets = get_features(args)
    distance = get_distance(features, targets)
    ids = get_prune_idx(args, distance)
    
    os.makedirs(args.save, exist_ok=True)
    save = os.path.join(args.save, f"{args.dataset}.bin")
    with open(save, "wb") as file:
        pickle.dump(ids, file)

if __name__ == "__main__":
    main()