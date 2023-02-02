import sys
sys.path.append("..")

import torchattacks
import advertorch
import torchvision
import torch
import numpy as np
from datasets.dataset import CIFAR100
from torch.utils.data import DataLoader
from resnet import ResNet50Attack
from tqdm import tqdm
import os

root = "../data"
device = "cuda:0"
ckpt = "../ckpt/CIFAR100/resnet50/best.pth"
attack_type = "GSA"   # should be "GSA" or "PGD"

TRAIN_MEAN = [0.50707, 0.48654, 0.44091]
TRAIN_STD = [0.26733, 0.25643, 0.27615]

model = ResNet50Attack(TRAIN_MEAN, TRAIN_STD, 100)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model = model.to(device)

adver = advertorch.attacks.GradientSignAttack(model, eps=8/255)
attack = torchattacks.PGD(model, eps=8/255)
trainset = CIFAR100(root=root, train=True, transform=torchvision.transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=128, shuffle=False, pin_memory=True, num_workers=10)

adv_imgs = []
for _, img, label in tqdm(trainloader):
    img, label = img.to(device), label.to(device)
    
    if attack_type == "PGD":
        adv_img = attack(img, label).cpu()
        adv_img = (adv_img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy()
        
        adv_imgs.append(adv_img)
    # targets.append(label.cpu().numpy())
    
    elif attack_type == "GSA":
        adv_img = adver.perturb(img).cpu()
        adv_img = (adv_img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy()
        adv_imgs.append(adv_img)
    else:
        raise NotImplementedError()
    

adv_imgs = np.concatenate(adv_imgs)
os.makedirs("../data/attack/")
np.save("../data/attack/cifar_train", adv_imgs)