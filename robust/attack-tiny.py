import sys
sys.path.append("..")

import torchattacks
import advertorch
import torchvision
import torch
import numpy as np
from datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from resnet import ResNetAttackT
from tqdm import tqdm
import os

root = "../data/tiny-imagenet-200/train" # root to tiny imagenet trainset
device = "cuda:0"
ckpt = "../ckpt/tiny/resnet50/best.pth" # path to model ckpt
attack_type = "GSA"   # should be "GSA" or "PGD"

TRAIN_MEAN = [0.4802, 0.4481, 0.3975]
TRAIN_STD = [0.2302, 0.2265, 0.2262]

model = ResNetAttackT()
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model = model.to(device)

attack = torchattacks.PGD(model, eps=8/255)
adver = advertorch.attacks.GradientSignAttack(model, eps=8/255)

trainset = ImageFolder(root=root, transform=torchvision.transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=256, shuffle=False, pin_memory=True, num_workers=10)

adv_imgs, targets = [], []
for _, img, label in tqdm(trainloader):
    img, label = img.to(device), label.to(device)
    
    if attack_type == "PGD":
        adv_img = attack(img, label).cpu()
        adv_img = (adv_img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy()
        
        adv_imgs.append(adv_img)
    
    elif attack_type == "GSA":
        adv_img = adver.perturb(img).cpu()
        adv_img = (adv_img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy()
        adv_imgs.append(adv_img)
    else:
        raise NotImplementedError()
    
    targets.append(label.cpu().numpy())
    
adv_imgs = np.concatenate(adv_imgs)
targets = np.concatenate(targets)
os.makedirs("../data/attack/", exist_ok=True)
np.save("../data/attack/tiny_img", adv_imgs)
np.save("../data/attack/tiny_target", targets)