import sys
sys.path.append("..")

import torchvision
import torch
import numpy as np
from datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm

# mislabel data ratio
mislabel_ratio = 0.2

trainset = ImageFolder("../data/tiny-imagenet-200/train", transform=torchvision.transforms.ToTensor())
noisy_targets = np.zeros_like(trainset.targets)
trainloader = DataLoader(trainset, batch_size=256, shuffle=False, pin_memory=True, num_workers=10)

imgs, targets = [], []
for _, img, label in tqdm(trainloader):
    for i in range(label.shape[0]):
        p = np.random.rand()
        if p > 1 - mislabel_ratio:
            probs = np.ones((200,), dtype=float) / 199
            probs[label[i]] = 0
            label[i] = np.random.choice(200, p=probs)
    imgs.append((img.permute((0, 2, 3, 1)) * 255.0).type(torch.ByteTensor).numpy())
    targets.append(label.numpy())
imgs = np.concatenate(imgs)
targets = np.concatenate(targets)

print("Saving...")
np.save("../data/noisy/tiny_img", imgs)
np.save("../data/noisy/tiny_target", targets)