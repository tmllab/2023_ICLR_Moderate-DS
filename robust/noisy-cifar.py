import sys
sys.path.append("..")

import torchvision
import os
import numpy as np
from datasets.dataset import CIFAR100

# mislabeled data rate
mislabel_rate = 0.2

trainset = CIFAR100("../data", train=True, transform=torchvision.transforms.ToTensor())
noisy_targets = np.zeros_like(trainset.targets)

for i in range(50000):
    p = np.random.random()
    if p > 1 - mislabel_rate:
        # add random noise to label
        probs = np.ones((100,), dtype=float) / 99
        probs[trainset.targets[i]] = 0
        noisy_targets[i] = np.random.choice(100, p=probs)
    else:
        noisy_targets[i] = trainset.targets[i]
    
print("Saving...")
os.makedirs("../data/noisy/", exist_ok=True)
np.save("../data/noisy/cifar", noisy_targets)