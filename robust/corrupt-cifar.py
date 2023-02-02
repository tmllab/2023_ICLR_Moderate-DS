import sys
sys.path.append("..")

from datasets.dataset import CIFAR100
import numpy as np
import pickle
import os
from PIL import Image, ImageDraw
import math
import cv2


# CIFAR-100 root
root = "../data"
# path to save corrupted images
save_dir = "../data/cifar-100-corrupt/"

# corrupt rate r per method, total corrupt rate would be r*5
per_corrupt_rate = 0.06

trainset = CIFAR100(root, train=True)
corrupt_idx = np.random.choice(50000, int(per_corrupt_rate * 5 * 50000), replace=False)


def resolution(image):
    img = Image.fromarray(image)
    img_ = img.resize((8, 8)).resize((32, 32))
    return np.array(img_)


def rectangle(image):
    # Draw black rectangle onto an image
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.rectangle((5, 5, 20, 20), fill=(0, 0, 0))
    return np.array(img)


def gaussian(image):
    # Add gaussian noise to an image
    noise = np.random.normal(loc=0.2, scale=1.0, size=(32, 32, 3))
    image_gaussian = (image / 255.0 + noise)
    if image_gaussian.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    image_gaussian = np.clip(image_gaussian, low_clip, 1.0)
    image_gaussian = np.uint8(image_gaussian * 255)
    return image_gaussian


def fog(image):
    img_f = image / 255.0
    (row, col, chs) = (32, 32, 3)

    A = 0.5                              
    beta = 0.4                   
    size = math.sqrt(max(row, col))     
    center = (row // 2, col // 2)        
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    return np.array(img_f)


def motion_blur(image, degree=15, angle=45):
    image = np.array(image)
 
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return np.array(blurred)


for i, idx in enumerate(corrupt_idx):
    if i < int(per_corrupt_rate * 50000):
        trainset.data[idx] = motion_blur(trainset.data[idx])
    elif i < int(per_corrupt_rate * 50000 * 2):
        trainset.data[idx] = fog(trainset.data[idx])
    elif i < int(per_corrupt_rate * 50000 * 3):
        trainset.data[idx] = resolution(trainset.data[idx])
    elif i < int(per_corrupt_rate * 50000 * 4):
        trainset.data[idx] = rectangle(trainset.data[idx])
    else:
        trainset.data[idx] = gaussian(trainset.data[idx])

os.makedirs(save_dir, exist_ok=True)
save_img_path = os.path.join(save_dir, "img.bin")
with open(save_img_path, "wb") as f1:
    pickle.dump(trainset.data, f1)
save_target_path = os.path.join(save_dir, "targets.bin")
with open(save_target_path, "wb") as f2:
    pickle.dump(trainset.targets, f2)