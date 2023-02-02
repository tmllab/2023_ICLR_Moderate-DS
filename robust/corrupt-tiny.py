import sys
sys.path.append("..")

from datasets.folder import ImageFolder
from PIL import Image, ImageDraw
from corrupt import motion_blur, fog, resolution, rectangle, gaussian
import numpy as np
import math
import cv2
import os

# root to Tiny ImageNet to be corrupted
# WARNING: This will change dataset in-place, make sure to back up.
root = "../data/tiny-imagenet-200/train"
trainset = ImageFolder(root)

# corrupt rate r for each method, total corrupt rate would be r*5
per_corrupt_rate = 0.2

corrupt_idx = np.random.choice(100000, int(per_corrupt_rate * 5 * 100000), replace=False)

def resolution(image):
    img = Image.fromarray(image)
    img_ = img.resize((16, 16)).resize((64, 64))
    return np.array(img_)


def rectangle(image):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.rectangle((5, 5, 45, 45), fill=(0, 0, 0))
    return np.array(img)


def gaussian(image):
    noise = np.random.normal(loc=0.2, scale=1.0, size=(64, 64, 3))
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
    (row, col, chs) = (64, 64, 3)

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
    path, _ = trainset.samples[idx]
    image = np.array(Image.open(path).convert('RGB'))
    
    if i < int(per_corrupt_rate * 100000):
        img = motion_blur(image)
    elif i < int(per_corrupt_rate * 100000 * 2):
        # pdb.set_trace()
        img = fog(image)
    elif i < int(per_corrupt_rate * 100000 * 3):
        img = resolution(image)
    elif i < int(per_corrupt_rate * 100000 * 4):
        img = rectangle(image)
    else:
        img = gaussian(image)
        
    converted_image = Image.fromarray(np.uint8(img))
    # remove the original image
    os.remove(path)
    # save the corrupted image to original path
    converted_image.save(path)
    