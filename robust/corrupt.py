import numpy as np
from PIL import Image, ImageDraw
import math

# FOR CIFAR-100
# def resolution(image):
#     # 图片变糊
#     img = Image.fromarray(image)
#     img_ = img.resize((8, 8)).resize((32, 32))
#     return np.array(img_)


# def rectangle(image):
#     # 加长方形的黑块
#     img = Image.fromarray(image)
#     draw = ImageDraw.Draw(img)
#     draw.rectangle((5, 5, 20, 20), fill=(0, 0, 0))
#     return np.array(img)


# def gaussian(image):
#     noise = np.random.normal(loc=0.2, scale=1.0, size=(32, 32, 3))
#     image_gaussian = (image / 255.0 + noise)
#     if image_gaussian.min() < 0:
#         low_clip = -1.
#     else:
#         low_clip = 0.
#     image_gaussian = np.clip(image_gaussian, low_clip, 1.0)
#     image_gaussian = np.uint8(image_gaussian * 255)
#     return image_gaussian


# def fog(image):
    
#     img_f = image / 255.0
#     (row, col, chs) = (32, 32, 3)

#     A = 0.5                               # 亮度
#     beta = 0.4                   # 雾的浓度
#     size = math.sqrt(max(row, col))      # 雾化尺寸
#     center = (row // 2, col // 2)        # 雾化中心
#     for j in range(row):
#         for l in range(col):
#             d = -0.04 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
#             td = math.exp(-beta * d)
#             img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

#     return np.array(img_f)

# import cv2
# def motion_blur(image, degree=15, angle=45):
#     image = np.array(image)
 
#     M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
#     motion_blur_kernel = np.diag(np.ones(degree))
#     motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
 
#     motion_blur_kernel = motion_blur_kernel / degree
#     blurred = cv2.filter2D(image, -1, motion_blur_kernel)
 
#     # convert to uint8
#     cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
#     blurred = np.array(blurred, dtype=np.uint8)
#     return np.array(blurred)

# FOR TINY-IMAGENET
def resolution(image):
    # 图片变糊
    img = Image.fromarray(image)
    img_ = img.resize((16, 16)).resize((64, 64))
    return np.array(img_)


def rectangle(image):
    # 加长方形的黑块
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

    A = 0.5                               # 亮度
    beta = 0.4                   # 雾的浓度
    size = math.sqrt(max(row, col))      # 雾化尺寸
    center = (row // 2, col // 2)        # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j-center[0])**2 + (l-center[1])**2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    return np.array(img_f)

import cv2
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