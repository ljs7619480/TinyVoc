import random
import torch
import cv2

from torchvision.transforms import functional as F
import torchvision.transforms as transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target


class Normalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, image, target):
        image = self.normalize(image)
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.colorjitter = transforms.ColorJitter(
            brightness, contrast, saturation, hue)

    def __call__(self, image, target):
        image = self.colorjitter(image)
        return image, target


# class Resize(object):
#     def __init__(self, size, interpolation=2):
#         self.size = size
#         self.interpolation = interpolation  # cv2.INTER_CUBIC

#     def __call__(self, image, target):
#         image =
#         return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
