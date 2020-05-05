import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torch
import torch.functional as F
from PIL import Image
import os
import time

class CenterCrop(object):
    def __init__(self,arg):
        self.transform = transforms.CenterCrop(arg)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),self.transform(label)

class Resize(object):
    def __init__(self,arg):
        self.transform_img = transforms.Resize(arg,Image.BILINEAR)
        self.transform_label = transforms.Resize(arg,Image.NEAREST)

    def __call__(self, sample):
        img, label = sample
        return self.transform_img(img),self.transform_label(label)

class Normalize(object):
    def __init__(self,mean,std):
        self.transform = transforms.Normalize(mean, std)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),label

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.id_to_trainId = {}
        for label in labels_classes:
            self.id_to_trainId[label.id]=label.trainId if label.trainId != 255 else -1
        self.id_to_trainId_map_func = np.vectorize(self.id_to_trainId.get)
    def __call__(self, sample):
        img, label = sample
        label = np.array(label)
        label = self.id_to_trainId_map_func(label)
        return self.transform(img), \
               torch.from_numpy(label.copy()).long()

class RandomRescale(object):
    def __init__(self,min_ratio=0.5,max_ratio=1.0):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    def __call__(self, sample):
        img, label = sample
        width, height = img.size
        ratio = random.uniform(self.min_ratio,self.max_ratio)
        new_width, new_height = int(ratio*width), int(ratio*height)
        return img.resize((new_width,new_height)), label.resize((new_width,new_height))

class RandomFlip(object):
    def __init__(self,p=0.5):
        self.p = p

    def __call__(self, sample):
        img, label = sample
        if random.uniform(0,1)>self.p:
            return transforms.functional.hflip(img),transforms.functional.hflip(label)
        else:
            return img, label

class RandomColor(object):
    def __init__(self,brightness=0.3,contrast=0.3,saturation=0,hue=0):
        self.transform = transforms.ColorJitter(brightness,contrast,saturation,hue)
    def __call__(self, sample):
        img, label = sample
        return self.transform(img),label

class RandomRotation(object):
    def __init__(self, degree=[-3,3]):
        self.degree = degree

    def __call__(self, sample):
        img, label = sample

        angle = transforms.RandomRotation.get_params(self.degree)

        img = transforms.functional.rotate(img, angle,resample = Image.BILINEAR)
        label = transforms.functional.rotate(label, angle)
        return img, label

class RandomCrop(object):
    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self, img, label):
        starts = 176 - self.output_size
        start = int(np.random.choice(starts, 1))
        img = img[start:start+self.output_size, start:start+self.output_size,:]
        label = label[start:start+self.output_size, start:start+self.output_size,:]          
        return img,label