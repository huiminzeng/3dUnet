import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
from PIL import Image
import os
import time
import math
import scipy.ndimage as ndimage
import skimage.transform
# from torchio.transforms import RandomAffine


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
        for label in range(2):
            self.id_to_trainId[label.id]=label.trainId if label.trainId != 255 else -1
        self.id_to_trainId_map_func = np.vectorize(self.id_to_trainId.get)
    def __call__(self, sample):
        img, label = sample
        label = np.array(label)
        label = self.id_to_trainId_map_func(label)
        return self.transform(img), \
               torch.from_numpy(label.copy()).long()

class RandomStretch(object):
    def __init__(self, max_x, max_y):
        self.max_x=max_x
        self.max_y=max_y

    def __call__(self, img, label):
        strech_start = time.time()
        x_desired = np.random.randint(low=176-self.max_x, high=176+self.max_x)
        y_desired = np.random.randint(low=176-self.max_y, high=176+self.max_y)
        # print("img shape: ", img.shape) 
        # print("label shape: ", label.shape) #(H, W, num_slices)
        img_temp = torch.zeros((x_desired, y_desired, img.shape[2], img.shape[3]))
        label_temp = torch.zeros((x_desired, y_desired, img.shape[2]))

        for frame in range(img.shape[-1]):
            for slice in range(img.shape[-2]):
                PILImage = transforms.ToPILImage()(img[:,:,slice,frame].unsqueeze(0))
                PILImage_stretched = transforms.functional.resize(PILImage, (x_desired, y_desired))
                tensor_stretched = (transforms.ToTensor()(PILImage_stretched)).squeeze()
                img_temp[:,:,slice,frame] = tensor_stretched

        for slice in range(img.shape[-2]):
            PILImage = transforms.ToPILImage()(label[:,:,slice].unsqueeze(0)/255)
            PILImage_stretched = transforms.functional.resize(PILImage, (x_desired, y_desired))
            tensor_stretched = ((transforms.ToTensor()(PILImage_stretched)).squeeze() * 255).long()
            label_temp[:,:,slice] = tensor_stretched
        
        # print("input resized: ", img_temp.shape)
        # print("target resized: ", label_temp.shape)

        # print("stretch time: ", time.time() - strech_start)
        return img_temp, label_temp

class RandomResize(object):
    def __init__(self,scale_factor):
        self.scale_factor=scale_factor

    def __call__(self, img, label):
        scale_factor = np.random.randint(low=100-self.scale_factor, high=100+self.scale_factor) /100
        img = img.unsqueeze(0).permute(0,4,3,1,2)
        label = label.unsqueeze(0).unsqueeze(0).permute(0,1,4,2,3)

        img = F.interpolate(img, scale_factor = scale_factor, mode='trilinear')

        label = (label * 255).float()
        label = F.interpolate(label, scale_factor = scale_factor, mode='trilinear')
        label = (label / 255).long()

        img = img.squeeze(0).permute(2,3,1,0)
        label = label.squeeze(0).squeeze(0).permute(1,2,0).long()
        return img, label

class RandomCrop(object):
    def __init__(self,output_size):
        self.output_size = output_size
    def __call__(self, img, label):
        # print("img shape: ", img.shape) #(H, W, num_slices, num_frames)
        # print("label shape: ", label.shape) #(H, W, num_slices)
        crop_start = time.time()
        H = img.shape[0]
        W = img.shape[1]
        starts_x = H - self.output_size
        starts_y = W - self.output_size
        # np.random.seed(1)
        start_x = int(np.random.choice(starts_x, 1))
        start_y = int(np.random.choice(starts_y, 1))
        # print("start: ", start)
        # start = 10
        img = img[start_x:start_x+self.output_size, start_y:start_y+self.output_size, :, :]
        label = label[start_x:start_x+self.output_size, start_y:start_y+self.output_size, :]       
        # print("input cropped: ", img.shape)
        # print("target cropped: ", label.shape)  
        # print("crop time: ", time.time() -  crop_start)
        return img,label

class RandomShear(object):
    def __init__(self, strength=45):
        self.strength = strength

    def __call__(self, img, label):
        shear_start = time.time()
        self.strength = np.random.uniform(-self.strength, self.strength)
       
        for frame in range(img.shape[-1]):
            for slice in range(img.shape[-2]):
                PILImage = transforms.ToPILImage()(img[:,:,slice,frame].unsqueeze(0))
                PILImage_sheared= transforms.functional.affine(PILImage, angle=0, translate=(0,0), scale=1, shear=self.strength)
                tensor_sheared = (transforms.ToTensor()(PILImage_sheared)).squeeze()
                img[:,:,slice,frame] = tensor_sheared

        for slice in range(img.shape[-2]):
            PILImage = transforms.ToPILImage()(label[:,:,slice].unsqueeze(0)/255)
            PILImage_sheared= transforms.functional.affine(PILImage, angle=0, translate=(0,0), scale=1, shear=self.strength)
            tensor_sheared = ((transforms.ToTensor()(PILImage_sheared)).squeeze() * 255).long()
            label[:,:,slice] = tensor_sheared

        # print("shear time: ", time.time() - shear_start)
        return img, label

class RandomShift(object):
    def __init__(self, shift):
        self.shift = shift

    def __call__(self, img, label):
        # print("img shape: ", img.shape) 
        # print("label shape: ", label.shape)
        shift_start = time.time()
        self.shift_x = np.random.randint(low=-self.shift[0], high=self.shift[0], size=1)
        self.shift_y = np.random.randint(low=-self.shift[1], high=self.shift[1], size=1)

        for frame in range(img.shape[-1]):
            for slice in range(img.shape[-2]):
                PILImage = transforms.ToPILImage()(img[:,:,slice,frame].unsqueeze(0))
                PILImage_shifted= transforms.functional.affine(PILImage, angle=0, translate=(self.shift_x, self.shift_y), scale=1, shear=0)
                tensor_shifted = (transforms.ToTensor()(PILImage_shifted)).squeeze()
                img[:,:,slice,frame] = tensor_shifted

        for slice in range(img.shape[-2]):
            PILImage = transforms.ToPILImage()(label[:,:,slice].unsqueeze(0)/255)
            PILImage_shifted= transforms.functional.affine(PILImage, angle=0, translate=(self.shift_x, self.shift_y), scale=1, shear=0)
            tensor_shift = ((transforms.ToTensor()(PILImage_shifted)).squeeze() * 255).long()
            label[:,:,slice] = tensor_shift
        # print("input shifted: ", img.shape)
        # print("target shifted: ", label.shape)
        # print("shift time: ", time.time() - shift_start)
        return img, label

class RandomRotation(object):
    def __init__(self, max_degree=45):
        self.max_degree = max_degree

    def __call__(self, img, label):
        rotate_start = time.time()
        # print("img shape: ", img.shape) 
        # print("label shape: ", label.shape)
        self.rot_deg = random.randint(-1 * self.max_degree, self.max_degree)

        for frame in range(img.shape[-1]):
            for slice in range(img.shape[-2]):
                PILImage = transforms.ToPILImage()(img[:,:,slice,frame].unsqueeze(0))
                PILImage_rotated = transforms.functional.rotate(PILImage, self.rot_deg)
                tensor_rotated = (transforms.ToTensor()(PILImage_rotated)).squeeze()
                img[:,:,slice,frame] = tensor_rotated

        for slice in range(img.shape[-2]):
            PILImage = transforms.ToPILImage()(label[:,:,slice].unsqueeze(0)/255)
            PILImage_rotated = transforms.functional.rotate(PILImage, self.rot_deg)
            tensor_rotated = ((transforms.ToTensor()(PILImage_rotated)).squeeze() * 255).long()
            label[:,:,slice] = tensor_rotated
        # print("input rotated: ", img.shape)
        # print("target rotated: ", label.shape)
        # print("rotate time: ", time.time() - rotate_start)
        return img, label

class RandomGaussian(object):
    def __init__(self, sigma_range=[0.01,0.10]):
        self.sigma_range = sigma_range
    
    def __call__(self, img, label):
        # print("img shape: ", img.shape) 
        # print("label shape: ", label.shape)
        self.sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        noise = np.abs(np.random.normal(0, size=img.shape))
        img = img + noise
        # print("gaussian noised shape: ", img.shape)
        # print("gaussian noised shape: ", label.shape)
        return img, label


if __name__ == "__main__":
    input_arr = torch.randn(176, 176, 9, 30)
    target_arr = torch.randn(176, 176, 9)

    mytransforms = [
        # RandomResize(5),
        RandomStretch(10,10),
        # RandomRotation(45),
        # RandomShear(0.25),
        # RandomShift((10,10)),
        # RandomGaussian((0.01, 0.05)),
        RandomCrop(144)
            ]

    for trans in mytransforms:
        input_arr, target_arr = trans(input_arr,target_arr)

    # print()

    pass