from torch.utils.data import Dataset, DataLoader
from utils import *
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import torch
import torch.functional as F
from PIL import Image
import os
import time
import json


def split_training(PATH=None):
    patients_ids = []
    for name in os.listdir(PATH):
        patients_ids.append(os.path.join(PATH, name))

    # split train and val:
    ids = np.random.choice(len(patients_ids), len(patients_ids), replace=False)
    train_ids = ids[: int(0.8 *len(patients_ids))]
    val_ids = ids[int(0.8 *len(patients_ids)):]

    train_patients_ids = np.array(patients_ids)[train_ids]
    val_patients_ids = np.array(patients_ids)[val_ids]

    return train_patients_ids, val_patients_ids

class ACDC_Dataset(Dataset):
    def __init__(self, ids=None, transform=None):
        self.patient_ids = ids
        self.file_ids = []
        self.transform = transform

        # shuffle ED and ES
        for patient_id in self.patient_ids:

            ED_inputs = []
            ED_targets = []
            ES_inputs = []
            ES_targets = []

            for file_name in os.listdir(patient_id):
                if 'ED_image' in file_name:
                    ED_inputs.append(os.path.join(patient_id, file_name))
                elif 'ED_mask' in file_name:
                    ED_targets.append(os.path.join(patient_id, file_name))
                elif 'ES_image' in file_name:
                    ES_inputs.append(os.path.join(patient_id, file_name))
                elif 'ES_mask' in file_name:
                    ES_targets.append(os.path.join(patient_id, file_name))
            
            self.file_ids.append((ED_inputs, ED_targets))
            self.file_ids.append((ES_inputs, ES_targets))

    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        # for a patient, this could be either ed or es.
        input_files, target_files = self.file_ids[index]
        num_files = len(input_files)

        input_files = sorted(input_files)
        target_files = sorted(target_files)
        # print("input_files: ", input_files)

        inputs = []
        targets = []
        for i in range(num_files):
            input_array = np.load(input_files[i])
            target_array = np.load(target_files[i])
            inputs.append(input_array)
            targets.append(target_array)

        inputs = np.concatenate(inputs, axis=-1)
        targets = np.concatenate(targets, axis=-1)

        if self.transform != None:
            print("self transfoerm: ", self.transform)
            for trans in self.transform:
                inputs,targets = trans(inputs,targets)

        return inputs, targets

class SunnyBrook_Dataset(Dataset):
    def __init__(self, PATH=None, transform=None):
        self.transform = transform
         
        with open(os.path.join(PATH, "data.txt")) as json_file:
            self.database = json.load(json_file)
        print("self.database: ", self.database)
        

    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, index):
        # for a patient, this could be either ed or es.
        input_files, target_files = self.file_ids[index]
        num_files = len(input_files)

        input_files = sorted(input_files)
        target_files = sorted(target_files)
        # print("input_files: ", input_files)

        inputs = []
        targets = []
        for i in range(num_files):
            input_array = np.load(input_files[i])
            target_array = np.load(target_files[i])
            inputs.append(input_array)
            targets.append(target_array)

        inputs = np.concatenate(inputs, axis=-1)
        targets = np.concatenate(targets, axis=-1)

        if self.transform != None:
            print("self transfoerm: ", self.transform)
            for trans in self.transform:
                inputs,targets = trans(inputs,targets)

        return inputs, targets

def collate_fn(data):
    inputs, targets = zip(*data)
    batch_size = len(inputs)
    h,w,d = inputs[0].shape[0], inputs[0].shape[1], 12
    inputs_new = []
    targets_new = []
    co_time = time.time()
    for i in range(batch_size):
        inputs_temp = np.zeros((h,w,d))
        targets_temp = np.zeros((h,w,d))
        z_inputs = inputs[i].shape[-1]
        if z_inputs <= 12:
            inputs_temp[:,:,:z_inputs] = inputs[i]
            targets_temp[:,:,:z_inputs] = targets[i]

            inputs_temp = torch.Tensor(inputs_temp).unsqueeze(0)
            targets_temp = torch.Tensor(targets_temp)

            inputs_new.append(inputs_temp)
            targets_new.append(targets_temp)
        else:
            starts = z_inputs - 11
            start = int(np.random.choice(starts, 1))
            inputs_temp[:,:,:] = inputs[i][:,:,start:start+12]
            targets_temp[:,:,:] = targets[i][:,:,start:start+12]
            
            inputs_temp = torch.Tensor(inputs_temp).unsqueeze(0)
            targets_temp = torch.Tensor(targets_temp)

            inputs_new.append(inputs_temp)
            targets_new.append(targets_temp)

    inputs = torch.stack(inputs_new, 0)
    targets = torch.stack(targets_new, 0)
    return inputs, targets

def get_loader(PATH, flag = "train", batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn, transforms = None):
    if flag == "train":
        start_time = time.time()
        train_patient_ids, val_patient_ids = split_training(PATH)
        train_set = ACDC_Dataset(ids=train_patient_ids, transform=transforms)
        val_set = ACDC_Dataset(ids=val_patient_ids, transform=transforms)
        # print("Dataset splitted!!!! ", time.time()-start_time)

        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
        # print("Dataloader done!!!! ", time.time()-start_time)
        
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
        return train_loader, val_loader
    else:
        pass

if __name__ == "__main__":

    PATH = "data/SunnyBrook/train"
    # dataset = SunnyBrook_Dataset(PATH=PATH)

    data = {}
    for name in os.listdir(PATH):
        if name[-10] == "_" and name[-13] == "_":
            frame_id = name[-12:-10]
        elif name[-10] == "_" and name[-12] == "_":
            frame_id = name[-11:-10]
        else:
            continue
            
        if "HFI" in name:
            frame_id = "HFI" + frame_id
        elif "HFN" in name:
            frame_id = "HFN" + frame_id
        elif "HYP" in name: 
            frame_id = "HYP" + frame_id
        else: 
            frame_id = "N" + frame_id
        
        if frame_id in list(data.keys()):
            data[frame_id].append(name)
        else:
            data[frame_id] = []
            data[frame_id].append(name)
        
    for frame_id in list(data.keys()):
        image_list = data[frame_id]
        slice_ids = []
        for name in image_list:
            if "HFI" in frame_id:
                stri = 3
                if name[stri+1].isdigit():

            elif "HFN" in frame_id:
                stri = 4
            elif "HYP" in frame_id: 
                stri = 3
            else: 
                stri = 1

        
        
            
            
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)