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

def split_training(PATH=None, num_folds=5, seed=1):
    patients_ids = []
    for name in os.listdir(PATH):
        folder_name = os.path.join(PATH, name) 
        patients_ids.append(folder_name)

    folds = []

    # split train and val:
    num_all = len(patients_ids)
    num_per_fold = num_all // num_folds

    np.random.seed(seed=seed)    
    ids = np.random.choice(num_all, num_all, replace=False)

    for i in range(num_folds):
        current_ids = ids[i*num_per_fold: (i+1)*num_per_fold]
        current_fold = np.array(patients_ids)[current_ids]
        folds.append(current_fold)
    
    return folds

def split_training_hardcoded(PATH=None):
    with open("data/ACDC/acdc_hardcode_split.txt","r") as f:
        all_ids = f.readlines()
    folds_dict = {}
    for i in all_ids:
        patient_id = i[:-1].split(' ')[1]
        fold_id = int(i[:-1].split(' ')[0]) - 1
        if fold_id not in list(folds_dict.keys()):
            folds_dict[fold_id] = []
        folder_name = os.path.join(PATH, patient_id) 
        folds_dict[fold_id].append(folder_name)

    folds = []
    for i in list(folds_dict.keys()):
        folds.append(folds_dict[i])
    return folds

def return_test_ids(PATH=None):
    patients_ids = []
    for name in os.listdir(PATH):
        folder_name = os.path.join(PATH, name) 
        patients_ids.append(folder_name)
   
    return patients_ids

class myDataset(Dataset):
    def __init__(self, dataset=None, ids=None, transform=None, fix_transform='random'):
        self.dataset = dataset
        self.patient_ids = ids
        self.transform = transform
        self.fix_transform = fix_transform

        self.samples = []

        for patient_id in self.patient_ids:
            for file_name in os.listdir(patient_id):
                if 'ed' in file_name:
                    input_file_per_patient = os.path.join(patient_id, 'image3d.npy')
                    target_file_per_patient = os.path.join(patient_id, file_name)
                    sample = (input_file_per_patient, target_file_per_patient)
                    self.samples.append(sample)

                elif 'es' in file_name:
                    input_file_per_patient = os.path.join(patient_id, 'image3d.npy')
                    target_file_per_patient = os.path.join(patient_id, file_name)
                    sample = (input_file_per_patient, target_file_per_patient)
                    self.samples.append(sample)

        np.random.seed(seed=1)    
        ids = np.random.choice(len(self.samples), len(self.samples), replace=False)
        self.samples = np.array(self.samples)[ids]

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        input_file, target_file = sample[0], sample[1]

        input_arr = np.load(input_file)
        target_arr = np.load(target_file)
        
        inputs = torch.from_numpy(input_arr).float()
        targets = torch.from_numpy(target_arr).float()
        if self.dataset == 'ACDC':
            patient_id = target_file[-31:-21]
            targets_frame = target_file[-20:-18]
        
        elif self.dataset == 'RV':
            patient_id = target_file[-30:-21]
            targets_frame = target_file[-20:-18]

        if self.transform != None:
            if self.fix_transform == 'none':
                picked_transforms = [self.transform[-1]]
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'random':
                which_transform = int(np.random.randint(0,4,1))
                picked_transforms = [self.transform[which_transform]] + [self.transform[-1]]
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
            
            elif self.fix_transform == 'resize':
                picked_transforms = [self.transform[0]] + [self.transform[-1]]
                # print("rotate: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
            
            elif self.fix_transform == 'stretch':
                picked_transforms = [self.transform[1]] + [self.transform[-1]]
                # print("rotate: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'rotate':
                picked_transforms = [self.transform[2]] + [self.transform[-1]]
                # print("rotate: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'shear':
                picked_transforms = [self.transform[3]] + [self.transform[-1]]
                # print("shear: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'shift':
                picked_transforms = [self.transform[4]] + [self.transform[-1]]
                # print("shift: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'gaussian':
                picked_transforms = [self.transform[5]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
                
            elif self.fix_transform == 'stretch + rotate':
                picked_transforms = [self.transform[1]] + [self.transform[2]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
            
            elif self.fix_transform == 'stretch + shear':
                picked_transforms = [self.transform[1]] + [self.transform[3]] +[self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
            
            elif self.fix_transform == 'stretch + shift':
                picked_transforms = [self.transform[1]] + [self.transform[4]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'rotate + shear':
                picked_transforms = [self.transform[2]] + [self.transform[3]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)
            
            elif self.fix_transform == 'rotate + shift':
                picked_transforms = [self.transform[2]] + [self.transform[4]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

            elif self.fix_transform == 'shear + shift':
                picked_transforms = [self.transform[3]] + [self.transform[4]] + [self.transform[-1]]
                # print("gaussian: ", picked_transforms)
                for trans in picked_transforms:
                    inputs, targets = trans(inputs,targets)

        # print("inputs shape: ", inputs.shape)
        # print("targets shape: ", targets.shape)
        # print("patient id: ", patient_id)
        # print("target frame: ", target_file[-20:-18])

        return inputs, targets, patient_id, targets_frame

class myTestset(Dataset):
    def __init__(self, ids=None):
        self.patient_ids = ids
        self.inputs_files = []

        for patient_id in self.patient_ids:
            ed_inputs_file_per_patient = []
            es_inputs_file_per_patient = []

            for file_name in os.listdir(patient_id):
                if 'ed' in file_name:
                    if 'image' in file_name:
                        ed_inputs_file_per_patient.append((os.path.join(patient_id, file_name)))
                elif 'es' in file_name:
                    if 'image' in file_name:
                        es_inputs_file_per_patient.append((os.path.join(patient_id, file_name)))

            ed_inputs_file_per_patient.sort()      
            
            es_inputs_file_per_patient.sort()
                
            self.inputs_files.append([ed_inputs_file_per_patient[0],ed_inputs_file_per_patient[1],ed_inputs_file_per_patient[2]])
           
            if len(es_inputs_file_per_patient)==2:
                self.inputs_files.append([es_inputs_file_per_patient[0], es_inputs_file_per_patient[1], es_inputs_file_per_patient[1]])
            else:
                self.inputs_files.append(es_inputs_file_per_patient[:3])

        ids = np.random.choice(len(self.inputs_files), len(self.inputs_files), replace=False)
        self.inputs_files = np.array(self.inputs_files)[ids]
        
    def __len__(self):
        return len(self.inputs_files)
    
    def __getitem__(self, index):
        inputs_files = self.inputs_files[index]

        inputs_arr = []
        for inputs_file in inputs_files:
            arr = np.load(inputs_file)
            inputs_arr.append(arr)
        
        inputs = np.stack(inputs_arr, axis=0)
        
        inputs = torch.from_numpy(inputs).float()

        patient_id = inputs_files[0][-28:-18]

        return inputs, patient_id

def return_collate_fn(flag='train', output_num_frames=3):
    if flag == 'train':
        def collate_fn(data):
            inputs, targets, patient_id, targets_frame = zip(*data)
            batch_size = len(inputs)
            inputs_new = []
            targets_new = []
            for i in range(batch_size):
                h, w, num_slices, num_frames = inputs[i].shape
                inputs_temp = np.zeros((h, w, 12, output_num_frames))
                targets_temp = np.zeros((h, w, 12))
                # we pick frames first
                if output_num_frames == 3:
                    # ed
                    if targets_frame[i] == '01':
                        inputs_frame_selected = inputs[i][:,:,:,[0,0,1]]
                    elif targets_frame[i] == '04':
                        inputs_frame_selected = inputs[i][:,:,:,[2,3,4]]
                    #es
                    else:
                        targets_frame_int = int(targets_frame[i])
                        inputs_frame_selected = inputs[i][:,:,:,[targets_frame_int-2,targets_frame_int-1,targets_frame_int]]

                elif output_num_frames == 12:
                    # ed
                    if targets_frame[i] == '01':
                        inputs_frame_selected = inputs[i][:,:,:,:12]
                    elif targets_frame[i] == '04':
                        inputs_frame_selected = inputs[i][:,:,:,:12]
                    # es
                    else:
                        if num_frames == 12:
                            inputs_frame_selected = inputs[i][:,:,:,:12]
                        else:
                            targets_frame_int = int(targets_frame[i])
                            if (num_frames - targets_frame_int) > 6:
                                inputs_frame_selected = inputs[i][:,:,:,targets_frame_int-6:targets_frame_int+6]
                            else:
                                inputs_frame_selected = inputs[i][:,:,:,-12:]

                # then, we do padding for slices
                if num_slices <= 12:
                    inputs_temp[:,:,:num_slices,:] = inputs_frame_selected
                    targets_temp[:,:,:num_slices] = targets[i]

                else:
                    starts = num_slices - 11     
                    start = int(np.random.choice(starts, 1))
                    inputs_temp[:,:,:,:] = inputs_frame_selected[:,:,start:start+12,:]
                    targets_temp[:,:,:] = targets[i][:,:,start:start+12]
                    
                inputs_temp = torch.Tensor(inputs_temp).permute(3,0,1,2)
                targets_temp = torch.Tensor(targets_temp)

                inputs_new.append(inputs_temp)
                targets_new.append(targets_temp)

            inputs = torch.stack(inputs_new, 0)
            targets = torch.stack(targets_new, 0).squeeze()

            return inputs, targets, patient_id

        return collate_fn

    elif flag == 'test':
        def collate_fn(data):
            inputs, patient_id = zip(*data)
            batch_size = len(inputs)
            h,w = inputs[1].shape[1], inputs[1].shape[2]
            d = 12
            inputs_new = []

            for i in range(batch_size):
                # hard coded
                inputs_temp = np.zeros((3,h,w,d))
                z_inputs = inputs[i].shape[-1]

                if z_inputs <= d:
                    inputs_temp[:,:,:,:z_inputs] = inputs[i]
                    inputs_temp = torch.Tensor(inputs_temp)
                    inputs_new.append(inputs_temp)

                else:
                    starts = z_inputs - d + 1 
                    start = int(np.random.choice(starts, 1))
                    inputs_temp[:,:,:,:] = inputs[i][:,:,:,start:start+d]
                    inputs_temp = torch.Tensor(inputs_temp)
                    inputs_new.append(inputs_temp)

            inputs = torch.stack(inputs_new, 0)

            return inputs, patient_id
            
        return collate_fn

def get_loader(dataset, flag = "train", batch_size=4, shuffle=False, num_workers=0, transforms = None, fix_transform='none', num_folds=5, val_fold=0, seed=0, num_frames=3):
    if flag == "train":
        PATH = os.path.join('data', dataset,'training')
        # folds = split_training(PATH)
        folds = split_training_hardcoded(PATH)
        val_patient_ids = folds[val_fold*2] + folds[val_fold*2+1]
        train_patient_ids = []
        train_folds = folds[:val_fold]*2 + folds[val_fold*2+1:]
        for i in range(len(train_folds)):
            train_patient_ids += list(train_folds[i])
        train_set = myDataset(dataset=dataset, ids=train_patient_ids, transform=transforms, fix_transform=fix_transform)
        val_set = myDataset(dataset=dataset, ids=val_patient_ids, transform=None)

        collate_fn = return_collate_fn(flag='train', output_num_frames=num_frames)
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
        # print("Dataloader done!!!! ", time.time()-start_time)
        
        val_loader = torch.utils.data.DataLoader(dataset=val_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
        return train_loader, val_loader

    elif flag == 'test':
        PATH = os.path.join(PATH, 'test')
        test_ids = return_test_ids(PATH)
        test_set = myTestset(test_ids)
        collate_fn = return_collate_fn(flag='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
        return test_loader

if __name__ == "__main__":
    dataset = "ACDC"

    mytransforms = [
        RandomResize(5),
        RandomStretch(10,10),
        RandomRotation(45),
        RandomShear(0.25),
        RandomShift((9,9)),
        RandomGaussian((0.01, 0.05)),
        RandomCrop(144)
            ]

    train_loader, val_loader = get_loader(dataset, shuffle=False, transforms=mytransforms, fix_transform='random', val_fold=0, num_folds=5, num_frames=3)
    start_time = time.time()
    for i, (inputs, targets, patient_id) in enumerate(train_loader):
        print("patient id: ", patient_id)
        print("inputs shape: ", inputs.shape)
        # print("targets shape: ", torch.sum(targets[0].long()==2))
        break

    # train_loader, val_loader = get_loader(PATH, transforms=transforms, val_fold=3, num_folds=5, num_frames=12)
    # for i, (inputs2, targets2, patient_id) in enumerate(val_loader):
    #     # pass
    #     print("patient id: ", patient_id)
    #     # print("inputs shape: ", inputs2.shape)
    #     # print("targets shape: ", targets2.shape)
    #     break
    
    # same = inputs[0][0] == inputs2[0][4]
    # test_loader = get_loader(PATH, flag='test')
    # for i, (inputs, patient_id) in enumerate(test_loader):
    #     # print("patient id: ", patient_id)
    #     print("inputs shape: ", inputs.shape)
    #     break
    # print(same)
        
    