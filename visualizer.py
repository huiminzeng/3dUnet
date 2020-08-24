import os 
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataloader_new import *
from model_new import *
from loss import *
from utils import *


def dataloader_check(path= "data/ACDC/", fix_transform=None, n_frames=3):
    PATH = "ACDC"

    mytransforms = [
        RandomResize(5),
        RandomStretch(10,10),
        RandomRotation(45),
        RandomShear(0.25),
        RandomShift((10,10)),
        RandomGaussian([0.001, 0.01]),
        RandomCrop(144)
            ]

    train_loader, val_loader = get_loader(PATH, shuffle=False, transforms=mytransforms, fix_transform=fix_transform, val_fold=0, num_folds=5, num_frames=n_frames)
    for i, (inputs, targets, patient_id) in enumerate(train_loader):
        inputs_arr = inputs.numpy()
        targets_arr = targets.numpy()

        print("patient id: ", patient_id)
        if fix_transform != None:
            save_dir = os.path.join('sanity_check/dataloader_check', 'augmentation_' + fix_transform)
        else:
            save_dir = os.path.join('sanity_check/dataloader_check', 'no_augmentation')
        save_dir = os.path.join(save_dir, 'n_frames_' + str(n_frames))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if i < 3:
            np.save(os.path.join(save_dir, "inputs_tensor_{}".format(i)), inputs_arr)
            np.save(os.path.join(save_dir, "targets_tensor_{}".format(i)), targets_arr)
            continue
        break

def save_preds(args, inputs, targets, preds, patient_id, model_name):
    save_dir_root = 'images/normal_training'
    if args.output_channels == 4:
        save_dir = os.path.join(save_dir_root, 'multi_class')
    elif args.output_channels == 2:
        save_dir = os.path.join(save_dir_root, 'binary')
        if args.target_class == 'LV':
            save_dir = os.path.join(save_dir, 'LV')
        elif args.target_class == 'RV':
            save_dir = os.path.join(save_dir, 'RV')

    if args.num_frames == 3:
        save_dir = os.path.join(save_dir, '3_frames')
    elif args.num_frames ==12:
        save_dir = os.path.join(save_dir , '12_frames')
    
    if args.fix_transform != None:
        save_dir = os.path.join(save_dir, 'augmentation_' + args.fix_transform)
    else:
        save_dir = os.path.join(save_dir, 'no_augmentation')

    save_dir = os.path.join(save_dir, "base_filter_" + str(args.base_n_filters))
    save_dir = os.path.join(save_dir, "batch_size_" + str(args.batch_size))

    save_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("saving predictions to: ", save_dir)

    np.save(os.path.join(save_dir, 'inputs'), inputs.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'predictions'), preds.detach().cpu().numpy())
    np.save(os.path.join(save_dir, 'targets'), targets.detach().cpu().numpy())


def save_preds_sub(args, inputs, preds, patient_id):
    batch_size = inputs.shape[0]
    save_dir_root = 'images'
    if args.num_classes == 4:
        save_dir_root = os.path.join(save_dir_root, 'submission')
        if args.transforms:
            save_dir = os.path.join(save_dir_root, 'augumented')
        else:
            save_dir = os.path.join(save_dir_root, 'plain')
    
    save_dir = os.path.join(save_dir, "base_filter_"+str(args.base_n_filters))
    save_dir = os.path.join(save_dir, "batch_size_"+str(args.batch_size))

    for i in range(batch_size):
        current_dir = os.path.join(save_dir, patient_id[i])
        if not os.path.exists(current_dir):
            os.makedirs(current_dir)
        np.save(os.path.join(current_dir, 'inputs'), inputs[i].detach().cpu().numpy())
        np.save(os.path.join(current_dir, 'predictions'), preds[i].detach().cpu().numpy())


def save_curves(args, train_dice, val_dice, train_loss, val_loss, fold):
    num_epochs = len(train_dice)
    _, axs = plt.subplots(1, 2, figsize=(48,18))
    axs[0].plot(list(range(num_epochs)), train_dice, label='Training DICE')
    axs[0].plot(list(range(num_epochs)), val_dice, label='Validation DICE')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('DICE')
    axs[0].legend()

    axs[1].plot(list(range(num_epochs)), train_loss, label='Training Loss')
    axs[1].plot(list(range(num_epochs)), val_loss, label='Validation Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    save_dir_root = 'curves/normal_training'
    if args.output_channels == 4:
        save_dir = os.path.join(save_dir_root, 'multi_class')
    elif args.output_channels == 2:
        save_dir = os.path.join(save_dir_root, 'binary')
        if args.target_class == 'LV':
            save_dir = os.path.join(save_dir, 'LV')
        elif args.target_class == 'RV':
            save_dir = os.path.join(save_dir, 'RV')

    if args.num_frames == 3:
        save_dir = os.path.join(save_dir, '3_frames')
    elif args.num_frames ==12:
        save_dir = os.path.join(save_dir , '12_frames')
    
    if args.fix_transform != None:
        save_dir = os.path.join(save_dir, 'augmentation_' + args.fix_transform)
    else:
        save_dir = os.path.join(save_dir, 'no_augmentation')

    save_dir = os.path.join(save_dir, "base_filter_" + str(args.base_n_filters))
    save_dir = os.path.join(save_dir, "batch_size_" + str(args.batch_size))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("saving curve to: ", save_dir)
    curve_name = str(args.dataset).lower() + '_net_3d_' + 'base_filter_num_' + str(args.base_n_filters) + '_training_curve_lr_' + str(args.learning_rate) + '_val_'+str(fold) +'.jpg'

    save_name = os.path.join(save_dir, curve_name)
    plt.savefig(save_name)

def load_pretrained(args):
    # load pretrained model
    if args.output_channels == 4:
        load_dir = os.path.join(args.load_dir, 'multi_class')
    elif args.output_channels == 2:
        load_dir = os.path.join(args.load_dir, 'binary')
        if args.target_class == 'LV':
            load_dir = os.path.join(load_dir, 'LV')
        elif args.target_class == 'RV':
            load_dir = os.path.join(load_dir, 'RV')
            
    if args.num_frames == 3:
        load_dir = os.path.join(load_dir, '3_frames')
    elif args.num_frames ==12:
        load_dir = os.path.join(load_dir , '12_frames')

    if args.fix_transform != 'none':
        load_dir = os.path.join(load_dir, 'augmentation_' + args.fix_transform)
    else:
        load_dir = os.path.join(load_dir, 'no_augmentation')

    load_dir = os.path.join(load_dir, "base_filter_"+str(args.base_n_filters))
    load_dir = os.path.join(load_dir, 'batch_size_'+ str(args.batch_size))

    return load_dir

def get_save_dir(args):
    if args.output_channels == 4:
        save_dir = os.path.join(args.save_dir, 'multi_class')
    elif args.output_channels == 2:
        save_dir = os.path.join(args.save_dir, 'binary')
        if args.target_class == 'LV':
            save_dir = os.path.join(save_dir, 'LV')
        elif args.target_class == 'RV':
            save_dir = os.path.join(save_dir, 'RV')
            
    if args.num_frames == 3:
        save_dir = os.path.join(save_dir, '3_frames')
    elif args.num_frames ==12:
        save_dir = os.path.join(save_dir , '12_frames')
        
    if args.fix_transform != 'none':
        save_dir = os.path.join(save_dir, 'augmentation_' + args.fix_transform)
    else:
        save_dir = os.path.join(save_dir, 'no_augmentation')

    save_dir = os.path.join(save_dir, "base_filter_" + str(args.base_n_filters))
    save_dir = os.path.join(save_dir, "batch_size_" + str(args.batch_size))
    
    return save_dir

if __name__ == "__main__":
    # print("resizing")
    # dataloader_check(fix_transform='resize', n_frames=3)
    # print("stretching")
    # dataloader_check(fix_transform='stretch', n_frames=3)
    # print("rotating")
    # dataloader_check(fix_transform='rotate', n_frames=3)
    # print("shear")
    # dataloader_check(fix_transform='shear', n_frames=3)
    # print("shift")
    # dataloader_check(fix_transform='shift', n_frames=3)
    print("gaussian")
    dataloader_check(fix_transform='gaussian', n_frames=3)
    # print("none")
    # dataloader_check(fix_transform=None, n_frames=3)

    pass