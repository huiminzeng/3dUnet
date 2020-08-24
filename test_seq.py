import os 
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# if use cross validation
from dataloader_new import *

from model_new import *
from loss import *
from utils import *
from visualizer import *

# Main funciton
def main(args):
    # specify GPU
    cuda = torch.device('cuda:'+args.cuda)

    # Build model
    model = UNet3D(in_channels=args.num_frames, base_n_filters=args.base_n_filters, n_classes=args.output_channels)
    
    # currently, we use val_loader to evaluate the model
    # Dice score
    dice_score = DiceScore_No_Padding(args.num_classes, args.target_class)

    # load_dir_root = args.load_dir
    load_dir = load_pretrained(args)
    num_models = len(os.listdir(load_dir))

    cross_val_his = []
    cross_rv_his = []
    cross_outer_his = []
    cross_lv_his = []

    for i in range(num_models):
        train_loader, val_loader = get_loader(args.dataset_test, flag="train", batch_size=args.batch_size, shuffle=args.shuffle, 
                                              num_workers=args.num_workers, transforms=None, fix_transform=args.fix_transform, 
                                              val_fold=i, num_folds=args.num_folds, num_frames=args.num_frames)
                                              
        # model_folder = str(args.dataset_pretrained).lower() + '_net_3d_multiclass_' + 'base_filter_' + str(args.base_n_filters) + "_val_" + str(i)
        model_folder = str(args.dataset_pretrained).lower() + '_net_3d_' + 'base_filter_' + str(args.base_n_filters) + "_val_" + str(i)
        load_model = os.path.join(load_dir, model_folder)
        checkpoint = os.path.join(load_model, 'best.checkpoint')
        print("we are loading model: ", load_model)
        model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
        # Targeted attack
        if torch.cuda.is_available():
            model = model.to(cuda)
        
        ###############################################################
        #################### VALIDATION ###############################
        ###############################################################
        model.eval()
        val_dice_his = []
        val_rv_his = []
        val_outer_his = []
        val_lv_his = []
        # val_acc_his = []
        # val_iou_his = []

        for i, (inputs, targets, patient_id) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs = inputs.to(cuda)
                targets = targets.to(cuda)

            outputs, predictions = model(inputs)

            # Pixel Accuracy
            # acc = torch.mean((predictions==targets).float())
            # val_acc_his.append(acc.item())

            # Dice
            dice, dice_rv, dice_outer, dice_lv = dice_score.compute(targets, predictions)
            val_dice_his.append(dice)   
            val_rv_his.append(dice_rv)   
            val_outer_his.append(dice_outer)   
            val_lv_his.append(dice_lv)       
            # break 

        val_dice_mean = np.mean(val_dice_his)
        val_rv_mean = np.mean(val_rv_his)
        val_outer_mean = np.mean(val_outer_his)
        val_lv_mean = np.mean(val_lv_his)

        cross_val_his.append(val_dice_mean)
        cross_rv_his.append(val_rv_mean)
        cross_outer_his.append(val_outer_mean)
        cross_lv_his.append(val_lv_mean)

        if args.cross_validation:
            print("we are saving visualization!")
            save_preds(args, inputs, targets, predictions, patient_id, model_folder)
            continue
        else:
            print("we don't perform cross validation!")
            print("we are saving visualization!")
            save_preds(args, inputs, targets, predictions, patient_id, model_folder)
            break

    cross_val_mean = np.mean(cross_val_his)
    cross_val_std = np.std(cross_val_his)

    cross_rv_mean = np.mean(cross_rv_his)
    cross_rv_std = np.std(cross_rv_his)

    cross_outer_mean = np.mean(cross_outer_his)
    cross_outer_std = np.std(cross_outer_his)

    cross_lv_mean = np.mean(cross_lv_his)
    cross_lv_std = np.std(cross_lv_his)

    # cross validaton
    print("DICE MEAN: {}, DICE STD: {}".format(cross_val_mean, cross_val_std)) 
    print("RV MEAN: {}, RV STD: {}".format(cross_rv_mean, cross_rv_std)) 
    print("OUTER MEAN: {}, OUTER STD: {}".format(cross_outer_mean, cross_outer_std)) 
    print("LV MEAN: {}, LV STD: {}".format(cross_lv_mean, cross_lv_std)) 
    print("="*15)
    print("="*15)
    print("TESTING OVER!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str , default='0', help='specify the gpu')
    parser.add_argument('--load_dir', type=str , default='trained_models/augmentation_check')

    # Database parameters
    parser.add_argument('--dataset_pretrained', type=str , default='ACDC', help='database for training')
    parser.add_argument('--dataset_test', type=str , default='ACDC', help='database for testing')
    
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--transforms', type=bool, default=True)
    parser.add_argument('--num_folds', type=int, default=5)

    # this just refers to the loss function used
    parser.add_argument('--num_frames', type=int, default=3)

    # this is used for computing loss and for setting save path.
    # if the dataset is ACDC, but train a binary, this must be set as LV or RV
    # if the dataset is LV or RV, this mustb be set as the LV or RV (same as dataset name)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--output_channels', type=int, default=4)
    parser.add_argument('--target_class', type=str, default='all')
    
    # Hyperparameterss
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle the data')
    parser.add_argument('--base_n_filters', type=int, default=4)
    parser.add_argument('--fix_transform', type=str, default='none', help='rotate/shear/shift/gaussian/random')
    

    # test setting
    parser.add_argument('--cross_validation', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)