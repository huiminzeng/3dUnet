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
    # Data augumentation
    mytransforms = [
        RandomResize(5),
        RandomStretch(10,10),
        RandomRotation(45),
        RandomShear(45),
        RandomShift((9,9)),
        RandomGaussian((0.001, 0.01)),
        RandomCrop(144)
            ]
    # Build dataloader
    for fold in range(args.start_fold, args.num_folds):
        # fold = args.val_fold
        print("fold: ", fold)
        # break
        train_loader, val_loader = get_loader(args.dataset, flag="train", batch_size=args.batch_size, shuffle=args.shuffle, 
                                              num_workers=args.num_workers, transforms=mytransforms, fix_transform=args.fix_transform, 
                                              val_fold=fold, num_folds=args.num_folds, num_frames=args.num_frames)

        torch.manual_seed(fold)
        model = UNet3D(in_channels=args.num_frames, base_n_filters=args.base_n_filters, n_classes=args.output_channels)
        
        # Loss and optimizer
        criterion = LossFunction(loss_type=args.loss_func, num_classes=args.num_classes, target_class=args.target_class)
        dice_score = DiceScore(num_classes=args.num_classes, target_class=args.target_class)
        
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate)

        # lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)

        if torch.cuda.is_available():
            model = model.to(cuda)
        else:
            pass
        
        ####################################################################
        ###################### TRAINING ####################################
        ####################################################################
        print("Training Start!!!!!!!!!! We are using {}-th fold as val fold".format(fold))
        num_batches = len(train_loader)
        print("we have %d batches in total!" %num_batches)

        best_model = None
        best_dice = 0
        best_epoch = 0
        train_loss_all = []
        train_dice_all = []
        val_loss_all = []
        val_dice_all = []

        for epoch in range(args.num_epochs):
            train_loss_his = [] # Recording the loss of classfication of clean data
            train_dice_his = []
            start_time = time.time()
            model.train()

            previous_time = time.time()
            for i, (inputs, targets, patient_id) in enumerate(train_loader):
                train_start = time.time()
                batch_size = inputs.shape[0]
                
                if torch.cuda.is_available():
                    inputs = inputs.to(cuda)
                    targets = targets.to(cuda)
                
                outputs, predictions = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                train_loss_his.append(loss.item())

                # Dice
                dice = dice_score.compute(targets, predictions)[0]
                train_dice_his.append(dice)

                # print("time: ", time.time() - previous_time)
                previous_time = time.time()
                
                if i % 10 == 0 or i == len(train_loader) - 1:
                    train_loss_mean = np.mean(train_loss_his)
                    train_dice_mean = np.mean(train_dice_his)
                    print("Train Epoch {}, Iter {}/{}, Train Loss: {}, DICE: {}".format(epoch, i, len(train_loader), train_loss_mean, train_dice_mean))
                # break

            train_loss_all.append(np.mean(train_loss_his))
            train_dice_all.append(np.mean(train_dice_his))
            ###############################################################
            #################### VALIDATION ###############################
            ###############################################################
            model.eval()
            val_loss_his = []
            val_dice_his = []
            # val_acc_his = []
            # val_iou_his = []

            for i, (inputs, targets, patient_id) in enumerate(val_loader):
                if torch.cuda.is_available():
                    inputs = inputs.to(cuda)
                    targets = targets.to(cuda)
                outputs, predictions = model(inputs)
                loss = criterion(outputs, targets)

                # Pixel Accuracy
                # acc = torch.mean((predictions==targets).float())
                # val_acc_his.append(acc.item())

                # Statistics
                val_loss_his.append(loss.item())

                # Dice
                dice = dice_score.compute(targets, predictions)[0]
                val_dice_his.append(dice)   
                # break     

            val_loss_mean = np.mean(val_loss_his)
            val_dice_mean = np.mean(val_dice_his)
            # val_acc_mean = np.mean(val_acc_his)
            # val_iou_mean = np.mean(val_iou_his)
            # print("EPOCH[{}|{}], DICE: {}, ACC: {}, IOU: {}, TIME: {}".format(epoch, args.num_epochs, 1-val_loss_mean, val_acc_mean, val_iou_mean, time.time()-start_time)) 
            print("EPOCH[{}|{}], DICE: {}, TIME: {}".format(epoch, args.num_epochs, val_dice_mean, time.time()-start_time)) 
            val_loss_all.append(val_loss_mean)
            val_dice_all.append(val_dice_mean)

            if val_dice_mean > best_dice:
                best_dice = val_dice_mean
                best_model = model.state_dict()
                best_optimizer = optimizer.state_dict()
                best_epoch = epoch
                
            lr_sheduler.step()

            if epoch % 10 == 0:

                save_curves(args, train_dice_all, val_dice_all, train_loss_all, val_loss_all, fold)
                model_name = str(args.dataset).lower() + '_net_3d_' + 'base_filter_' + str(args.base_n_filters) + '_val_' + str(fold)
                
                save_dir = get_save_dir(args)
                save_dir = os.path.join(save_dir, model_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print("saving model to: ", save_dir)

                save_name = os.path.join(save_dir, "best.checkpoint")
                best_checkpoint = {'model_state_dict': best_model,
                                'optimizer_state_dict': best_optimizer,
                                'epoch': best_epoch}

                torch.save(best_checkpoint, save_name)
            
        break

    print("TRAINGING IS OVER!!!!!!!!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str , default='0', help='specify the gpu')
    parser.add_argument('--save_dir', type=str , default='trained_models/augmentation_check')

    # Database parameters
    parser.add_argument('--dataset', type=str , default='ACDC', help='database for training')
    
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--val_fold', type=int, default=0)
    parser.add_argument('--start_fold', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=3)
    parser.add_argument('--fix_transform', type=str, default='none', help='rotate/shear/shift/gaussian/random')
    
    # loss function and model output channel
    parser.add_argument('--num_classes', type=int, default=4, help='if 2, then the dataset is RV/LV, if 4, then ACDC')
    parser.add_argument('--output_channels', type=int, default=4)

    # this is used for computing loss and for setting save path.
    # if the dataset is ACDC, but train a binary, this must be set as LV or RV
    # if the dataset is LV or RV, this mustb be set as the LV or RV (same as dataset name)
    parser.add_argument('--target_class', type=str, default='all', help='if ACDC, then all/RV/LV; if RV/LV, then RV/LV')
    parser.add_argument('--continue_training', type=bool, default=True)
    
    # Hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=5e-3)
    parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the data')
    parser.add_argument('--base_n_filters', type=int, default=4)
    parser.add_argument('--loss_func', type=str, default='DiceLoss')

    args = parser.parse_args()
    print(args)
    main(args)