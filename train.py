import os 
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from dataloader import get_loader
from model_new import *
from loss import *
from utils import *

# Main funciton
def main(args):
    # specify GPU
    cuda = torch.device('cuda:'+args.cuda)
    # Data augumentation
    if args.transforms:
        transforms = [
                RandomCrop(148)
            ]
    else:
        transforms = None
    # Build dataloader
    train_loader, val_loader = get_loader(PATH=args.data_path+'training', flag="train", batch_size=args.batch_size,
                                          shuffle=args.shuffle, num_workers=args.num_workers, transforms = transforms)

    model = UNet3D_ACDC()
    # Loss and optimizer
    criterion = LossFunction(loss_type=args.loss_func)
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    lr_sheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.99)
    # Targeted attack
    if torch.cuda.is_available():
        model = model.to(cuda)
    else:
        pass

    # Training
    num_batches = len(train_loader)
    print("we have %d batches in total!" %num_batches)
    best_model = 0
    best_loss = 1e5
    ####################################################################
    ###################### TRAINING ####################################
    ####################################################################
    print("Training Start!!!!!!!!!!")
    for epoch in range(args.num_epochs):
        train_loss_his = [] # Recording the loss of classfication of clean data
        start_time = time.time()
        model.train()

        for i, (inputs, targets) in enumerate(train_loader):
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

            if i % 10 == 0 or i == len(train_loader) - 1:
                train_loss_mean = np.mean(np.array(train_loss_his))
                # print("latent vector eshape: ", latent_vector.shape)
                print("{}, Train Epoch {}, Iter {}/{}, Train Loss: {}".format(
                    time.strftime("%Y-%b-%d %H:%M:%S"), epoch, i, len(train_loader), train_loss_mean))

        ###############################################################
        #################### VALIDATION ###############################
        ###############################################################
        model.eval()
        val_loss_his = []

        for i, (inputs, targets) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs = inputs.to(cuda)
                targets = targets.to(cuda)

            outputs, predictions = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            val_loss_his.append(loss.item())

        val_loss_mean = np.mean(np.array(val_loss_his))
        print("EPOCH[{}|{}], VAL LOSS: {}, TIME: {}".format(epoch, args.num_epochs, val_loss_mean, time.time()-start_time)) 

        if epoch % 10 == 0:
            np.save("results/ES_image.npy", (inputs[0].squeeze())[:,:,2].detach().cpu().numpy())
            np.save("results/ES_outputs_{}.npy".format(epoch), (predictions[0][:,:,2].detach().cpu().numpy()))
            np.save("results/ES_mask.npy", (targets[0].squeeze())[:,:,2].detach().cpu().numpy())

        if val_loss_mean < best_loss:
            best_model = model
            best_loss = val_loss_mean
        
        lr_sheduler.step()

    # torch.save(best_model.state_dict(), 'trained_model/acdc_net_3d')
    print("TRAINGING IS OVER!!!!!!!!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=str , default='0', help='specify the gpu')

    # Database parameters
    parser.add_argument('--data_path', type=str , default='data/ACDC/', help='database for training')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--transforms', type=bool, default=False)

    # Hyperparameterss
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--shuffle', type=bool, default=False, help='Shuffle the data')
    parser.add_argument('--base_n_filters', type=int, default=26)
    parser.add_argument('--loss_func', type=str, default='DiceLoss')

    args = parser.parse_args()
    print(args)
    main(args)