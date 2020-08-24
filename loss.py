import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, loss_type='CrossEntropy', num_classes=4, target_class='all'):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'CrossEntropy':
            self._loss_fn = CrossEntropyLoss()
        elif loss_type == 'DiceLoss':
            self._loss_fn = DiceLoss(num_classes, target_class)
        else:
            raise NotImplementedError
    def forward(self, outputs, targets):
        return self._loss_fn(outputs, targets)

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.xe_loss = nn.CrossEntropyLoss()
    def forward(self, outputs, targets):
        device = outputs.device
        self.xe_loss = self.xe_loss.to(device)
        return self.xe_loss(outputs, targets)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, target_class):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.target_class = target_class

    def one_hot(self, targets):
        masks = []
        for i in range(self.num_classes):
            mask_temp = (targets == i).unsqueeze(1)
            masks.append(mask_temp)
        self.mask = torch.cat(masks, dim=1)
        
    def forward(self, outputs, targets):
        self.one_hot(targets)
        probability = F.softmax(outputs, dim=1)

        if self.num_classes == 4:
            if self.target_class == 'RV':
                # we loss for RV + BG
                mask_partial = self.mask[:,[0,1],:,:]
                loss = torch.sum(mask_partial * probability, dim=(2,3,4)) / (torch.sum(mask_partial, dim=(2,3,4)) + torch.sum(probability, dim=(2,3,4)))
            
            elif self.target_class == 'LV':
                # we loss for LV + BG
                mask_partial = self.mask[:,[0,3],:,:]
                loss = torch.sum(mask_partial * probability, dim=(2,3,4)) / (torch.sum(mask_partial, dim=(2,3,4)) + torch.sum(probability, dim=(2,3,4)))
            
            elif self.target_class == 'all':
                loss = torch.sum(self.mask * probability, dim=(2,3,4)) / (torch.sum(self.mask, dim=(2,3,4)) + torch.sum(probability, dim=(2,3,4)))
            
            loss = 2*torch.mean(loss, dim=1)
            loss = torch.mean(1 - loss)
        
        elif self.num_classes == 2:
            loss = torch.sum(self.mask * probability, dim=(2,3,4)) / (torch.sum(self.mask, dim=(2,3,4)) + torch.sum(probability, dim=(2,3,4)))
            
            loss = 2*torch.mean(loss, dim=1)
            loss = torch.mean(1 - loss)


        return loss
    
class DiceScore():
    def __init__(self, num_classes, target_class):
        super(DiceScore, self).__init__()
        self.num_classes = num_classes
        self.target_class = target_class

    def compute(self, targets, predictions):
        masks = []
        masks_preds = []

        if self.num_classes == 4: 
        
            if self.target_class == 'RV':
                mask = (targets == 1).type(torch.uint8)
                dice_score = torch.sum((mask * predictions).float(), dim=(1,2,3)) / (torch.sum(mask, dim=(1,2,3)) + torch.sum(predictions, dim=(1,2,3))).float()
                dice_score = 2*torch.mean(dice_score, dim=0).item()
                return dice_score, 0, 0, 0

            elif self.target_class == 'LV':
                mask = (targets == 3).type(torch.uint8)
                dice_score = torch.sum((mask * predictions).float(), dim=(1,2,3)) / (torch.sum(mask, dim=(1,2,3)) + torch.sum(predictions, dim=(1,2,3))).float()
                dice_score = 2*torch.mean(dice_score, dim=0).item()
                return dice_score, 0, 0, 0
        
            elif self.target_class == 'all':
                for i in range(self.num_classes):
                    mask_temp = (targets == i).unsqueeze(1).type(torch.uint8)
                    masks.append(mask_temp)

                    mask_pred_temp = (predictions == i).unsqueeze(1).type(torch.uint8)
                    masks_preds.append(mask_pred_temp)

                mask = torch.cat(masks, dim=1).detach().cpu().numpy() # (batch_size, num_classses, h, w, d)
                mask_pre = torch.cat(masks_preds, dim=1).detach().cpu().numpy() # (batch_size, num_classes, h, w, d)
                dice_score = np.sum(mask * mask_pre, axis=(2,3,4)) / (np.sum(mask, axis=(2,3,4)) + np.sum(mask_pre, axis=(2,3,4)))
                dice_score = np.mean(2*np.mean(dice_score, axis=1))
                
                # print("mask shape: ", mask.shape)
                dice_split = [] # RV, LV outer, LV
                for i in range(1,4):
                    mask_each = mask[:,[0,i],:,:,:]
                    mask_each_pre = mask_pre[:,[0,i],:,:,:]

                    dice_each = np.sum(mask_each * mask_each_pre, axis=(2,3,4)) / (np.sum(mask_each, axis=(2,3,4)) + np.sum(mask_each_pre, axis=(2,3,4)))
                    dice_each = np.mean(2*np.mean(dice_each, axis=1))

                    dice_split.append(dice_each)

                return dice_score, dice_split[0], dice_split[1], dice_split[2]
            
        elif self.num_classes == 2:
            mask = (targets == 1).type(torch.uint8)
            dice_score = torch.sum((mask * predictions).float(), dim=(1,2,3)) / (torch.sum(mask, dim=(1,2,3)) + torch.sum(predictions, dim=(1,2,3)))
            dice_score = 2*torch.mean(dice_score, dim=0).item()
            return dice_score, 0, 0, 0



class DiceScore_No_Padding():
    def __init__(self, num_classes, target_class):
        super(DiceScore_No_Padding, self).__init__()
        self.num_classes = num_classes
        self.target_class = target_class

    def compute(self, targets, predictions):
        batch_size = targets.shape[0]
        dice_score_batch = []
        dice_split_batch = []

        if self.num_classes == 4:
            for i in range(batch_size):
                slice_pick = torch.sum(targets[i], dim=(0,1)) > 0
                slice_pick = torch.sum(slice_pick)

                if self.target_class == 'RV':
                    mask_selected = (targets[i][:,:,:slice_pick] == 1).type(torch.uint8)
                    preds_selected = predictions[i][:,:,:slice_pick] # this is binary classification, we should predict 1
                    
                    dice_score_per_patient = 2 * torch.sum((mask_selected*preds_selected).float()) / (torch.sum(mask_selected) + torch.sum(predictions)).float()
                    dice_score_per_patient = dice_score_per_patient.item()

                    dice_score_batch.append(dice_score_per_patient)
                    dice_split_batch.append([0,0,0])

                elif self.target_class == 'LV':
                    mask_selected = (targets[i][:,:,:slice_pick] == 3).type(torch.uint8)
                    preds_selected = predictions[i][:,:,:slice_pick] # this is binary classification, we should predict 1
                    
                    dice_score_per_patient = 2 * torch.sum((mask_selected*preds_selected).float()) / (torch.sum(mask_selected) + torch.sum(predictions)).float()
                    dice_score_per_patient = dice_score_per_patient.item()

                    dice_score_batch.append(dice_score_per_patient)
                    dice_split_batch.append([0,0,0])
        
                elif self.target_class == 'all':
                    mask_selected_per_patient = []
                    preds_selected_per_patient = []
                    
                    for j in range(1, self.num_classes):
                        mask_selected = (targets[i][:,:,:slice_pick] == j).type(torch.uint8)
                        preds_selected = (predictions[i][:,:,:slice_pick] == j).type(torch.uint8)
                        
                        mask_selected_per_patient.append(mask_selected)
                        preds_selected_per_patient.append(preds_selected)

                    mask = torch.stack(mask_selected_per_patient, dim=0).detach().cpu().numpy() # (num_classses, h, w, d_selected)
                    preds = torch.stack(preds_selected_per_patient, dim=0).detach().cpu().numpy() # (num_classes, h, w, d_selected)
                    
                    dice_score = np.sum(mask * preds, axis=(1,2,3)) / (np.sum(mask, axis=(1,2,3)) + np.sum(preds, axis=(1,2,3)))
                    dice_score = 2*np.mean(dice_score)
                    dice_score_batch.append(dice_score)
                
                    # print("mask shape: ", mask.shape) # (3, H, W, D)
                    dice_split = [] # RV, LV outer, LV
                    for j in range(0, self.num_classes-1):
                        mask_each_class = mask[j,:,:,:]
                        preds_each_class = preds[j,:,:,:]

                        dice_each = np.sum(mask_each_class * preds_each_class) / (np.sum(mask_each_class) + np.sum(preds_each_class))
                        dice_each = 2*np.mean(dice_each)

                        dice_split.append(dice_each)

                    dice_split_batch.append(dice_split)
        
        elif self.num_classes == 2:
            print("the dataset has binary classes")
            for i in range(batch_size):
                slice_pick = torch.sum(targets[i], dim=(0,1)) > 0
                mask_selected = (targets[i][:,:,slice_pick] == 1).type(torch.uint8)
                preds_selected = predictions[i][:,:,slice_pick] # this is binary classification, we should predict 1
                
                print("mask selected shape: ", mask_selected.shape)
                
                dice_score_per_patient = 2 * torch.sum((mask_selected * preds_selected).float()) / (torch.sum(mask_selected) + torch.sum(preds_selected))
                print("dice score per patient: ", dice_score_per_patient.item())
                dice_score_per_patient = dice_score_per_patient.item()

                dice_score_batch.append(dice_score_per_patient)
                dice_split_batch.append([0,0,0])

        dice_score_batch = np.mean(np.array(dice_score_batch))
        dice_split_batch = np.mean(np.array(dice_split_batch), axis=0)

        return dice_score_batch, dice_split_batch[0], dice_split_batch[1], dice_split_batch[2]