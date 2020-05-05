import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, loss_type='CrossEntropy'):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        if loss_type == 'CrossEntropy':
            self._loss_fn = CrossEntropyLoss()
        elif loss_type == 'DiceLoss':
            self._loss_fn = DiceLoss()
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
    def __init__(self):
        super(DiceLoss, self).__init__()
    def one_hot(self, targets):
        mask_0 = (targets == 0).unsqueeze(1)
        mask_1 = (targets == 1).unsqueeze(1)
        self.mask = torch.cat((mask_0,mask_1), dim=1)
    def forward(self, outputs, targets):
        self.one_hot(targets)
        probability = F.softmax(outputs, dim=1)
        loss = torch.sum(self.mask * probability, dim=(2,3)) / (torch.sum(self.mask, dim=(2,3)) + torch.sum(probability, dim=(2,3)))
        loss = torch.sum(loss, dim=1)
        loss = 1 - torch.mean(loss)
        return loss