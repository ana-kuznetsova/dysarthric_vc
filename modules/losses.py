import torch
from torch import nn
from  torch.nn import functional as F


#MI loss from official implementation of CLUB
#https://github.com/Linear95/CLUB/blob/master/mi_estimators.py

class EncLossGeneral(nn.Module):
    '''
    This class computes the loss over basic setup of the encoder
    The loss is the weighted sum over embedding reconstru
    '''
    def __init__(self, alpha1=1, alpha2=1):
        super(EncLossGeneral, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.rc_loss = nn.L1Loss()
        self.spk_ce_loss = nn.CrossEntropyLoss()

    def forward(self, feats, rc_feats, cls_out, cls_target):
        loss_1 = self.alpha1*self.rc_loss(feats, rc_feats)
        loss_2 = self.alpha2*self.spk_ce_loss(cls_out, cls_target)
        total_loss = loss_1 + loss_2
        return total_loss