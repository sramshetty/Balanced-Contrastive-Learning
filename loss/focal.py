import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    
    def __init__(self, cls_num_list, weight=None, gamma=2., reduction=None):
        super(FocalLoss, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, x, target):
        # log_prob = F.log_softmax(x, dim=-1)
        log_prob=self.m_list
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob, 
            target, 
            weight=self.weight,
            reduction = self.reduction
        )