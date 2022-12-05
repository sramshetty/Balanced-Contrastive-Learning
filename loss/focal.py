import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


# class LogitAdjust(nn.Module):

#     def __init__(self, cls_num_list, tau=1, weight=None):
#         super(LogitAdjust, self).__init__()
#         cls_num_list = torch.cuda.FloatTensor(cls_num_list)
#         cls_p_list = cls_num_list / cls_num_list.sum()
#         m_list = tau * torch.log(cls_p_list)
#         self.m_list = m_list.view(1, -1)
#         self.weight = weight

#     def forward(self, x, target):
#         x_m = x + self.m_list
#         return F.cross_entropy(x_m, target, weight=self.weight)


# class FocalLoss(nn.Module):
#     def __init__(self, cls_num_list, weight=None, gamma=2.):
#         super(FocalLoss, self).__init__()
#         cls_num_list = torch.cuda.FloatTensor(cls_num_list)
#         cls_p_list = cls_num_list / cls_num_list.sum()
#         m_list = torch.log(cls_p_list)
#         self.m_list = m_list.view(1, -1)
#         self.weight = weight
#         self.gamma = gamma

#     def forward(self, x, target):
#         # log_prob = F.log_softmax(x, dim=-1)
#         log_prob= x + self.m_list
#         prob = torch.exp(log_prob)
#         return F.nll_loss(
#             ((1 - prob) ** self.gamma) * log_prob, 
#             target,
#             weight=self.weight
#         )

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
