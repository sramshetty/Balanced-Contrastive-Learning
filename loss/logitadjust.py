import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None):
        super(LogitAdjust, self).__init__()
        cls_num_list = torch.cuda.FloatTensor(cls_num_list)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        self.weight = weight

    def forward(self, x, target):
        x_m = x + self.m_list
        # Soft labels https://stackoverflow.com/questions/68907809/soft-cross-entropy-in-pytorch#:~:text=Pytorch%20CrossEntropyLoss%20Supports%20Soft%20Labels,target%20(see%20the%20doc).
        # p = F.log_softmax(x, 1, dtype=torch.float)
        # w_labels = self.weight*target if self.weight is not None else target
        # loss = -(w_labels*p).sum() / (w_labels).sum()
        # return loss / x.shape[0]
        return F.cross_entropy(x_m, target, weight=self.weight)
