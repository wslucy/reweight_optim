from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class CSKDLoss(nn.Module):
    """
    CSKD by Wslucy:
    """
    def  __init__(self, alpha = 1, beta = 8, temperature = 4):
        super(CSKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, y_s, y_t, delta, lambda_, target, epoch):
        # correct strict y_t, y_s
        y_t, y_s = cskd(y_t, y_s, delta, lambda_, target)
        # KL loss
        loss_kd = kd_loss(y_s, y_t, self.temperature)
        return loss_kd 

def cskd(y_t, y_s, delta, lambda_, target):
    y_t_target = y_t * torch.zeros_like(y_t).scatter_(1, target.unsqueeze(1), 1)
    y_t_delta = delta * torch.zeros_like(y_t).scatter_(1, target.unsqueeze(1), 1)
    y_t = y_t_target - y_t_delta

    y_s_target = y_s * torch.zeros_like(y_s).scatter_(1, target.unsqueeze(1), 1)
    y_s_lambda = lambda_ * torch.zeros_like(y_s).scatter_(1, target.unsqueeze(1), 1)
    y_s = y_s_target - y_t_delta - y_s_lambda

    return y_t, y_s

def kd_loss(y_s, y_t, temperature):
    p_s = F.log_softmax(y_s/temperature, dim=1)
    p_t = F.softmax(y_t/temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (temperature**2)
    return loss
