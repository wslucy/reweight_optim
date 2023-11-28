from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MadKDLoss(nn.Module):
    """
    v1: 
    
    """
    def  __init__(self, temperature = 4, alpha=1, beta=1):
        super(MadKDLoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_s_model, y_t_model, y_s_mlp_rev, y_t_mlp_rev, target, epoch):
        # KL loss for model
        loss_kd_model = kd_loss(y_s_model, y_t_model, self.temperature)
        
        # reverse KL loss for mlp
        loss_kd_mlp = kd_loss(y_s_mlp_rev, y_t_mlp_rev, self.temperature)

        return self.alpha * loss_kd_model #+ self.beta * loss_kd_mlp

#############################kd#############################
def kd_loss(y_s, y_t, temperature):
    p_s = F.log_softmax(y_s/temperature, dim=1)
    p_t = F.softmax(y_t/temperature, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (temperature**2)
    return loss
#############################kd#############################


############################dkd#############################
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
############################dkd#############################


############################DIST############################
def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))

def dist_loss(z_s, z_t, beta=2.0, gamma=2.0, tau=4.0):
        y_s = (z_s / tau).softmax(dim=1)
        y_t = (z_t / tau).softmax(dim=1)
        inter_loss = tau**2 * inter_class_relation(y_s, y_t)
        intra_loss = tau**2 * intra_class_relation(y_s, y_t)
        loss = beta * inter_loss + gamma * intra_loss
        return loss
############################DIST############################