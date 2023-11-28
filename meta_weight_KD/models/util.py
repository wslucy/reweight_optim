from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature-map layer)"""
    def __init__(self, s_shape, t_shape):
        super(ConvReg, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        self.s_H = s_H
        self.t_H = t_H
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, t):
        x = self.conv(x)
        if self.s_H == 2 * self.t_H or self.s_H * 2 == self.t_H or self.s_H >= self.t_H:
            return self.relu(self.bn(x)), t
        else:
            return self.relu(self.bn(x)), F.adaptive_avg_pool2d(t, (self.s_H, self.s_H))
            
class SelfA(nn.Module):
    """Cross-layer Self Attention"""
    def __init__(self, feat_dim, s_n, t_n, soft, factor=4): 
        super(SelfA, self).__init__()

        self.soft = soft
        self.s_len = len(s_n)
        self.t_len = len(t_n)
        self.feat_dim = feat_dim

        # query and key mapping
        for i in range(self.s_len):
            setattr(self, 'query_'+str(i), MLPEmbed(feat_dim, feat_dim//factor))
        for i in range(self.t_len):
            setattr(self, 'key_'+str(i), MLPEmbed(feat_dim, feat_dim//factor))
        
        for i in range(self.s_len):
            for j in range(self.t_len):
                setattr(self, 'regressor'+str(i)+str(j), Proj(s_n[i], t_n[j]))
               
    def forward(self, feat_s, feat_t):
        
        sim_s = list(range(self.s_len))
        sim_t = list(range(self.t_len))
        bsz = self.feat_dim

        # similarity matrix
        for i in range(self.s_len):
            sim_temp = feat_s[i].reshape(bsz, -1)
            sim_s[i] = torch.matmul(sim_temp, sim_temp.t())
        for i in range(self.t_len):
            sim_temp = feat_t[i].reshape(bsz, -1)
            sim_t[i] = torch.matmul(sim_temp, sim_temp.t())
        
        # calculate student query
        proj_query = self.query_0(sim_s[0])
        proj_query = proj_query[:, None, :]
        for i in range(1, self.s_len):
            temp_proj_query = getattr(self, 'query_'+str(i))(sim_s[i])
            proj_query = torch.cat([proj_query, temp_proj_query[:, None, :]], 1)
        
        # calculate teacher key
        proj_key = self.key_0(sim_t[0])
        proj_key = proj_key[:, :, None]
        for i in range(1, self.t_len):
            temp_proj_key = getattr(self, 'key_'+str(i))(sim_t[i])
            proj_key =  torch.cat([proj_key, temp_proj_key[:, :, None]], 2)
        
        # attention weight: batch_size X No. stu feature X No.tea feature
        energy = torch.bmm(proj_query, proj_key)/self.soft
        attention = F.softmax(energy, dim = -1)
        
        # feature dimension alignment
        proj_value_stu = []
        value_tea = []
        for i in range(self.s_len):
            proj_value_stu.append([])
            value_tea.append([])
            for j in range(self.t_len):            
                s_H, t_H = feat_s[i].shape[2], feat_t[j].shape[2]
                if s_H > t_H:
                    source = F.adaptive_avg_pool2d(feat_s[i], (t_H, t_H))
                    target = feat_t[j]
                elif s_H <= t_H:
                    source = feat_s[i]
                    target = F.adaptive_avg_pool2d(feat_t[j], (s_H, s_H))
                
                proj_value_stu[i].append(getattr(self, 'regressor'+str(i)+str(j))(source))
                value_tea[i].append(target)

        return proj_value_stu, value_tea, attention

class Proj(nn.Module):
    """feature dimension alignment by 1x1, 3x3, 1x1 convolutions"""
    def __init__(self, num_input_channels=1024, num_target_channels=128):
        super(Proj, self).__init__()
        self.num_mid_channel = 2 * num_target_channels
        
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        
        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv3x3(self.num_mid_channel, self.num_mid_channel),
            nn.BatchNorm2d(self.num_mid_channel),
            nn.ReLU(inplace=True),
            conv1x1(self.num_mid_channel, num_target_channels),
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class MLPEmbed(nn.Module):
    """non-linear mapping for attention calculation"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)
        self.regressor = nn.Sequential(
            nn.Linear(dim_in, 2 * dim_out),
            self.l2norm,
            nn.ReLU(inplace=True),
            nn.Linear(2 * dim_out, dim_out),
            self.l2norm,
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        
        return x

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class SRRL(nn.Module):
    """ICLR-2021: Knowledge Distillation via Softmax Regression Representation Learning"""
    def __init__(self, *, s_n, t_n): 
        super(SRRL, self).__init__()
                
        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
        ))
        
    def forward(self, feat_s, cls_t):
        
        feat_s = feat_s.unsqueeze(-1).unsqueeze(-1)
        temp_feat = self.transfer(feat_s)
        trans_feat_s = temp_feat.view(temp_feat.size(0), -1)

        pred_feat_s=cls_t(trans_feat_s)

        return trans_feat_s, pred_feat_s

class SimKD(nn.Module):
    """CVPR-2022: Knowledge Distillation with the Reused Teacher Classifier"""
    def __init__(self, *, s_n, t_n, factor=2): 
        super(SimKD, self).__init__()
       
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))       

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
        def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        
        # A bottleneck design to reduce extra parameters
        # 与self.transfer  = nn.Sequential(...)功能相同
        setattr(self, 'transfer', nn.Sequential(
            conv1x1(s_n, t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(t_n//factor, t_n//factor),
            # depthwise convolution
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(t_n//factor, t_n),
            nn.BatchNorm2d(t_n),
            nn.ReLU(inplace=True),
            ))
        
    def forward(self, feat_s, feat_t, cls_t): 
        # print(f"shape of feat_s,feat_t: feat_s.shape,feat_t.shape!!!") # (64, 256, 8, 8), (64, 256, 8, 8)
        # Spatial Dimension Alignment // 空间维度对齐
        s_H, t_H = feat_s.shape[2], feat_t.shape[2]
        # 把大的特征矩阵变小，使得两个特征矩阵的维度相同
        if s_H > t_H:
            source = F.adaptive_avg_pool2d(feat_s, (t_H, t_H))
            target = feat_t
        else:
            source = feat_s
            target = F.adaptive_avg_pool2d(feat_t, (s_H, s_H))
        
        trans_feat_t=target
        
        # Channel Alignment
        trans_feat_s = getattr(self, 'transfer')(source)

        # Prediction via Teacher Classifier
        temp_feat = self.avg_pool(trans_feat_s)
        temp_feat = temp_feat.view(temp_feat.size(0), -1)
        pred_feat_s = cls_t(temp_feat)
        
        return trans_feat_s, trans_feat_t, pred_feat_s
    
class MadKD(nn.Module):
    """ 
    Match in all dimension of KD by wslucy
    v1: 
    ys_model, yt_model for training the model
    ys_mlp, yt_mlp for training the mlp
    """
    def __init__(self, hidden_size=256, cls_num=100): 
        super(MadKD, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = cls_num
        self.output_size = cls_num

        self.grl = GradientReversalLayer()
        self.mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
        )
        self.apply(self.weight_init)      
        
    def forward(self, y_s, y_t, lambda_, labels):
        # trans logits only for training the model, only retain the grad of model
        # attantion! variable under this line will clear its' grad
        for param in self.mlp.parameters():
            param.requires_grad = False
        y_s_model = self.mlp(y_s)
        with torch.no_grad():
            y_t_model = self.mlp(y_t)
        for param in self.mlp.parameters():
            param.requires_grad = True
        
        # trans logits only for training the mlp, only retain the reverse grad of mlp
        y_s_mlp_rev = self.grl(self.mlp(y_s.detach()), lambda_)
        with torch.no_grad():
            y_t_mlp_rev = self.grl(self.mlp(y_t.detach()), lambda_)

        return y_s_model, y_t_model, y_s_mlp_rev, y_t_mlp_rev
    
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads.neg()
        return dx, None
    

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(200, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer1 = nn.Linear(hidden_size, 1)
        self.output_layer2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        delta = self.output_layer1(x)
        lambda_ = self.output_layer1(x)
        return delta, lambda_