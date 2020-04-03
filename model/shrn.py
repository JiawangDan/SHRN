import math
import torch
import torch.nn as nn
import model.ops as ops
import numpy as np
import torch.nn.functional

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=True) #bias=False
        self.relu1 = nn.ReLU(True) #nn.ReLU()
        self.fc2   = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=True) #bias=False
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)

class Selective(nn.Module):
    def __init__(self, channels, reduce=4):
        super(Selective, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels//reduce)
        self.fc2_1 = nn.Linear(channels//reduce, channels)
        self.fc2_2 = nn.Linear(channels//reduce, channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        x1 = torch.unsqueeze(x1,dim=1)
        x2 = torch.unsqueeze(x2,dim=1)
        features = torch.cat([x1, x2], dim=1)
        fea_u = torch.sum(features, dim=1)
        fea_s = self.gap(fea_u).squeeze(dim=3).squeeze(dim=2)
        fea_z = self.fc1(fea_s)
        vector1 = self.fc2_1(fea_z).unsqueeze_(dim=1)
        vector2 = self.fc2_2(fea_z).unsqueeze_(dim=1)
        attention_vectors = torch.cat([vector1, vector2], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (features * attention_vectors).sum(dim=1)
        return fea_v

class SSPBlock(nn.Module):
    def __init__(self, channels, split=4, groups=1, bias=True, add=True):
        super(SSPBlock, self).__init__()
        self.split = split
        # self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=groups, bias=bias)
        # self.relu1 = nn.ReLU(inplace=True)
        self.channels_split = channels // split
        self.conv2_1 = nn.Conv2d(self.channels_split, self.channels_split, kernel_size=3, stride=1, padding=1, dilation=1, groups=groups, bias=bias)
        self.conv2_2 = nn.Conv2d(self.channels_split, self.channels_split, kernel_size=3, stride=1, padding=2, dilation=2, groups=groups, bias=bias)
        self.conv2_3 = nn.Conv2d(self.channels_split, self.channels_split, kernel_size=3, stride=1, padding=3, dilation=3, groups=groups, bias=bias)
        self.conv2_4 = nn.Conv2d(self.channels_split, self.channels_split, kernel_size=3, stride=1, padding=4, dilation=4, groups=groups, bias=bias)
        self.select1 = Selective(self.channels_split, reduce=4)
        self.select2 = Selective(self.channels_split, reduce=4)
        self.select3 = Selective(self.channels_split, reduce=4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1,  dilation=1, groups=groups, bias=bias)
        # self.CA = ChannelAttention(channels=64,reduction=16)
        # self.SA = SpatialAttention(kernel_size=3)
        self.add = add

    def forward(self, x):
        xs = torch.chunk(x, self.split, 1)
        ys = []
        for s in range(self.split):
            if s == 0:
                ys.append(self.conv2_1(xs[s]))
            elif s == 1:
                ys.append(self.select1(ys[-1], self.conv2_2(xs[s])))
            elif s == 2:
                ys.append(self.select2(ys[-1], self.conv2_3(xs[s])))
            elif s == 3:
                ys.append(self.select3(ys[-1], self.conv2_4(xs[s])))

        res = torch.cat(ys, dim=1)
        res = self.relu2(res)
        res = self.conv3(res)
        # res = self.SA(self.CA(res))
        # res = self.SA(res)

        if self.add:
            res += x
        return res

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()

        self.scale = kwargs.get("scale")
        self.multi_scale = kwargs.get("multi_scale")
        self.group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        modules_body = [
            SSPBlock(channels=64, split=4, groups=1, bias=True, add=True) \
            for _ in range(16) ]
        modules_body.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.body = nn.Sequential(*modules_body)

        self.upsample = ops.UpsampleBlock(64, scale=self.scale,
                                          multi_scale=self.multi_scale,
                                          group=self.group)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x, scale):
        x = self.sub_mean(x)
        x = self.entry(x)

        res = self.body(x)
        res += x

        res = self.upsample(res, scale=scale)

        res = self.exit(res)
        res = self.add_mean(res)

        return res
