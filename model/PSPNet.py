import torch
import torch.nn as nn
import torch.nn.functional as F

from .ResNet import resnet50

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

'''
Pyramid Pooling Module
pooling n x n -> 1 x 1 conv -> bilinear interpolation
'''

class PPModule(nn.Module):
    def __init__(
        self,
        channels: int       # #of feature map
    ):
        super(PPModule, self).__init__()

        self.pooled_channels = int(channels/4)
        self.channels = channels

        # 1x1xn
        self.pool1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False),
            #nn.BatchNorm2d(self.pooled_channels)
        )
        # 2x2xn
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False),
            #nn.BatchNorm2d(self.pooled_channels)
        )
        # 3x3xn
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False),
            #nn.BatchNorm2d(self.pooled_channels)
        )
        # 6x6xn
        self.pool6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False),
            #nn.BatchNorm2d(self.pooled_channels)
        )

    def forward(self, feature):
        # pooling and upsampling
        h,w = feature.size(2), feature.size(3)
        out1 = F.interpolate(self.pool1(feature), size=(h,w), mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.pool2(feature), size=(h,w), mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.pool3(feature), size=(h,w), mode='bilinear', align_corners=True)
        out6 = F.interpolate(self.pool6(feature), size=(h,w), mode='bilinear', align_corners=True)

        # concat
        out = torch.cat((feature, out1,out2,out3,out6),1)

        return out

class PSPNet(nn.Module):
    def __init__(
        self, args, num_classes 
    ):
        super(PSPNet,self).__init__()
        self.encoder = resnet50(pretrained = args.pretrained, dilation =args.dilation)
        self.pp = PPModule(channels=512*4)
        self.final_conv = nn.Sequential(
            nn.Conv2d(4096, 1024,kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, num_classes,kernel_size=1)
        )

    def forward(self,x):
        h,w = x.size(2),x.size(3)
        out = self.encoder(x)
        out = self.pp(out)
        out = self.final_conv(out)
        out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=True)

        return out

'''
net = PSPNet(num_classes=10)
input = torch.randn(1,3,512,512)
output = net(input)
'''