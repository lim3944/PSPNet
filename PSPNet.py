import torch
import torch.nn as nn
import torch.nn.functional as F

import ResNet

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
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False)
        )
        # 2x2xn
        self.pool2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False)
        )
        # 3x3xn
        self.pool3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False)
        )
        # 6x6xn
        self.pool6 = nn.Sequential(
            nn.AdaptiveAvgPool2d(6),
            nn.Conv2d(self.channels, self.pooled_channels,kernel_size=1,bias=False)
        )

    def forward(self, feature):
        # pooling and upsampling
        h,w = feature.size(2), feature.size(3)
        out1 = F.interpolate(self.pool1(feature), size=(h,w), mode='bilinear')
        out2 = F.interpolate(self.pool2(feature), size=(h,w), mode='bilinear')
        out3 = F.interpolate(self.pool3(feature), size=(h,w), mode='bilinear')
        out6 = F.interpolate(self.pool6(feature), size=(h,w), mode='bilinear')

        # concat
        out = torch.cat((feature, out1,out2,out3,out6),1)

        return out

class PSPNet(nn.Module):
    def __init__(
        self, num_classes 
    ):
        super(PSPNet,self).__init__()
        self.encoder = ResNet.resnet50()
        self.pp = PPModule(channels=512*4)
        self.drop = nn.Dropout2d(p=0.3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(4096, 1024,kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(1024, num_classes, kernel_size=3)
        )

    def forward(self,x):
        h,w = x.size(2),x.size(3)
        out = self.encoder(x)
        out = self.pp(out)
        out = self.dropout(out)
        out = self.final_conv(out)

        out = F.interpolate(out, size=(h,w), mode='bilinear',align_corners=False)

        return out

'''
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.upsample(input=x, size=(h, w), mode='bilinear')
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes=18, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50',
                 pretrained=True):
        super().__init__()
        self.encoder = ResNet.resnet50()
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            #nn.LogSoftmax()
        )

    def forward(self, x):
        f = self.feats(x) 
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)
'''