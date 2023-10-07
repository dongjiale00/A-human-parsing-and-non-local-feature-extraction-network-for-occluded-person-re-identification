from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools
from .ms_non_local_block import MSPyramidAttentionContextModule
import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, cfg, last_stride=2, block=Bottleneck, layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)   # add missed relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=last_stride)

        self.cls_head = nn.Sequential(
            nn.Conv2d(
                in_channels=2048,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.bigG = cfg.MODEL.IF_BIGG
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.part_num = cfg.CLUSTERING.PART_NUM
        self.part_cls_layer = nn.Conv2d(in_channels=256,
                                        out_channels=self.part_num,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        self.context_l4_1 = MSPyramidAttentionContextModule(in_channels=2048, out_channels=2048, c1=2048 // 2,
                                                            c2=2048 // 2,
                                                            dropout=0, fusion="+", sizes=([1]),
                                                            if_gc=0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, part_map):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.relu(x)    # add missed relu
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        print(x.shape)
        x=F.upsample(x, size=(64, 32), mode='bilinear', align_corners=True)
        print(x.shape)
        x = self.context_l4_1(x)

        if self.bigG:
            y_g = self.gap(x)
        x = self.cls_head(x)

        N, f_h, f_w = x.size(0), x.size(2), x.size(3)
        part_cls_score = self.part_cls_layer(x)

        part_pred = F.softmax(part_cls_score, dim=1)

        y_part = []
        for p in range(1, self.part_num):
            y_part.append(self.gap(x * part_pred[:, p, :, :].view(N, 1, f_h, f_w)))
        y_part = torch.cat(y_part, 1)
        # print("normal")
        if not self.bigG:
            y_g = self.gap(x)  # full
        y_fore = self.gap(x * torch.sum(part_pred[:, 1:self.part_num, :, :], 1).view(N, 1, f_h, f_w))  # foreground

        # print(y_part.shape)

        return y_part, y_g, y_fore, x, part_cls_score

    def random_init(self):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_param(self, pretrained_path):
        pretrained_dict = torch.load(pretrained_path)
        logger.info('=> loading pretrained model {}'.format(pretrained_path))
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            logger.info(
                '=> loading {} pretrained model {}'.format(k, pretrained_path))
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)










