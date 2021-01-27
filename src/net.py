#!/usr/bin/python3
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU) or isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.Sigmoid):
            pass
        else:
            m.initialize()


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out + x, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 23, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                   nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('../res/resnet101-5d3b4d8f.pth'), strict=False)


class RefineBlock(nn.Module):
    def __init__(self, channel):
        super(RefineBlock, self).__init__()
        self.conv1_l = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv1_b = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv3_l = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv3_b = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv4_l = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv4_b = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv5_l = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.BatchNorm2d(channel))
        self.conv5_b = nn.Sequential(nn.Conv2d(channel * 2, channel, 3, 1, 1), nn.BatchNorm2d(channel))

        self.conv1_res = nn.Sequential(nn.Conv2d(channel * 2 + 1, 1, 1), nn.BatchNorm2d(1))
        self.conv1_mask = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1), nn.BatchNorm2d(1))

    def forward(self, left, bottom, last_res):
        last_res = F.interpolate(last_res, left.shape[2:], mode='bilinear')
        mask = torch.sigmoid(last_res)
        mask = 4 * torch.mul(mask, 1 - mask)
        mask = F.relu(self.conv1_mask(4 * torch.mul(mask, 1 - mask)))

        left = F.relu(self.conv1_l(left))
        bottom = F.relu(self.conv1_b(last_res + F.interpolate(bottom, left.shape[2:], mode='bilinear')))

        common = torch.mul(left, bottom)
        left = F.relu(self.conv4_l(left + F.relu(self.conv3_l(common))))
        bottom = F.relu(self.conv4_b(bottom + F.relu(self.conv3_b(common))))

        left = F.relu(self.conv5_l(torch.cat((torch.mul(mask, left), left), dim=1)))
        bottom = F.relu(self.conv5_b(torch.cat((torch.mul(mask, bottom), bottom), dim=1)))
        last_res = F.relu(self.conv1_res(torch.cat((last_res, left, bottom), dim=1)))

        return left, bottom, last_res

    def initialize(self):
        weight_init(self)


class RefineFlow(nn.Module):
    def __init__(self, channel):
        super(RefineFlow, self).__init__()
        self.block4 = RefineBlock(channel=channel)
        self.block3 = RefineBlock(channel=channel)
        self.block2 = RefineBlock(channel=channel)

    def forward(self, c5, c4, c3, c2, last_res):
        c4, bottom, last_res = self.block4(c4, c5, last_res)
        c3, bottom, last_res = self.block3(c3, bottom, last_res)
        c2, bottom, last_res = self.block2(c2, bottom, last_res)

        return c4, c3, c2, bottom

    def initialize(self):
        weight_init(self)


class PGwork(nn.Module):
    def __init__(self, cfg, channel=128):
        super(PGwork, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()
        # self.bkbone = ResNet101()
        self.reduce5 = nn.Sequential(nn.Conv2d(2048, channel, 1), nn.BatchNorm2d(channel))
        self.reduce4 = nn.Sequential(nn.Conv2d(1024, channel, 1), nn.BatchNorm2d(channel))
        self.reduce3 = nn.Sequential(nn.Conv2d(512, channel, 1), nn.BatchNorm2d(channel))
        self.reduce2 = nn.Sequential(nn.Conv2d(256, channel, 1), nn.BatchNorm2d(channel))

        self.refineflow1 = RefineFlow(channel=channel)
        self.refineflow2 = RefineFlow(channel=channel)

        self.score5 = nn.Sequential(nn.Conv2d(2048, 512, 3, 1, 1, dilation=2),
                                    nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 128, 3, 1, 1, dilation=2),
                                    nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 1, 3, 1, 1, dilation=2))
        self.score4 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.score3 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.score2 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.score_refine1 = nn.Conv2d(channel, 1, 3, 1, 1)
        self.score_refine2 = nn.Conv2d(channel, 1, 3, 1, 1)

        self.initialize()

    def forward(self, x):
        c2, c3, c4, c5 = self.bkbone(x)
        result5 = self.score5(c5)
        c5, c4, c3, c2 = F.relu(self.reduce5(c5)), F.relu(self.reduce4(c4)), \
                         F.relu(self.reduce3(c3)), F.relu(self.reduce2(c2))
        c4, c3, c2, bottom = self.refineflow1(c5, c4, c3, c2, result5)
        result_refine1 = self.score_refine1(bottom)
        c4, c3, c2, bottom = self.refineflow2(c5, c4, c3, c2, result_refine1)
        result_refine2 = self.score_refine2(bottom)

        result4 = self.score4(c4)
        result3 = self.score3(c3)
        result2 = self.score2(c2)

        if result_refine1.shape != x.shape:
            result_refine1 = F.interpolate(result_refine1, x.shape[2:], mode='bilinear')
        if result_refine2.shape != x.shape:
            result_refine2 = F.interpolate(result_refine2, x.shape[2:], mode='bilinear')
        if result5.shape != x.shape:
            result5 = F.interpolate(result5, x.shape[2:], mode='bilinear')
        if result4.shape != x.shape:
            result4 = F.interpolate(result4, x.shape[2:], mode='bilinear')
        if result3.shape != x.shape:
            result3 = F.interpolate(result3, x.shape[2:], mode='bilinear')
        if result2.shape != x.shape:
            result2 = F.interpolate(result2, x.shape[2:], mode='bilinear')

        return result_refine1, result_refine2, result5, result4, result3, result2

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
