#!/usr/bin/python3
# coding=utf-8
import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import cv2
import matplotlib.pyplot as plt

plt.ion()
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import dataset
from net import PGwork


class Test(object):
    def __init__(self, Dataset, Network, path, index='33'):
        ## dataset
        self.index = index
        self.cfg = Dataset.Config(datapath=path, snapshot='./out_20200605_2108/model-' + index, mode='test')
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=0)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for image, mask, shape, name in self.loader:
                image = image.cuda().float()
                out_refine1, out_refine2, out5, out4, out3, out2 = self.net(image)
                out = F.interpolate(out_refine2, shape, mode='bilinear')

                pred = (torch.sigmoid(out[0, 0]) * 255).cpu().numpy()
                head = '../eval/maps/WFBNet-' + self.cfg.snapshot.split('/')[-2][-4:] + '-' + self.index + '/' + \
                       self.cfg.datapath.split('/')[-1]
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', pred)


if __name__ == '__main__':
    for path in ['../data/ECSSD', '../data/PASCAL-S', '../data/DUTS', '../data/HKU-IS', '../data/DUT-OMRON']:
        t = Test(dataset, PGwork, path, sys.argv[1])
        t.save()
