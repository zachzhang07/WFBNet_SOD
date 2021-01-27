#!/usr/bin/python3
# coding=utf-8
import os
import sys
import datetime

sys.path.insert(0, '../')
sys.dont_write_bytecode = True
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import PGwork
from apex import amp


def newloss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)

    return bce + iou.mean()


def train(Dataset, Network, savepath, params):
    ## dataset
    cfg = Dataset.Config(datapath='../data/DUTS',
                         savepath=savepath,
                         mode='train', batch=16, lr=0.025, momen=0.9,
                         decay=5e-4, epoch=34)
    data = Dataset.Data(cfg)

    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=0)
    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    sw = SummaryWriter(cfg.savepath)
    global_step = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()

            out_refine1, out_refine2, out5, out4, out3, out2 = net(image)
            out_refine1_loss = newloss(out_refine1, mask) * params['refine1_ratio']
            out_refine2_loss = newloss(out_refine2, mask) * params['refine2_ratio']
            out2_loss = newloss(out2, mask) * params['out2_ratio']
            out5_loss = newloss(out5, mask) * params['out5_ratio']
            out3_loss = newloss(out3, mask) * params['out3_ratio']
            out4_loss = newloss(out4, mask) * params['out4_ratio']
            loss = out_refine1_loss + out_refine2_loss + out2_loss + out5_loss + out3_loss + out4_loss

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()

            # log
            global_step += 1
            sw.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'out_refine1_loss': out_refine1_loss.item(),
                                    'out_refine2_loss': out_refine2_loss.item(),
                                    'loss2': out2_loss.item(), 'loss3': out3_loss.item(),
                                    'loss4': out4_loss.item(), 'loss5': out5_loss.item()},
                           global_step=global_step)

            if step % 50 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss.item()))

        if epoch > cfg.epoch - 4:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


if __name__ == '__main__':
    savepath = './out_' + datetime.datetime.now().strftime("%Y%m%d_%H%M")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    shutil.copyfile('train.py', savepath + '/train.py')
    shutil.copyfile('net.py', savepath + '/net.py')
    shutil.copyfile('dataset.py', savepath + '/dataset.py')
    params = {'refine1_ratio': 0.5, 'refine2_ratio': 0.5, 'out2_ratio': 0.5, 'out5_ratio': 0.5, 'out4_ratio': 0.125,
              'out3_ratio': 0.25}
    train(dataset, PGwork, savepath, params)
