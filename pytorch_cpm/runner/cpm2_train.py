import os
import sys
from pathlib import Path
import time
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tensorboardX import SummaryWriter
# import pytorch_lightning as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.face_cpm_model import FaceNet
from dataset.landmark_dataset import FaceDataset
from dataset.lsp_lspet_data import LSP_Data
from utils.evaluation import decode_preds, compute_nme

# os.chdir('H:\Project\Github\hrnet_facial_landmark')


def parse_args(cmds): 
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('--input', '-i', help='image file path')
    parser.add_argument('--valid', '-v', help='image file path')
    parser.add_argument('--pretrained', help='pretrained weights file path')
    parser.add_argument('--weights', help='checkpoint weights file path')
    parser.add_argument('--output', '-o', help='save output name')

    parser.add_argument('--image-root', help='image file root path')
    parser.add_argument('--num-class', default=68, type=int)

    parser.add_argument('--epochs', type=int, default=50)  
    parser.add_argument('--batch-size', type=int, default=4)   
    parser.add_argument('--accumulate', type=int, default=1)  
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9)   
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0001)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--lr-step-size', default=15, type=int)

    parser.add_argument('--save-interval', type=int, default=10, help='save checkpoint after interval epoch')
    parser.add_argument('--print-freq', '-p', default=10, type=int, help='print frequency (default: 10)')

    return parser.parse_args(cmds)

def compute_loss(outputs, target):
    """
    Parameters
    ----------
    outputs 预测的 [heatmap]
    target  真实的 heatmap
    """
    total_loss = 0
    criterion = nn.MSELoss(reduction='sum').cuda()
    for heatmaps_y in outputs:
        total_loss += criterion(heatmaps_y, target)
    return total_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def getLr_scheduler(optimizer, LR_STEP, gamma, last_epoch=0):
    if isinstance(LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, LR_STEP, gamma, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, LR_STEP, gamma, last_epoch-1
        )


def datasetFactory(args):
    train_data = FaceDataset(args.input, image_root=args.image_root, is_train=True)
    valid_data = FaceDataset(args.valid, image_root=args.image_root, is_train=False)
    # print(train_data[0][0].shape)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True)

    val_loader = DataLoader(
        dataset=valid_data,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    return train_loader, val_loader

class Trainer:
    def __init__(self, output=None):
        if output is None:
            _timestamp = time.strftime("cpm_%Y%m%d_%H%M%S", time.localtime())
            output_dir = "output/" + _timestamp
        else:
            output_dir = os.path.dirname(output)

        self.o_dir = Path(output_dir)
        os.makedirs(self.o_dir, exist_ok=True)
        self.tb_log_dir = "logs/" + _timestamp

        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.writer.train_i = 0
        self.writer.valid_i = 0

        self.start_epoch = 0
        self.best_nme = 1e8
        self._e = 0

    def save_checkpoint(self, ckpt={}, is_best=False):
        ckpt2 = {
            "is_best": is_best,
            "optimizer": self.optimizer.state_dict(),
            "state_dict": self.model.state_dict()
        }
        ckpt2.update(ckpt)
        # torch.save(ckpt2, self.o_dir / 'last.pth')
        if is_best:
            torch.save(ckpt2, self.o_dir / 'best.pth'.format(self._e))
        else:
            torch.save(ckpt2, self.o_dir / 'checkpoint_{}.pth'.format(self._e))

    def load_checkpoint(self, checkpoint_file):
        if checkpoint_file and os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = checkpoint['epoch']
            self.best_nme = checkpoint['best_nme']

            self.model.load_state_dict(checkpoint['state_dict'].state_dict())
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found", checkpoint_file)

    def train(self, train_loader, val_loader, args):
        best_nme = self.best_nme

        writer = self.writer
        end = time.time()
        output_size = val_loader.dataset.output_size

        for _e in range(self.start_epoch, args.epochs):
            self._e = _e
            self.model.train()
            lossAm = AverageMeter()
            nmeAm = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            for j, (img, target, _) in enumerate(train_loader):
                data_time.update(time.time() - end)
                # print(img.shape)
                preds = self.model(img.cuda())
                loss = compute_loss(preds, target.cuda())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=200, norm_type=2)
                if (j+1) % args.accumulate == 0:
                    self.optimizer.step() 
                    self.optimizer.zero_grad()

                lossAm.update(loss, img.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                if j % 100 == 0: 
                    print(_e,'train', j, loss.item())
                    writer.train_i += 1
                    writer.add_scalar('Train/Loss', loss.item(), writer.train_i)

            writer.add_scalar('Train/Epoch', _e, writer.train_i)       
            writer.add_scalar('Train/Lr', self.optimizer.state_dict()['param_groups'][0]['lr'], writer.train_i)
            val_lossAm = AverageMeter()

            self.model.eval()
            with torch.no_grad():
                for j, (img, target, meta) in enumerate(val_loader):    
                    preds = self.model(img.cuda())
                    val_loss = compute_loss(preds, target.cuda())

                    kpts = decode_preds(preds[-1], meta['center'], meta['scale'], output_size)
                    nmes = compute_nme(kpts, meta['pts'])
                    # print(j, nmes, nmes.shape, img.shape[0])
                    val_lossAm.update(val_loss, img.shape[0])
                    nme = nmes.mean()
                    nmeAm.update(nmes, img.shape[0])              
                    if j % 100 == 0: 
                        print(_e,'valid', j,val_loss.item())
                        writer.valid_i += 1
                        writer.add_scalar('Valid/Loss', val_loss.item(), writer.valid_i)
                        writer.add_scalar('Valid/Nme', nme.item(), writer.valid_i)  
                        
            # writer.add_image('Valid/Image', meta[0]["image_path"], writer.valid_i)                  
            print("epoch=", _e, lossAm.avg, val_lossAm.avg, nmeAm.avg, data_time.avg, batch_time.avg)

            if best_nme > nme:
                best_nme = nme
                is_best = True
            else:
                is_best = False

            ckpt = {
                "epoch": _e + 1,
                "nme": nme,
                "loss": val_lossAm.avg,
                "best_nme": best_nme
            }
            if (1 + _e) % args.save_interval == 5:
                self.save_checkpoint(ckpt=ckpt, is_best=True)
            if is_best:
                self.save_checkpoint(ckpt=ckpt, is_best=True)
                
            self.lr_scheduler.step(_e)

        torch.save(self.model.state_dict(), 'final_state.pth')
        self.writer.close()

def main(cmds=None):
    args = parse_args(cmds)
    print(args)

    LR = args.lr # 1e-4 # 2e-2
    step_size = args.lr_step_size
    pretrained=args.pretrained

    train_loader, val_loader = datasetFactory(args)

    model = FaceNet(num_class=args.num_class)
    model.init_weights()
   
    if pretrained and os.path.exists(pretrained):
        model.load_weights(pretrained, 30)
    model = model.cuda()

    params = [
        {'params': [p for n, p in model.named_parameters() if 'stage1.weight' in n], 'lr': 5 * LR},
        {'params': [p for n, p in model.named_parameters() if 'stage1.bias' in n], 'lr': 10 * LR},
        {'params': [p for n, p in model.named_parameters() if 'weight' in n and 'stage1' not in n], 'lr': LR},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n and 'stage1' not in n], 'lr': 2 * LR}
    ]
    # params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(params, lr=LR, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)

    # criterion = torch.nn.MSELoss().cuda()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=0.1
    )

    tr = Trainer(output=args.output)
    tr.model = model
    tr.optimizer = optimizer
    tr.lr_scheduler = lr_scheduler
    tr.load_checkpoint(args.weights)
    tr.train(train_loader, val_loader, args)

if __name__ == "__main__":
    main()