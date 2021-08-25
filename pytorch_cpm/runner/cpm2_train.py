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
    parser.add_argument('--weights', help='weights file path')

    parser.add_argument('--num-class', default=68, type=int)
    parser.add_argument('--image-root', help='weights file path')
    parser.add_argument('--epochs', type=int, default=50)  
    parser.add_argument('--batch-size', type=int, default=4)   
    parser.add_argument('--accumulate', type=int, default=1)  
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.001, dest='lr')
    parser.add_argument('--momentum', type=float, default=0.9)   
    parser.add_argument('--weight-decay', '-wd', type=float, default=0.0001)
    parser.add_argument('--output', '-o', default='result.png', help='save output image name')
    parser.add_argument('--workers', default=4, type=int)

    return parser.parse_args(cmds)

def compute_loss(outputs, target):
    """
    Parameters
    ----------
    outputs 预测的 [heatmap]
    target  真实的 heatmap
    """
    total_loss = 0
    criterion = nn.MSELoss(reduction='mean').cuda()
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
    def __init__(self, output_dir=None):
        if output_dir is None:
            _timestamp = time.strftime("cpm_%Y%m%d_%H%M%S", time.localtime())
            output_dir = "output/" + _timestamp
        self.o_dir = Path(output_dir)
        os.makedirs(self.o_dir, exist_ok=True)
        self.tb_log_dir = "logs/" + _timestamp

        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.writer.train_i = 0
        self.writer.valid_i = 0

    def train(self, train_loader, val_loader, args):
        best_nme = 1e8

        writer = self.writer
        end = time.time()
        output_size = val_loader.dataset.output_size

        for epoch in range(args.epochs):
            self.model.train()
            lossAm = AverageMeter()
            nmeAm = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            for j, (img, target, _) in enumerate(train_loader):
                data_time.update(time.time() - end)
                # print(img.shape)
                output = self.model(img.cuda())
                loss = compute_loss(output, target.cuda())

                loss.backward()
                if (j+1) % args.accumulate == 0:
                    self.optimizer.step() 
                    self.optimizer.zero_grad()

                lossAm.update(loss, img.shape[0])
                batch_time.update(time.time() - end)
                end = time.time()
                if j % 100 == 0: 
                    print(epoch,'train', j, loss.item())
                    writer.train_i += 1
                    writer.add_scalar('Train/Loss', loss.item(), writer.train_i)
            
            writer.add_scalar('Train/Epoch', epoch, writer.train_i)       
            writer.add_scalar('Train/Lr', self.optimizer.state_dict()['param_groups'][0]['lr'], writer.train_i)
            val_lossAm = AverageMeter()

            self.model.eval()
            with torch.no_grad():
                for j, (img, target, meta) in enumerate(val_loader):    
                    output = self.model(img.cuda())
                    val_loss = compute_loss(output, target.cuda())

                    preds = decode_preds(output[-1], meta['center'], meta['scale'], output_size)
                    # print(preds, preds.shape)
                    nmes = compute_nme(preds, meta['pts'])
                    # print(j, nmes, nmes.shape, img.shape[0])
                    val_lossAm.update(val_loss, img.shape[0])
                    nme = nmes.mean()
                    nmeAm.update(nmes, img.shape[0])              
                    if j % 100 == 0: 
                        print(epoch,'valid', j,val_loss.item())
                        writer.valid_i += 1
                        writer.add_scalar('Valid/Loss', val_loss.item(), writer.valid_i)
                        writer.add_scalar('Valid/Nme', nme.item(), writer.valid_i)  
                        
            # writer.add_image('Valid/Image', meta[0]["image_path"], writer.valid_i)                  
            print("epoch=", epoch, lossAm.avg, val_lossAm.avg, nmeAm.avg, data_time.avg, batch_time.avg)

            if best_nme > nme:
                best_nme = nme
                is_best = True
            else:
                is_best = False

            ckpt = {
                "epoch": epoch + 1,
                "nme": nme,
                "loss": val_lossAm.avg,
                "optimizer": self.optimizer.state_dict(),
                "state_dict": self.model.state_dict()
            }
            torch.save(ckpt, self.o_dir / 'last.pth')
            if epoch % 10 == 4:
                torch.save(ckpt, self.o_dir / 'checkpoint_{}.pth'.format(epoch))
            if is_best:
                torch.save(ckpt, self.o_dir / 'best.pth'.format(epoch))
    
            self.lr_scheduler.step(epoch)

        torch.save(self.model.state_dict(), 'final_state.pth')
        self.writer.close()

def main(cmds=None):
    args = parse_args(cmds)
    print(args)

    LR = args.lr # 1e-4 # 2e-2
    step_size = 15 # args.lr_step_size
    weights_file=args.weights

    train_loader, val_loader = datasetFactory(args)

    model = FaceNet(num_class=args.num_class)
    model.init_weights()
   
    if weights_file and os.path.exists(weights_file):
        model.load_weights(weights_file, 30)
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

    tr = Trainer()
    tr.model = model
    tr.optimizer = optimizer
    tr.lr_scheduler = lr_scheduler
    tr.train(train_loader, val_loader, args)

if __name__ == "__main__":
    main()