import os
import sys
from pathlib import Path
import time
import argparse

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tensorboardX import SummaryWriter
# import pytorch_lightning as pl
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from pytorch_cpm.models.face_cpm_model import FaceNet
import pytorch_cpm.utils.Mytransforms as Mytransforms


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
    parser.add_argument('--dataset-type', type=str.upper, choices=["LSP", "COCO", "CSV"], help='dataset type')

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
    transformer = Mytransforms.Compose([Mytransforms.RandomResized(),
            Mytransforms.RandomRotate(40),
            Mytransforms.RandomCrop(368),
            Mytransforms.RandomHSV((0.8, 1.2), (0.8, 1.2), (25, 25)),
            # Mytransforms.RandomHorizontalFlip(),
        ])

    if args.dataset_type == "LSP":
        from pytorch_cpm.dataset.lsp_lspet_data import LSP_Data
        mode = 'lspet' if 'lspet' in args.input else 'lsp'
        train_data = LSP_Data(mode, args.input, stride=8, transformer=transformer)
        if args.valid:
            mode = 'lspet' if 'lspet' in args.val_dir else 'lsp'
            valid_data = LSP_Data(mode, args.input, stride=8, 
                transformer=Mytransforms.Compose([Mytransforms.TestResized(368)]))
    elif args.dataset_type == "CSV":
        from pytorch_cpm.dataset.csv_keypoint_dataset import CsvDataset
        train_data = CsvDataset(args.input, image_root=args.image_root, transformer=transformer)
        if args.valid:
            valid_data = CsvDataset(args.valid, image_root=args.image_root,
                transformer=Mytransforms.Compose([Mytransforms.TestResized(368)]))
    else:
        from pytorch_cpm.dataset.landmark_dataset import FaceDataset
        train_data = FaceDataset(args.input, image_root=args.image_root, is_train=True)
        if args.valid:
            valid_data = FaceDataset(args.valid, image_root=args.image_root, is_train=False)

    if args.valid is None:
        from torch.utils.data import random_split
        num_valid = max(1, int(len(train_data) * 0.1))
        print("splite data to valid", num_valid)
        train_data, valid_data = random_split(train_data, [len(train_data) - num_valid, num_valid])

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
            output_name = time.strftime("cpm_%Y%m%d_%H%M%S", time.localtime())
            output_dir = "output/" + output_name
        else:
            output_dir = os.path.dirname(output)
            output_name = os.path.splitext(os.path.basename(output))[0]

        self.o_dir = Path(output_dir)
        os.makedirs(self.o_dir, exist_ok=True)
        self.tb_log_dir = "logs/" + output_name

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

            self.model.load_state_dict(checkpoint['state_dict']) # .state_dict()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found", checkpoint_file)

    def train_epoch(self, train_loader, args):
        writer = self.writer
        _e = self._e
        self.model.train()
        lossAm = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        tic = time.time()
        for j, (img, target, _) in enumerate(train_loader):
            data_time.update(time.time() - tic)
            # print(img.shape)
            preds = self.model(img.cuda())

            loss = sum([self.criterion(hm_y, target.cuda()) for hm_y in preds])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
            if (j+1) % args.accumulate == 0:
                self.optimizer.step() 
                self.optimizer.zero_grad()

            lossAm.update(loss, img.shape[0])
            batch_time.update(time.time() - tic)
            tic = time.time()
            if j % 100 == 0: 
                print(_e,'train', j, loss.item())
                writer.train_i += 1
                writer.add_scalar('Train/Loss', loss.item(), writer.train_i)

        writer.add_scalar('Train/Epoch', _e, writer.train_i)       
        writer.add_scalar('Train/Lr', self.optimizer.state_dict()['param_groups'][0]['lr'], writer.train_i)
        print("epoch=", _e, lossAm.avg, data_time.avg, batch_time.avg)

    def valid_epoch(self, val_loader, args):
        writer = self.writer
        _e = self._e
        val_lossAm = AverageMeter()
        nmeAm = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for j, (img, target, meta) in enumerate(val_loader):    
                preds = self.model(img.cuda())

                val_loss = sum([self.criterion(hm_y, target.cuda()) for hm_y in preds])
                val_lossAm.update(val_loss, img.shape[0])
                # from pytorch_cpm.utils.evaluation import decode_preds, compute_nme
                    
                # kpts = decode_preds(preds[-1], meta['center'], meta['scale'], output_size)
                # nmes = compute_nme(kpts, meta['pts'])
                # # print(j, nmes, nmes.shape, img.shape[0])
                # nme = nmes.mean()
                nme = 0
                # nmeAm.update(nmes, img.shape[0])  

                if j % 100 == 0: 
                    print(_e,'valid', j,val_loss.item())
                    writer.valid_i += 1
                    writer.add_scalar('Valid/Loss', val_loss.item(), writer.valid_i)
                    # writer.add_scalar('Valid/Nme', nme.item(), writer.valid_i)  
                    
        # writer.add_image('Valid/Image', meta[0]["image_path"], writer.valid_i)                  
        print("epoch=", _e, val_lossAm.avg, nmeAm.avg)
        return {
            "epoch": _e + 1,
            "nme": nme,
            "loss": val_lossAm.avg
        }

    def train(self, train_loader, val_loader, args):
        writer = self.writer
        tic = time.time()
        # output_size = val_loader.dataset.output_size

        for _e in range(self.start_epoch, args.epochs):
            self._e = _e
            self.train_epoch(train_loader, args)
            ckpt = self.valid_epoch(val_loader, args)
            nme = ckpt["nme"]
            if self.best_nme > nme:
                self.best_nme = nme
                is_best = True
                ckpt["best_nme"] = self.best_nme
                self.save_checkpoint(ckpt=ckpt, is_best=True)
            else:
                is_best = False
            if (1 + _e) % args.save_interval == 5:
                self.save_checkpoint(ckpt=ckpt, is_best=False)

            self.lr_scheduler.step(_e)
        torch.save(self.model.state_dict(), self.o_dir / 'final_state.pth')
        self.writer.close()

def main(cmds=None):
    args = parse_args(cmds)
    print(args)

    LR = args.lr # 1e-4 # 2e-2
    step_size = args.lr_step_size
    pretrained=args.pretrained

    train_loader, val_loader = datasetFactory(args)

    model = FaceNet(num_class=args.num_class+1)
    model.init_weights()
   
    if pretrained and os.path.exists(pretrained):
        model.load_weights(pretrained, 30)
    model = model.cuda()

    _params = [
        {'params': [p for n, p in model.named_parameters() if 'stage1.weight' in n], 'lr': 5 * LR},
        {'params': [p for n, p in model.named_parameters() if 'stage1.bias' in n], 'lr': 10 * LR},
        {'params': [p for n, p in model.named_parameters() if 'weight' in n and 'stage1' not in n], 'lr': LR},
        {'params': [p for n, p in model.named_parameters() if 'bias' in n and 'stage1' not in n], 'lr': 2 * LR}
    ]
    # _params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(_params, lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(_params, lr=LR, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)

    # criterion = torch.nn.MSELoss().cuda()
    tr = Trainer(output=args.output)
    tr.model = model
    tr.optimizer = optimizer
    tr.load_checkpoint(args.weights)
    tr.criterion = nn.MSELoss(reduction='sum').cuda()
    tr.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, last_epoch=tr.start_epoch-1, step_size=step_size, gamma=0.1
    )
    tr.train(train_loader, val_loader, args)

if __name__ == "__main__":
    main()