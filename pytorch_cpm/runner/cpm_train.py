# -*-coding:UTF-8-*-
import os
import sys
import argparse
import time

import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.cpm_model import CPM
from dataset.lsp_lspet_data import LSP_Data
from dataset.csv_keypoint_dataset import CsvDataset
from utils.utils import Config, adjust_learning_rate, \
    AverageMeter, save_checkpoint
from utils.misc_utils import namespaceMerge
import utils.Mytransforms as Mytransforms

__DIR__ = os.path.dirname(os.path.abspath(__file__))
       

def parse_args(cmds=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(__DIR__, 'config.yaml'), help='to set the parameters')
    parser.add_argument('--train_dir','-i', help='the path of train file')
    parser.add_argument('--val_dir', '-v', help='the path of val file')
    parser.add_argument('--num-class', default=14, type=int)
    parser.add_argument('--image-root')
    parser.add_argument('--dataset-type', type=str.upper, choices=["LSP", "COCO", "CSV"], help='dataset type')
    parser.add_argument('--model-name', default='../ckpt/cpm', help='model name to save parameters')
    parser.add_argument('--pretrained', default='../ckpt/cpm_latest.pth.tar',
                        help='the path of pretrained model')
    
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--accumulate', default=1, type=int)
    parser.add_argument('--learning-rate', '-lr', type=float, dest='base_lr')
    parser.add_argument('--gpu', default=None, nargs='+', type=int, dest='gpu', help='the gpu used')

    return parser.parse_args(cmds)

def construct_model(args):

    model = CPM(k=args.num_class)
    # load pretrained model
    # state_dict = torch.load(args.pretrained)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)

    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def get_parameters(model, config, isdefault=True):
    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    lr_4 = []
    lr_8 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if ('model1_' not in key) and ('model0.' not in key):
            if key[-4:] == 'bias':
                lr_8.append(value)
            else:
                lr_4.append(value)
        elif key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
            {'params': lr_2, 'lr': config.base_lr * 2.},
            {'params': lr_4, 'lr': config.base_lr * 4.},
            {'params': lr_8, 'lr': config.base_lr * 8.}]

    return params, [1., 2., 4., 8.]

def dataFactory(train_dir, val_dir, args):
    # train
    mode = 'lspet' if 'lspet' in train_dir else 'lsp'
    
    transformer = Mytransforms.Compose([Mytransforms.RandomResized(),
        Mytransforms.RandomRotate(40),
        Mytransforms.RandomCrop(368),
        Mytransforms.RandomHSV((0.8, 1.2), (0.8, 1.2), (25, 25)),
        # Mytransforms.RandomHorizontalFlip(),
    ])
    dataset = LSP_Data(mode, train_dir, 8, transformer=transformer)
    # dataset = CsvDataset(train_dir, image_root=args.image_root,transformer=transformer)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, drop_last=True,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
# val
    if args.val_dir:
        mode = 'lspet' if 'lspet' in args.val_dir else 'lsp'
        val_dataset = LSP_Data(mode, args.val_dir, 8,
            Mytransforms.Compose([Mytransforms.TestResized(368)]))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=args.workers, pin_memory=True)
    else:
        val_loader = None

    return train_loader, val_loader

class Trainer:
    def __init__(self):
        """
        # HEAT_WEIGHT = 46 * 46 * (1+args.num_class) / 1.0
        multiple=multiple
        """
        self._iter = 0

    def save_ckpt(self, ckpt={}, is_best=False):
        model_name = self.config.model_name
        if is_best:
            model_name = model_name +"_best.pth"
        _ckpt = {
            'iter': self._iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        _ckpt.update(ckpt)
        save_checkpoint(_ckpt, is_best, model_name)

    def training_step(self, batch, batch_idx):        
        input, heatmap, centermap = batch
        input_var = torch.autograd.Variable(input)
        heatmap_var = torch.autograd.Variable(heatmap.cuda())
        centermap_var = torch.autograd.Variable(centermap.cuda())

        heat_list = self.model(input_var, centermap_var)
        assert torch.isnan(heat_list[0]).sum() == 0, print(loss)
        assert heat_list[0].shape == heatmap_var.shape, print(heat_list[0].shape, heatmap_var.shape)

        loss_list = [self.criterion(heat, heatmap_var) for heat in heat_list]
        # bSize = input.size(0)

        loss = sum(loss_list)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=200, norm_type=2)
        self.optimizer.step()

        return loss_list, input.size(0)

    def test_step(self, batch, batch_idx):
        input, heatmap, centermap = batch

        input_var = torch.autograd.Variable(input)
        heatmap_var = torch.autograd.Variable(heatmap.cuda(async=True))
        centermap_var = torch.autograd.Variable(centermap.cuda(async=True))

        heat_list = self.model(input_var, centermap_var)
        assert heat_list[0].shape == heatmap_var.shape, (heat_list[0].shape, heatmap_var.shape)
        loss_list = [self.criterion(heat, heatmap_var) for heat in heat_list]
        
        return loss_list, input.size(0)

    def train_epoch(self, train_loader):
        multiple = [1., 2., 4., 8.]
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_list = [AverageMeter() for i in range(6)]

        config = self.config
        tic = time.time()
        for i, (batch) in enumerate(train_loader):
            learning_rate = adjust_learning_rate(self.optimizer, self._iter, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - tic)

            loss_list, bSize = self.training_step(batch, i)

            batch_time.update(time.time() - tic)
            tic = time.time()

            loss = sum(loss_list)
            losses.update(loss.item(), bSize)
            for cnt, l in enumerate(loss_list):
                losses_list[cnt].update(l.item(), bSize)

            self._iter += 1
            if self._iter % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\t'
                    'Learning rate = {2}\t'
                    'Loss = {loss.val:.8f} (avg = {loss.avg:.8f})'.format(
                    self._iter, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses), tic='\t')
                print("loss=", [l.val for l in losses_list], "avg=", [l.avg for l in losses_list])
                # print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), '-' * 80)

        batch_time.reset()
        data_time.reset()
        losses.reset()
        for los in losses_list:
            los.reset()


    def valid_epoch(self, val_loader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_list = [AverageMeter() for i in range(6)]

        self.config = config
        tic = time.time()

        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                loss_list, bSize = self.test_step(batch, j)

                loss = sum(loss_list)
                losses.update(loss.item(), bSize)
                for cnt, l in enumerate(loss_list):
                    losses_list[cnt].update(l.item(), bSize)

                batch_time.update(time.time() - tic)
                tic = time.time()

                if j % config.display == 0:
                    print('Test Iteration: {0}\t'
                            'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                            'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\t'
                            'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\t'.format(
                        j, config.display, batch_time=batch_time,
                        data_time=data_time, loss=losses))
                    for cnt in range(6):
                        print('Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})\t'
                                .format(cnt + 1, loss1=losses_list[cnt]))

        # print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime()), '-' * 80)
        
        # batch_time.reset()
        # losses.reset()
        # for cnt in range(6):
        #     losses_list[cnt].reset()
        
    def train(self, train_loader, val_loader, config):
        self.config = config

        tic = time.time()

        self._iter = config.start_iters
        best_loss = config.best_model

        assert config.test_interval != 0

        while self._iter < config.max_iter:
        # for epochs in range(50):
            self.train_epoch(train_loader)
            self.save_ckpt(ckpt={"loss": losses.avg})
            # val
            if config.val_dir and self._iter % config.test_interval == 0:
                self.valid_epoch(val_loader)

                if losses.avg < best_loss:
                    is_best = True
                    best_loss = losses.avg
                    self.save_ckpt(ckpt={"loss": losses.avg}, is_best=is_best)
                else:
                    is_best = False
            self.model.train()
            print("run epochs finished ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print("train finished")

def main():
    args = parse_args()
    config = Config(args.config)
    args = namespaceMerge(args, config)

    model = construct_model(args)
    tr = Trainer()
    train_loader, val_loader = dataFactory(args.train_dir, args.val_dir, args)
    tr.model = model
    tr.criterion = nn.MSELoss(reduction='sum').cuda()

    _params, _multiple = get_parameters(tr.model, config, False)
    tr.optimizer = torch.optim.SGD(_params, args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True
    tr.train(train_loader, val_loader, args)

if __name__ == '__main__':
    cmds = ["--gpu","0","--train_dir","H:/Dataset/keypoint/lsp/lsp_dataset","--config","../config/config.yaml"]
    main()
