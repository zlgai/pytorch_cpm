# -*-coding:UTF-8-*-
import argparse
import time
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
sys.path.append("..")
from utils.utils import adjust_learning_rate as adjust_learning_rate
from utils.utils import AverageMeter as AverageMeter
from utils.utils import save_checkpoint as save_checkpoint
from utils.utils import Config as Config
import cpm_model
from lsp_lspet_data import LSP_Data
from lsp_lspet_data2 import CsvDataset
import Mytransforms


def parse_args(cmds=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=None, nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default='../ckpt/cpm_latest.pth.tar',type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--train_dir', help='the path of train file')
    parser.add_argument('--val_dir', help='the path of val file')
    parser.add_argument('--model-name', default='../ckpt/cpm', help='model name to save parameters')
    parser.add_argument('--num-class', default=14, type=int)
    parser.add_argument('--image-root')
    return parser.parse_args(cmds)


def construct_model(args):

    model = cpm_model.CPM(k=args.num_class)
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
        dataset=dataset,
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=True)
# val
    if args.val_dir:
        mode = 'lspet' if 'lspet' in val_dir else 'lsp'
        val_loader = torch.utils.data.DataLoader(
            lsp_lspet_data.LSP_Data(mode, val_dir, 8,
                              Mytransforms.Compose([Mytransforms.TestResized(368),
                                                    ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)
    else:
        val_loader = None

    return train_loader, val_loader

class Trainer:
    def __init__(self):
        self._iter = 0

    def save_ckpt(self, model_name, is_best=0):
        save_checkpoint({
            'iter': self._iter,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, is_best, model_name)

        """
        HEAT_WEIGHT
        multiple=multiple
        """
    def training_step(self, batch, batch_idx):        
        input, heatmap, centermap = batch
        input_var = torch.autograd.Variable(input)
        heatmap_var = torch.autograd.Variable(heatmap.cuda())
        centermap_var = torch.autograd.Variable(centermap.cuda())

        heat_list = self.model(input_var, centermap_var)
        assert torch.isnan(heat_list[0]).sum() == 0, print(loss)
        assert heat_list[0].shape == heatmap_var.shape, print(heat_list[0].shape, heatmap_var.shape)

        loss_list = [self.criterion(heat, heatmap_var) * self.HEAT_WEIGHT for heat in heat_list]
        # bSize = input.size(0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=200, norm_type=2)
        self.optimizer.step()

        return loss_list.detach(), input.size(0)

    def test_step(self, batch, batch_idx):
        input, heatmap, centermap = batch

        input_var = torch.autograd.Variable(input)
        heatmap_var = torch.autograd.Variable(heatmap.cuda(async=True))
        centermap_var = torch.autograd.Variable(centermap.cuda(async=True))

        heat_list = self.model(input_var, centermap_var)
        assert heat_list[0].shape == heatmap_var.shape, (heat_list[0].shape, heatmap_var.shape)
        loss_list = [self.criterion(heat, heatmap_var) * self.HEAT_WEIGHT for heat in heat_list]
        
        return loss_list.detach(), input.size(0)

    def train_epoch(self, train_loader):
        multiple = [1., 2., 4., 8.]
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_list = [AverageMeter() for i in range(6)]

        config = self.config

        for i, (batch) in enumerate(train_loader):
            learning_rate = adjust_learning_rate(self.optimizer, self._iter, config.base_lr, policy=config.lr_policy,
                                                policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            loss_list, bSize = self.training_step(batch, i)

            batch_time.update(time.time() - end)
            end = time.time()

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
                    data_time=data_time, loss=losses), end='\t')
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

        model.eval()
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                loss_list, bSize = self.test_step(batch, j)

                loss = sum(loss_list)
                losses.update(loss.item(), bSize)
                for cnt, l in enumerate(loss_list):
                    losses_list[cnt].update(l.item(), bSize)

                batch_time.update(time.time() - end)
                end = time.time()

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
        
    def train(model, train_loader, val_loader, args):
        config = Config(args.config)
        cudnn.benchmark = True

        self.criterion = nn.MSELoss().cuda()

        params, multiple = get_parameters(model, config, False)

        self.optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
                                    weight_decay=config.weight_decay)
        self.model = model
        self.config = config

        end = time.time()

        self._iter = config.start_iters
        best_loss = config.best_model

        self.HEAT_WEIGHT = 46 * 46 * (1+args.num_class) / 1.0

        assert config.test_interval != 0

        while self._iter < config.max_iter:
        # for epochs in range(50):
            self.train_epoch(train_loader)
            self.save_ckpt(args.model_name)
            # val
            if args.val_dir is not None and self._iter % config.test_interval == 0:
                self.valid_epoch(val_loader)
                if is_best:
                    self.save_ckpt(args.model_name+"_best.pth", is_best=losses.avg < best_loss)
                best_loss = min(best_loss, losses.avg)
            model.train()
            print("run epochs finished ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        print("train finished")

def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    model = construct_model(args)
    tr = Trainer()
    
    train_loader, val_loader = dataFactory(args.train_dir, args.val_dir, args)

    tr.train(model, train_loader, val_loader, args)

if __name__ == '__main__':
    cmds = ["--gpu","0","--train_dir","H:/Dataset/keypoint/lsp/lsp_dataset","--config","../config/config.yml"]
    main()

