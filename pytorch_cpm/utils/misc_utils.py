import sys
import os
import os.path as osp


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res


class __Dict2Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def dictMerge(dct, dct_low):
    """
        merge two dict, first dict has high priority, last dict has low priority

        dct:        default item, high priority
        dct_low:    low priority

        example:
            dictMerge({"1":1, "2":2}, {"1":0, "3":3}) = {"1":1, "2":2, "3":3}
    """
    dct_low.update({k: v for k, v in dct.items() if v is not None})
    return dct_low


def namespaceMergeDict(args, cfg):
    """
        args: NameSpace     ,default item, high priority
        cfg: dict           ,low priority
    """
    opt = vars(args)
    for k, v in cfg.items():
        if k in opt.keys() and opt[k] is not None:
            continue
        setattr(args, k, v)
    return args


def namespaceMerge(args, args_low):
    """
        args: NameSpace     ,default item, high priority
        args_low: NameSpace           ,low priority

        namespaceMerge(ns(1=1, 2=2), ns(1=0, 3=3)) = {"1":1, "2":2, "3":3}
    """
    opt = vars(args)
    opt_low = vars(args_low)
    for k, v in opt_low.items():
        if k in opt.keys() and opt[k] is not None:
            continue
        setattr(args, k, v)
    return args


def namespaceMergeTest():
    from argparse import Namespace as ns
    assert namespaceMerge(ns(a=1,b=2), ns(a=0, c=3))==ns(a=1, b=2, c=3)
    assert namespaceMerge(ns(a=None,b=2), ns(a=0, c=3))==ns(a=0, b=2, c=3)
    assert namespaceMerge(ns(a=None,b=2), ns(c=3))==ns(a=None, b=2, c=3)


def pathSplit(x):
    ff = os.path.splitext(x)
    f2 = os.path.split(ff[0])
    return [f2[0], f2[-1], ff[-1]]


def dictIndex2list(d):
    d2 = [(k, v) for k, v in d.items()]
    d2.sort(key=lambda x: x[1])
    return [s[0] for s in d2]

def dictIndex2listTest():
    d = {'cat': 0, 'dog': 1}
    d2 = dictIndex2list(d)
    assert d2 == ['cat', 'dog']


def activateCondaEnv(pth=None):
    if pth is None:
        pth = sys.executable
    base = os.path.dirname(os.path.abspath(pth))
    if os.name == "nt":
        lst = [os.path.join(base, r"Lib\site-packages\torch\lib"),
            os.path.join(base, r"Library\mingw-w64\bin"),
            os.path.join(base, r"Library\usr\bin"),
            os.path.join(base, r"Library\bin"),
            os.path.join(base, r"Scripts"),
            os.path.join(base, r"bin"),
            base,
            os.environ.get('path')]
    else:
        lst =  [os.path.join(base, r"../lib/site-packages/torch/lib"),
            base, 
            os.environ.get('path')]
    os.environ['path'] = ';'.join(lst)