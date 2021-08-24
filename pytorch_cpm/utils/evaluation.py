# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from .transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor

    return type: torch.LongTensor

    Parameters
    ----------
    scores: shape=[N,C,H,W], ctype=float, 典型shape为[1,68,64,64]


    Returns
    -------
    preds：shape=[N,C,2], ctype=uint，典型shape为[1,68,2]
    
    Notes
    -----
        1. 首先对张量N×C×H×W合并HW维度，编程N×C×HW
        2. 计算 argmax，得到最大值和索引
        3. 把索引从1d转换成2d形式
        4. 对索引基于最大值执行过滤：大于0的保留，小于0的置零
        5. 返回预测索引
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3) + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3)) + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme0(preds, meta):
    """
        所有预测关键点和groundtruth关键点之间的L2 Norm，然后除以（关键点的个数*两只眼睛之间的距离）

        interocular: 两只眼睛之间的距离
    """
    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def compute_nme(preds, targets):
    """
        preds: shape = [N, C, 2]
        所有预测关键点和groundtruth关键点之间的L2 Norm，然后除以关键点的个数，除以人脸半径

        这里把人脸简单地当成矩形，简单计算对角线的一半作为人脸半径 。有的方法是把人眼距离作为半径
    """
    # targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    
    for i in range(N):
        
        pts_pred, pts_gt = preds[i, ], target[i, ]
        w = pts_gt[:,0].max()-pts_gt[:,0].min()
        h = pts_gt[:,1].max()-pts_gt[:,1].min()
        radius = np.sqrt(w*w+h*h)/2
        # print("radius", radius)
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / L/radius
    return rmse

def decode_preds(output, center, scale, res):
    """
    对预测结果执行后处理，得到关键点
    
    1. 计算索引，预测输出图是64×64，
    2. 微调索引，切割图是256×256，需要微调来增加0.25 pixel的精度
    3. 对关键点执行反切割变换，从切割图恢复到原图

    Parameters
    ----------
    output: 预测输出结果，score_map，shape=[N,C,H,W], 典型值[2,68,256,256]
    center：中心点（切图相关）
    scale：尺度（切图相关）
    res: output的宽高shape, 典型值 [64, 64]

    Return
    ----------
    preds: shape=[N, C, 2]
    """
    coords = get_preds(output)  # float type

    coords = coords.cpu()
    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if (px > 1) and (px < res[0]) and (py > 1) and (py < res[1]):
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds
