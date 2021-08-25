# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random
import sys

import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import cv2

from pytorch_cpm.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel


class FaceDataset(data.Dataset):
    def __init__(self, csv_file, image_root=None, is_train=True, transform=None):
        # specify annotation file for dataset
        self.csv_file = csv_file
        self.image_root = image_root

        self.is_train = is_train
        self.transform = transform
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = [368, 368]
        self.output_size = [46, 46]
        self.sigma = 1.0
        self.scale_factor = 0.15 ###
        # self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = 'Gaussian'
        self.flip = False

        # load annotations
        self.landmarks_frame = pd.read_csv(self.csv_file)

        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([255, 255, 255], dtype=np.float32)

    def __len__(self):
        return len(self.landmarks_frame)

    def rect2center(self, rct):
        x,y,w,h = (rct[0]+rct[2])//2, (rct[1]+rct[3])//2, (rct[2]-rct[0]), (rct[3]-rct[1])
        wh2scale = lambda w, h: math.sqrt(w * h) * 1.2 / 200
        scale = wh2scale(w, h)
        center = torch.Tensor([max(x, 0), max(y, 0)])
        return center, scale
    
    def __getitem__(self, idx):

        """
        1. random fliplr
        2. crop and rotate
        3. gen Heatmap
        
        Returns
        -------
        img: shape=[C,H,W]
        heatmaps: shape=[N,H2,W2]，典型shape为[68, 46, 46]
        pts: shape=[N,2]， 对应原图的pts。
        tpts: shape=[N,2], 对应crop的pts。

        """
        image_path = self.landmarks_frame.iloc[idx, -1]
        if self.image_root:
            image_path = os.path.join(self.image_root, image_path)

        center, scale = self.rect2center(self.landmarks_frame.iloc[idx, -5:-1])

        pts = self.landmarks_frame.iloc[idx, 0:-5].values
        pts = pts.astype('float').reshape(-1, 2)
        
        ### scale *= 1.25

        # print(image_path, end=' ')
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        # cv2.imshow('aa', img.astype(np.uint8))
        # cv2.waitKey(0)
        # return img, pts

        rot = 0
        if self.is_train:
            scale *=  random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
            # rot = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img, pts, center = self.fliplr(img, pts, center)

        # if self.transform:
        #     pass

        img, tpts = self.crop(img, pts, center, scale, rot)
        img = self.preprocess_img(img)
        heatmaps = self.preprocess_pts(tpts)

        meta = {'index': idx, 'center': torch.Tensor(center), 'scale': scale,
                'pts': torch.Tensor(pts), 'tpts': torch.Tensor(tpts), 'image_path': image_path}

        return img, heatmaps, meta


    def crop(self, img, pts, center, scale, rot=0):
        nparts = pts.shape[0]
        img = crop(img, center, scale, self.input_size, rot=rot)

        tpts = pts.copy()
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                               scale, self.output_size, rot=rot)
        return img, tpts

    def fliplr(self, img, pts, center):
        img = np.fliplr(img)
        pts = fliplr_joints(pts, width=img.shape[1], dataset='300W')
        center[0] = img.shape[1] - center[0]
        return img, pts, center

    def preprocess_img(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        img = torch.Tensor(img)
        return img

    def preprocess_pts(self, pts):
        nparts = pts.shape[0]
        heatmaps = np.zeros((nparts, self.output_size[0], self.output_size[1]))
        for i in range(nparts):
            heatmaps[i] = generate_target(heatmaps[i], pts[i]-1, self.sigma, label_type=self.label_type)

        heatmaps = torch.Tensor(heatmaps)
        return heatmaps

    def origin(self, idx, is_crop=None, is_show=None):
        image_path = self.landmarks_frame.iloc[idx, -1]
        if self.image_root:
            image_path = os.path.join(self.image_root, image_path)

        center, scale = self.rect2center(self.landmarks_frame.iloc[idx, -5:-1])

        pts = self.landmarks_frame.iloc[idx, 0:-5].values
        pts = pts.astype('float').reshape(-1, 2)
        
        scale *= 1.25
        # print(image_path, end=' ')
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        if is_crop:
            rot = 0
            scale *=  random.uniform(1 - self.scale_factor, 1 + self.scale_factor)
            img, tpts = self.crop(img, pts, center, scale, rot)
            if is_show:
                tpts2 = (tpts-1) * 368/46+1
                img2 = drawKeypoints(img, tpts2)
                cv2.imshow('aaa', img2.astype(np.uint8))
                cv2.waitKey(0)
            return img, tpts
        else:
            if is_show:
                cv2.imshow('aa', img.astype(np.uint8))
                cv2.waitKey(0)
            return img, pts

def drawKeypoints(img, pts):
    img2 = img.copy()
    for pt in pts:
        x = int(pt[0])
        y = int(pt[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(255, 0, 0))
    return img2

if __name__ == '__main__':
    pass
