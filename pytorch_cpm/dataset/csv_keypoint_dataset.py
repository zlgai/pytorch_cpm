# -*-coding:UTF-8-*-
import os
import scipy.io
import numpy as np
import glob
import math
import torch
import torch.utils.data as data
import scipy.misc
from PIL import Image
import cv2
import Mytransforms
import pandas as pd

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


def kpts2heatmap(kpts, in_shape, out_shape, add_background=None):
    """

    Parameters
    ----------
    kpts: shape=[N,2], ctype=float, 典型shape为[68,2]

    Returns
    -------
    heatmap：shape=[N,C,2], ctype=uint，典型shape为[1, 68, 2]

    """
    sigma = 3.0

    if add_background:
        ofs = 1
    else:
        ofs = 0

    heatmap = np.zeros((out_shape[0], out_shape[1], len(kpts) + ofs), dtype=np.float32)
    # height // stride ,width // stride
    for i in range(len(kpts)):
        # resize from 368 to 46
        x = (kpts[i][0]) / in_shape[0] * out_shape[0]
        y = (kpts[i][1]) / in_shape[1] * out_shape[1]
        heat_map = guassian_kernel(size_h=out_shape[0], size_w=out_shape[1], center_x=x, center_y=y, sigma=sigma)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        heatmap[:, :, i + ofs] = heat_map

    if add_background:
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background
    return heatmap


class CsvDataset(data.Dataset):
    """
        Args:
            root_dir (str): the path of train_val dateset.
            stride (float): default = 8
            transformer (Mytransforms): expand dataset.
        Notice:
            you have to change code to fit your own dataset except LSP

    """

    def __init__(self, csv_file, image_root=None, transformer=None ):
        self.csv_file = csv_file
        self.landmarks_frame = pd.read_csv(self.csv_file)
        self.image_root = image_root

        self.transformer = transformer
        self.sigma = 3.0
        self.input_size = [368, 368]
        self.output_size = [46, 46]
        self.mean = [128.0, 128.0, 128.0]
        self.std = [256.0, 256.0, 256.0]

        self.genCentermap()

    def __getitem__(self, idx):
        """

        Returns
        -------
        img         shape=[3, 368, 368]
        heatmap     shape=[15, 46, 46]
        centermap   shape=[1, 368, 368]
        """
        img_path = self.landmarks_frame.iloc[idx, -1]
        if self.image_root:
            img_path = os.path.join(self.image_root, img_path)

        center, scale = self.rect2center(self.landmarks_frame.iloc[idx, -5:-1])
        center = center.numpy()
        pts = self.landmarks_frame.iloc[idx, 0:-5].values
        kpt = pts.astype('float').reshape(-1, 2)
        img = np.array(cv2.imread(img_path), dtype=np.float32)

        # expand dataset
        if self.transformer:
            img, kpt, center = self.transformer(img, kpt, center, scale)

        heatmap = kpts2heatmap(kpt, self.input_size, self.output_size, add_background=True)
        img = Mytransforms.normalize(Mytransforms.to_tensor(img), self.mean, self.std)

        heatmap = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(self.centermap)
        return img, heatmap, centermap

    def __len__(self):
        return len(self.landmarks_frame)

    def rect2center(self, rct):
        x,y,w,h = (rct[0]+rct[2])//2, (rct[1]+rct[3])//2, (rct[2]-rct[0]), (rct[3]-rct[1])
        wh2scale = lambda w, h: math.sqrt(w * h) * 1.2 / 368
        scale = wh2scale(w, h)
        center = torch.Tensor([max(x, 0), max(y, 0)])
        return center, scale

    def genCentermap(self):
        center = [self.input_size[0]//2, self.input_size[1]//2]
        centermap = np.zeros((self.input_size[0], self.input_size[1], 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=self.input_size[0], size_w=self.input_size[1], center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        self.centermap = centermap

    def origin(self, idx):
        img_path = self.landmarks_frame.iloc[idx, -1]
        if self.image_root:
            img_path = os.path.join(self.image_root, img_path)

        center, scale = self.rect2center(self.landmarks_frame.iloc[idx, -5:-1])
        center = center.numpy()
        pts = self.landmarks_frame.iloc[idx, 0:-5].values
        kpt = pts.astype('float').reshape(-1, 2)
        img = np.array(cv2.imread(img_path), dtype=np.float32)
        return img, kpt, {"center": center, "scale": scale}
    
