import os
import pprint
import argparse
import cv2
import math
import os.path as osp
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from cameraViewer import CameraViewer


__DIR__ = os.path.dirname(os.path.abspath(__file__))


class HrnetPredictor(object):
    def __init__(self, model_file):
        self.loadModel(model_file)
        self.width = 368
        self.height = 368
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([255, 255, 255], dtype=np.float32)
        self.output_size = [46, 46]
    
    def loadModel(self, weights_file):
        from FaceNet import FaceNet

        model = FaceNet(num_class=68)
        # if weights_file and os.path.exists(weights_file):
        #     model.load_weights(weights_file)

        checkpoint = torch.load(weights_file)
        if checkpoint.get('state_dict'):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, False)
        del checkpoint
        self.model = model

    def preprocess(self, img):
        if img.shape[0:2] != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height))

        input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose([2, 0, 1])

        input_tensor = torch.tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor

    def postProcess(self, score_map):
        from evaluation import get_preds, decode_preds
        from transforms import transform_preds

        # coords = get_preds(score_map)  # float type
        # # 
        # scale = self.width/200
        # center = torch.Tensor([self.width//2-1, self.width//2-1])
        # preds = transform_preds(coords[0], center, scale, self.output_size)
        kpts = decode_preds(score_map, center=[[368/2-1, 368/2-1]], scale=[368/200], res=(46,46))[0]
        print(kpts)
        return kpts

    def farward(self, x):
        with torch.no_grad():
            return self.model(x)[-1].data.cpu()

    def predict(self, x):
        input_tensor = self.preprocess(x)
        score_map = self.farward(input_tensor)
        return self.postProcess(score_map)
    def draw(self, img, preds):
        return draw_pts(img, preds)
def draw_pts(img, kpts):
    img2 = img.copy()
    for k in kpts:
        x = int(k[0])
        y = int(k[1])
        cv2.circle(img2, (x, y), radius=2, thickness=-1, color=(0, 0, 255))
    return img2



def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')
    args = parser.parse_args(cmds)
    return args


def predictWrap(source, model, config=None):

    H, W = model.height, model.width
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    for i, img in enumerate(imgs):
        t0 = time.time()
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (H, W)) 
        kpts = model.predict(img)
        print("time: ", time.time() - t0)

        img2 = draw_pts(img, kpts)
        cv2.imshow(cmv.title, img2)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("predict finished")
def main(cmds=None):
    args = parse_args(cmds)
    model = HrnetPredictor(args.model_file)
    predictWrap(args.input, model, args)


if __name__ == '__main__':
    cmds = ['-i', r'H:\Project\Github\hrnet_facial_landmark\data\some_images', '-m', r'H:\Project\Github\hrnet_facial_landmark\output\cpm_20210819_140733\best.pth']
    main(cmds)
