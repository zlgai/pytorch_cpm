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
        self.mean = [127, 127, 127]
        self.std = [255, 255, 255]
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

        mean = np.array(self.mean, dtype=np.float32)
        std = np.array(self.std, dtype=np.float32)

        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - mean) / std
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
        preds = decode_preds(score_map, center=[[368/2-1, 368/2-1]], scale=[368/200], res=(46,46))[0]
        print(preds)
        return preds

    def farward(self, x):
        return self.model(x)[-1].data.cpu()

    def predict(self, x):
        input_tensor = self.preprocess(x)
        score_map = self.farward(input_tensor)
        return self.postProcess(score_map)

def draw_pts(img, preds):
    for i in range(0, 68):
        cv2.circle(img, (int(preds[i][0]), int(preds[i][1])), 2, (0,255,0), -1)
    return img


def predictWrap(source, model_file, config=None):
    model = HrnetPredictor(model_file)

    W, H = model.width, model.height
    cmv = CameraViewer(source)
    imgs = cmv.stream()
    for i, img in enumerate(imgs):
        t0 = time.time()
        if img.shape[0:2] != (W, H):
            img = cv2.resize(img, (W, H)) 
        preds = model.predict(img)
        print("time: ", time.time() - t0)

        img = draw_pts(img, preds)
        cv2.imshow(cmv.title, img)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("predict finished")

def parse_args(cmds=None):
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input','-i',help='input image path')
    args = parser.parse_args(cmds)
    return args


def main(cmds=None):
    args = parse_args(cmds)
    predictWrap(args.input, args.model_file, args)


if __name__ == '__main__':
    cmds = ['-i', r'H:\Project\Github\hrnet_facial_landmark\data\some_images', '-m', r'H:\Project\Github\hrnet_facial_landmark\output\cpm_20210819_140733\best.pth']
    main(cmds)
