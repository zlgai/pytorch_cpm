import numpy as np
import torch
import time
import cv2
from pytorch_cpm.cameraViewer import CameraViewer

stride = 8
sigma = 3.0


def construct_model(pre_model_path, num_class=14):

    model = cpm_model.CPM(k=num_class)
    state_dict = torch.load(pre_model_path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()

    return model

def get_kpts(maps, img_h = 368.0, img_w = 368.0):

    # maps (1,15,46,46)
    maps = maps.clone().cpu().data.numpy()
    map_6 = maps[0]

    kpts = []
    for m in map_6[1:]:
        h, w = np.unravel_index(m.argmax(), m.shape)
        x = int(w * img_w / m.shape[1])
        y = int(h * img_h / m.shape[0])
        kpts.append([x,y])
    return kpts


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)


class CpmPredictor(object):
    def __init__(self, model_file, num_class=14):
        self.loadModel(model_file, num_class=num_class)
        self.width = 368
        self.height = 368
        self.mean = np.array([128.0, 128.0, 128.0], dtype=np.float32)
        self.std = np.array([256.0, 256.0, 256.0], dtype=np.float32)
        self.output_size = [46, 46]
        self.center = [184, 184]

        self.genCentermap()
        self.model.eval()

    def loadModel(self, weights_file, num_class):
        self.model = construct_model(weights_file, num_class=num_class)

    def genCentermap(self):
        # shape [N,C, H,W], [1,1, 368,368]
        center = self.center
        centermap = np.zeros((self.width, self.height, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=self.height, size_w=self.width, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map
        centermap = torch.from_numpy(centermap.transpose((2, 0, 1)))
        centermap = torch.unsqueeze(centermap, 0)
        self.centermap = centermap

    def preprocess(self, img):
        if img.shape[0:2] != (self.height, self.width):
            img = cv2.resize(img, (self.height, self.width))
        
        input_image = img
        # input_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
        #print("input_image.shape:", input_image.shape)
        input_image = input_image.astype(np.float32)
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose([2, 0, 1])

        input_tensor = torch.tensor(input_image)
        input_tensor = input_tensor.unsqueeze(0)
        #print("input_tensor.shape", input_tensor.shape)
        return input_tensor

    def postProcess(self, score_map):
        kpts = get_kpts(score_map, img_h=self.height, img_w=self.width)
        print(kpts)
        return kpts

    def farward(self, x):
        with torch.no_grad():
            return self.model(x, self.centermap)[-1].data.cpu()

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
    import argparse
    parser = argparse.ArgumentParser(description='Train Face Alignment')
    parser.add_argument('--model-in', '-m', dest="model_file", help='model parameters', required=True)
    parser.add_argument('--input', '-i', help='input image path')
    parser.add_argument('--output', '-o', help='output image path')
    args = parser.parse_args(cmds)
    return args


def predictWrap(source, model, args=None):
    H, W = model.height, model.width
    cmv = CameraViewer(source, args)
    imgs = cmv.stream()
    for i, img in enumerate(imgs):
        t0 = time.time()
        if img.shape[0:2] != (H, W):
            img = cv2.resize(img, (H, W)) 
        kpts = model.predict(img)

        print("time: ", time.time() - t0)
        img2 = model.draw(img, kpts)
        cv2.imshow(cmv.title.format(i=i), img2)
        cv2.imwrite('test.jpg', img2)
        k = cv2.waitKey(cmv.waitTime)
        if k == 27:
            break
    print("predict finished")

def singleImageDemo(img_path, model):
    img = cv2.imread(img_path)
    assert img is not None, "not found %s" % img_path
    print(img.shape)
    img = cv2.resize(img, (model.height, model.width))
    kpts = model.predict(img)
    img2 = draw_pts(img, kpts)
    cv2.imwrite('test.jpg', img2)

def main(cmds=None):
    args = parse_args(cmds)
    img_path = args.input
    model = CpmPredictor(args.model_file, num_class=68)

    predictWrap(img_path, model)
    # singleImageDemo(img_path, model)


if __name__ == "__main__":
    cmds = ['-m', '../ckpt/cpm_latest.pth.tar', '-i', '../samples/test_example2.png']
    cmds = ['-m', '../ckpt/face68_latest.pth_63000.tar', '-i', r'../samples/1.txt']
    main()