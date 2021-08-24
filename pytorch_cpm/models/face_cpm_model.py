import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, init, ModuleList
import torch.nn.functional as F


class FaceNet(Module):
    insize  =  368

    def __init__(self, num_class=70):
        super(FaceNet, self).__init__()

        ot = num_class#  + 1
        cct = 128 + ot
        # cnn to make feature map
        self.relu = ReLU()
        self.max_pooling_2d = MaxPool2d(kernel_size = 2, stride = 2)
        self.conv1_1_stage1 = Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2_stage1 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1_stage1 = Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2_stage1 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1_stage1 = Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2_stage1 = Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3_stage1 = Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_4_stage1 = Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1_stage1 = Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2_stage1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3_stage1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_4_stage1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_1_stage1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_2_stage1 = Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_3_CPM = Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)

        # stage1
        self.conv6_1_CPM = Conv2d(in_channels = 128, out_channels = 512, kernel_size = 1, stride = 1, padding = 0)
        self.conv6_2_CPM = Conv2d(in_channels = 512, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

        # stage2
        self.Mconv1_stage2 = Conv2d(in_channels = cct, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv2_stage2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv3_stage2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv4_stage2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv5_stage2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv6_stage2 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.Mconv7_stage2 = Conv2d(in_channels = 128, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

        # stage3
        self.Mconv1_stage3 = Conv2d(in_channels = cct, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv2_stage3 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv3_stage3 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv4_stage3 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv5_stage3 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv6_stage3 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.Mconv7_stage3 = Conv2d(in_channels = 128, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

        # stage4
        self.Mconv1_stage4 = Conv2d(in_channels = cct, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv2_stage4 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv3_stage4 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv4_stage4 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv5_stage4 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv6_stage4 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.Mconv7_stage4 = Conv2d(in_channels = 128, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

        # stage5
        self.Mconv1_stage5 = Conv2d(in_channels = cct, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv2_stage5 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv3_stage5 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv4_stage5 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv5_stage5 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv6_stage5 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.Mconv7_stage5 = Conv2d(in_channels = 128, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

        # stage6
        self.Mconv1_stage6 = Conv2d(in_channels = cct, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv2_stage6 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv3_stage6 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv4_stage6 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv5_stage6 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 7, stride = 1, padding = 3)
        self.Mconv6_stage6 = Conv2d(in_channels = 128, out_channels = 128, kernel_size = 1, stride = 1, padding = 0)
        self.Mconv7_stage6 = Conv2d(in_channels = 128, out_channels = ot, kernel_size = 1, stride = 1, padding = 0)

    def __call__(self, x):
        heatmaps  =  []

        h  =  self.relu(self.conv1_1_stage1(x))
        h  =  self.relu(self.conv1_2_stage1(h))
        h  =  self.max_pooling_2d(h)
        h  =  self.relu(self.conv2_1_stage1(h))
        h  =  self.relu(self.conv2_2_stage1(h))
        h  =  self.max_pooling_2d(h)
        h  =  self.relu(self.conv3_1_stage1(h))
        h  =  self.relu(self.conv3_2_stage1(h))
        h  =  self.relu(self.conv3_3_stage1(h))
        h  =  self.relu(self.conv3_4_stage1(h))
        h  =  self.max_pooling_2d(h)
        h  =  self.relu(self.conv4_1_stage1(h))
        h  =  self.relu(self.conv4_2_stage1(h))
        h  =  self.relu(self.conv4_3_stage1(h))
        h  =  self.relu(self.conv4_4_stage1(h))
        h  =  self.relu(self.conv5_1_stage1(h))
        h  =  self.relu(self.conv5_2_stage1(h))
        h  =  self.relu(self.conv5_3_CPM(h))
        feature_map  =  h

        # stage1
        h  =  self.relu(self.conv6_1_CPM(h))
        h  =  self.conv6_2_CPM(h)
        heatmaps.append(h)

        # stage2
        h  =  torch.cat([h, feature_map], dim= 1) # channel concat
        h  =  self.relu(self.Mconv1_stage2(h))
        h  =  self.relu(self.Mconv2_stage2(h))
        h  =  self.relu(self.Mconv3_stage2(h))
        h  =  self.relu(self.Mconv4_stage2(h))
        h  =  self.relu(self.Mconv5_stage2(h))
        h  =  self.relu(self.Mconv6_stage2(h))
        h  =  self.Mconv7_stage2(h)
        heatmaps.append(h)

        # stage3
        h  =  torch.cat([h, feature_map], dim= 1) # channel concat
        h  =  self.relu(self.Mconv1_stage3(h))
        h  =  self.relu(self.Mconv2_stage3(h))
        h  =  self.relu(self.Mconv3_stage3(h))
        h  =  self.relu(self.Mconv4_stage3(h))
        h  =  self.relu(self.Mconv5_stage3(h))
        h  =  self.relu(self.Mconv6_stage3(h))
        h  =  self.Mconv7_stage3(h)
        heatmaps.append(h)

        # stage4
        h  =  torch.cat([h, feature_map], dim= 1) # channel concat
        h  =  self.relu(self.Mconv1_stage4(h))
        h  =  self.relu(self.Mconv2_stage4(h))
        h  =  self.relu(self.Mconv3_stage4(h))
        h  =  self.relu(self.Mconv4_stage4(h))
        h  =  self.relu(self.Mconv5_stage4(h))
        h  =  self.relu(self.Mconv6_stage4(h))
        h  =  self.Mconv7_stage4(h)
        heatmaps.append(h)

        # stage5
        h  =  torch.cat([h, feature_map], dim= 1) # channel concat
        h  =  self.relu(self.Mconv1_stage5(h))
        h  =  self.relu(self.Mconv2_stage5(h))
        h  =  self.relu(self.Mconv3_stage5(h))
        h  =  self.relu(self.Mconv4_stage5(h))
        h  =  self.relu(self.Mconv5_stage5(h))
        h  =  self.relu(self.Mconv6_stage5(h))
        h  =  self.Mconv7_stage5(h)
        heatmaps.append(h)

        # stage6
        h  =  torch.cat([h, feature_map], dim= 1) # channel concat
        h  =  self.relu(self.Mconv1_stage6(h))
        h  =  self.relu(self.Mconv2_stage6(h))
        h  =  self.relu(self.Mconv3_stage6(h))
        h  =  self.relu(self.Mconv4_stage6(h))
        h  =  self.relu(self.Mconv5_stage6(h))
        h  =  self.relu(self.Mconv6_stage6(h))
        h  =  self.Mconv7_stage6(h)
        heatmaps.append(h)
        
        ### heatmaps = [h]
        ### return h ###
        # return ModuleList(heatmaps)
        return tuple(heatmaps)

    def farward(self, x):
        return self.__call__(x)

    def init_weights(self):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # torch.nn.init.normal_(m.weight, std=0.001)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def load_weights(self, pretrained='', cutoff=None):
        # self.load_state_dict(torch.load(weights_file))
        import os
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            keys = list(model_dict.keys())
            if cutoff:
                keys = keys[0:cutoff]
                print(keys)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in keys}
            for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
        else:
            print("not found")