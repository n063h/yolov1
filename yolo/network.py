import torchvision.models as model
import torch.nn as nn
import torch
import numpy as np

class YOLOv1(nn.Module):
    def __init__(self):
        super(YOLOv1, self).__init__()
        self.cls_num = 20  # number of classes
        self.grid_num=7
        self.bbox_num = 2
        print("\n------Initiating YOLO v1------\n")
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=7 // 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=1 // 2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=3 // 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.conv_layer6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 1024, out_features=4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
        )
        self.conn_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=7*7*30),
            nn.Sigmoid()
        )


    def forward(self, input):
        conv_layer1 = self.conv_layer1(input)
        conv_layer2 = self.conv_layer2(conv_layer1)
        conv_layer3 = self.conv_layer3(conv_layer2)
        conv_layer4 = self.conv_layer4(conv_layer3)
        conv_layer5 = self.conv_layer5(conv_layer4)
        yolo_feature = self.conv_layer6(conv_layer5)
        yolo_feature_flatten = yolo_feature.view(yolo_feature.shape[0], -1)
        conn_layer1 = self.conn_layer1(yolo_feature_flatten)
        output = self.conn_layer2(conn_layer1)
        return output.reshape(-1, self.grid_num,self.grid_num,self.bbox_num*5+self.cls_num) #batch_size,7,7,30
