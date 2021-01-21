import torch.nn as nn
import math,copy
import torch
import torch.nn.functional as F
import torchvision
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class YOLOv1_Resnet(nn.Module):

    def __init__(self,model_name="resnet34",pretrained=True):
        super(YOLOv1_Resnet, self).__init__()
        self.output_num=7*7*30
        if model_name=="resnet18":
            model = torchvision.models.resnet18(pretrained=pretrained)
        if model_name=="resnet34":
            model = torchvision.models.resnet34(pretrained=pretrained)
        if model_name=="resnet101":
            model = torchvision.models.resnet101(pretrained=pretrained)
        for param in model.parameters():
            param.requires_grad = False
        self.conv1,self.bn1,self.relu,self.maxpool,self.layer1,self.layer2,self.layer3,self.layer4,self.avgpool=model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4,model.avgpool
        input_num = model.fc.in_features
        #self.fc = nn.Linear(input_num*14*14, self.output_num)
        self.conn_layer1 = nn.Sequential(
            nn.Linear(in_features=14 * 14 * input_num, out_features=4096),
            nn.Dropout(),
            nn.LeakyReLU(0.1)
        )
        self.conn_layer2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=7*7*30),
            nn.ReLU()
        )

    def feature_extractor(self,img):
        f=self.conv1(img)
        f=self.bn1(f)
        f = self.relu(f)
        f = self.maxpool(f)
        f = self.layer1(f)
        f = self.layer2(f)
        f = self.layer3(f)
        f = self.layer4(f)
        # f = self.avgpool(f)
        return f

    def forward(self,input):
        feature = self.feature_extractor(input)
        feature=feature.view(input.shape[0],-1)
        conn1=self.conn_layer1(feature)
        conn2=self.conn_layer2(conn1)
        output=conn2.view(-1,7,7,30)
        return output