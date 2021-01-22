import torch
from torch.autograd import Variable
from util.calc_iou import *
from yolo.network import YOLOv1
import torchvision.transforms as transforms
from yolo.yolo_resnet import YOLOv1_Resnet
from yolo.yolo_vgg import vgg19_bn
import cv2
from PIL import Image
import numpy as np

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]

test_transformer = [
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
iou_threshold=.5
bbox_conf_threshold=.1
nms_threshold=.5

def get_img(img_path):
    # img = np.array(Image.open(img_path))
    # img = img.transpose((1, 0, 2))
    # image = np.resize(img, (448, 448, 3))
    # for t in test_transformer:
    #     image=t(image)
    img = cv2.imread(img_path)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (448, 448))
    for t in test_transformer:
        img = t(img)
    return img[None,:,:,:]


def nms(boxes): # x1,y1,x2,y2,conf,cls
    num=boxes.shape[0]
    tb=[0]*num
    boxes=boxes.numpy()
    boxes=sorted(boxes,key=lambda x:x[4],reverse=True)
    while 0 in tb:
        for i in range(num):
            if tb[i]!=0:continue
            win=boxes[i]
            tb[i]=1
            for j in range(i+1,num):
                if tb[j] != 0: continue
                if calc_iou(torch.Tensor(win[:4]),torch.Tensor(boxes[j][:4]))>nms_threshold:
                    tb[j]=-1
    win=[]
    for i in range(num):
        if tb[i]==1:
            win.append(boxes[i])
    return torch.Tensor(win)


def get_box(pred):
    conf_mask=pred[:,:,:,4]>=bbox_conf_threshold
    conf_mask= conf_mask.unsqueeze(-1).expand_as(pred)
    conf_grid_a=pred[conf_mask].view(-1,30)
    conf_mask = pred[:, :, :, 9] >= bbox_conf_threshold
    conf_mask = conf_mask.unsqueeze(-1).expand_as(pred)
    conf_grid_b = pred[conf_mask].view(-1, 30)
    conf_grid=torch.cat([conf_grid_a,conf_grid_b])


    # better_mask_a=conf_grid[:,4]>conf_grid[:,9]
    # better_mask_b = conf_grid[:, 4] <= conf_grid[:, 9]
    # better_mask_a = better_mask_a.unsqueeze(-1).expand_as(conf_grid)
    # better_mask_b = better_mask_b.unsqueeze(-1).expand_as(conf_grid)
    # better_grid_a=conf_grid[better_mask_a].view(-1,30)
    conf_box=[]
    for i in range(conf_grid.shape[0]):
        grid=conf_grid[i]
        conf_cls_ind=grid[10:30].argmax()
        conf_cls=grid[10+conf_cls_ind]
        if grid[4]>grid[9]:
            grid[4]*=conf_cls
            if grid[4]<bbox_conf_threshold:continue
            conf_box.append(np.append(grid[:5].numpy(),conf_cls_ind))

        else:
            grid[9] *= conf_cls
            if grid[9] < bbox_conf_threshold: continue
            conf_box.append(np.append(grid[5:10].numpy(),conf_cls_ind))
    conf_box=torch.Tensor(conf_box)
    for i in range(len(conf_box)):
        conf_box[i][:4]=xywh2xyxy(conf_box[i][:4])
    nms_box = nms(conf_box)

    return nms_box



def predict(model,img_path):
    input= get_img(img_path)
    with torch.no_grad():
        pred = model(input)  # 1x7x7x30
        box = get_box(pred) # n*5
    return box





if __name__ == '__main__':
    load_path='./model/YOLOv1_normal_sigmoid_not_Fronzen_best_train.pth'
    model = vgg19_bn()
    model.cpu()
    model.load_state_dict(torch.load(load_path,map_location=torch.device('cpu')))
    model.eval()
    arr=['./data/VOC2007/JPEGImages/006018.jpg','./data/VOC2007/JPEGImages/003164.jpg','./data/VOC2007/JPEGImages/001894.jpg']
    for i in arr:
        predict(model,i)




