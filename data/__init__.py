from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch,cv2
import random
from util.predict import *
from PIL import Image
from data import read_xml,write_txt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

voc_dir='./data/VOC2007'
train_txt_path='./data/train_data.txt'
test_txt_path='./data/test_data.txt'



class voc_dataset(Dataset):
    def __init__(self,data_type,transform):
        txt_path=train_txt_path if data_type=='train' else test_txt_path
        try:
            txt = open(txt_path, 'r')
        except Exception:
            init_txt()
        else:
            txt.close()

        self.transform = transform
        # 目标正方形
        self.img_size=448
        self.grid_num=7
        self.class_num=20
        self.box_num=2

        with open(txt_path) as f:
            lines= f.readlines()
        self.labels =[]
        self.gt_boxes=[]
        self.img_paths=[]
        self.img_real_size = []

        for line in lines:
            blocks=line.split(' ')
            self.img_paths.append(blocks[0])
            img = np.array(Image.open(blocks[0]))
            #w,h
            real_size=(img.shape[1],img.shape[0])
            self.img_real_size.append(real_size)
            #一张图片多个box和label
            gt_boxes=[]
            img_labels = []
            for b in blocks[1:]:
                grid=b.split('+')
                xmin = float(grid[0])
                ymin = float(grid[1])
                xmax =float(grid[2])
                ymax =float(grid[3])
                x, y, w, h=self.convert_box(xmin,ymin,xmax,ymax,real_size)
                c = int(grid[4])
                gt_boxes.append([x,y,w,h])
                img_labels.append(c)
            self.gt_boxes.append(gt_boxes)
            self.labels.append(img_labels)




    def __getitem__(self,index):
        img = cv2.imread(self.img_paths[index])
        img = self.BGR2RGB(img)
        img = cv2.resize(img,(self.img_size,self.img_size))
        for t in self.transform:
            img = t(img)
        gt_boxes=self.gt_boxes[index]
        labels=self.labels[index]
        target = self.format(gt_boxes, labels)  # 7x7x30

        return img,target,self.img_paths[index]

    def convert_box(self,xmin,ymin,xmax,ymax,real_size):
        """
        :param xmin: x1
        :param ymin: y1
        :param xmax: x2
        :param ymax: y2
        :param real_size: real_w,real_h
        :return: x,y,w,h
        """
        real_w,real_h=real_size
        #以img为单位
        x = (xmin+xmax)/(2*real_w)
        y = (ymin + ymax) / (2*real_h)
        w=(xmax-xmin)/real_w
        h=(ymax-ymin)/real_h
        return x,y,w,h

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def format(self,boxes,labels):
        """
        :param boxes: torch.size([2,4]) or torch.size([box_num_in_img,(xywh)])
        :param labels: torch.size([2])  or torch.size([box_num_in_img])
        :return: torch.size([7,7,30]) or torch.size([grid_num,grid_num,box_num*5+class_num])
        """
        target = torch.zeros(self.grid_num,self.grid_num,5*2+self.class_num)
        grid_ratio=1./self.grid_num
        for i in range(len(boxes)):
            box=boxes[i]
            box_grid_ind_x=int(np.ceil(box[0]/grid_ratio))-1
            box_grid_ind_y=int(np.ceil(box[1]/grid_ratio))-1
            if target[box_grid_ind_x][box_grid_ind_y][4]==1:#已经有目标标注了
                continue
            target[box_grid_ind_x][box_grid_ind_y][4]=1 #confidence
            target[box_grid_ind_x][box_grid_ind_y][0:4]=torch.Tensor(box) #xywh

            target[box_grid_ind_x][box_grid_ind_y][9] = 1 #confidence
            target[box_grid_ind_x][box_grid_ind_y][5:9] = torch.Tensor(box) #xywh

            target[box_grid_ind_x][box_grid_ind_y][10 + labels[i]] = 1 #cls

        return target

    def __len__(self):
        return len(self.labels)


def init_txt():
    xml_files_obj = read_xml.read(voc_dir)
    # write_txt.wirte(VOC_CLASSES, xml_files_obj, './data/voc2007.txt')
    i=int(len(xml_files_obj)*0.7)
    random.shuffle(xml_files_obj)
    write_txt.wirte(VOC_CLASSES, xml_files_obj[:i], train_txt_path)
    write_txt.wirte(VOC_CLASSES, xml_files_obj[i:], test_txt_path)
    return



if __name__ == '__main__':
    dataset = voc_dataset('train', transform = [transforms.ToTensor()])
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    for i, (x, y,z) in enumerate(train_loader):
        print(i)
    b = [123]
