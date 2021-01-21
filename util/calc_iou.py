import torch
img_size=448
grid_num=7


def xyxy2xywh(xmin, ymin, xmax, ymax, real_size):
    real_w, real_h = real_size
    # 以img为单位
    x = (xmin + xmax) / (2 * real_w)
    y = (ymin + ymax) / (2 * real_h)
    w = (xmax - xmin) / real_w
    h = (ymax - ymin) / real_h
    return x, y, w, h

def xywh2xyxy(box,img_size=448):
    a=box
    w=a[2]*img_size
    h=a[3]*img_size
    b=torch.zeros(4)
    b[:2]=a[:2]*img_size
    b[2] = b[0] + w / 2
    b[3] = b[1] + h / 2
    b[0] = b[0]-w/2
    b[1] = b[1]-h/2

    return b


def calc_iou(A, B):
    boxA = xywh2xyxy(A)
    boxB = xywh2xyxy(B)

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

if __name__ == '__main__':
    x1,y1,x2,y2=130,114,417,243
    x,y,w,h=xyxy2xywh(x1,y1,x2,y2,(448,448))
    b=xywh2xyxy(torch.Tensor([x,y,w,h]))
    print(b)
    calc_iou(torch.Tensor([0.1,0.2,1,0.7]),torch.Tensor([0.1,0.2,1,0.7]))
