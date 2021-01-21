import torch
import torch.nn as nn
from util.calc_iou import calc_iou
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
use_gpu = torch.cuda.is_available()
class Loss_yolov1(torch.nn.Module):
    def __init__(self):
        super(Loss_yolov1,self).__init__()
        self.l_coord = 5
        self.l_noobj = .5


    def forward(self,pred,target):
        batch_size=pred.shape[0]
        # batch_size,7,7
        which_grid_with_obj=(target[:,:,:,4]==1)
        which_grid_without_obj = (target[:, :, :, 4] !=1)
        # batch_size,7,7,30
        which_grid_with_obj = which_grid_with_obj.unsqueeze(-1).expand_as(target)
        which_grid_without_obj = which_grid_without_obj.unsqueeze(-1).expand_as(target)
        # all_obj_num*30
        target_grid_with_obj=target[which_grid_with_obj].view(-1,30)
        pred_grid_with_obj = pred[which_grid_with_obj].view(-1,30)
        all_obj_num = target_grid_with_obj.shape[0]
        # (all_obj_num*2,5)
        target_box_with_obj=target_grid_with_obj[:,:10].contiguous().view(-1,5)
        # (all_obj_num*2,5)
        pred_box_with_obj=pred_grid_with_obj[:,:10].contiguous().view(-1,5)
        # (all_no_obj_num,30)
        target_grid_without_obj=target[which_grid_without_obj].view(-1,30)
        pred_grid_without_obj = pred[which_grid_without_obj].view(-1,30)

        #find the respond_bbox (higher iou) of every_pair bbox of every grid
        #each grid has 2 bbox and 1 target box
        # (all_obj_num,5)
        # pred_respond=torch.FloatTensor(target_grid_with_obj.shape[0],5)
        #
        # for i in range(0, pred_respond.shape[0]):
        #     box_a=pred_grid_with_obj[i,:5]
        #     box_b=pred_grid_with_obj[i,5:10]
        #     target_box=target_grid_with_obj[i,0:5]
        #     # iou_a=calc_iou(box_a[:4],target_box[:4])
        #     # iou_b = calc_iou(box_b[:4], target_box[:4])
        #     iou_a=self.compute_iou(box_a[:4].view(1,-1).clone(),target_box[:4].view(1,-1).clone())[0]
        #     iou_b = self.compute_iou(box_b[:4].view(1, -1).clone(), target_box[:4].view(1, -1).clone())[0]
        #     if iou_a>iou_b:
        #         pred_respond[i]=box_a
        #     else:
        #         pred_respond[i] =box_b
        #     #conf_obj_target  is the higher iou
        #     target_box[4]=iou_a if iou_a>iou_b else iou_b
        # #obj loss
        #     #loc_loss
        # xy_loss=F.mse_loss(pred_respond[:,0:2],target_grid_with_obj[:,0:2],size_average=False)
        # wh_loss = F.mse_loss(pred_respond[:, 2:4], target_grid_with_obj[:, 2:4],size_average=False)
        # loc_loss = (xy_loss+wh_loss)

        coo_response_mask = torch.cuda.BoolTensor(all_obj_num*2,5) if use_gpu else torch.BoolTensor(all_obj_num*2,5)
        coo_response_mask.zero_()
        for i in range(0, all_obj_num*2,2):
            # box1 = box_pred[i:i + 2]
            # box1 = pred_grid_with_obj[i, :4]
            # box2 = pred_grid_with_obj[i, 5:9]
            box1=Variable(torch.cuda.FloatTensor(pred_box_with_obj[i,:4])  if use_gpu else torch.FloatTensor(pred_box_with_obj[i,:4]))
            box2 = Variable(torch.cuda.FloatTensor(pred_box_with_obj[i+1,:4]) if use_gpu else torch.FloatTensor(pred_box_with_obj[i+1,:4]))

            # box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            # box1_xyxy[:, :2] = box1[:, :2] - 0.5 * box1[:, 2:4]
            # box1_xyxy[:, 2:4] = box1[:, :2] + 0.5 * box1[:, 2:4]
            # box2 = box_target[i].view(-1, 5)
            # box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            # box2_xyxy[:, :2] = box2[:, :2] - 0.5 * box2[:, 2:4]
            # box2_xyxy[:, 2:4] = box2[:, :2] + 0.5 * box2[:, 2:4]
            # boxt=target_grid_with_obj[i,:5]
            boxt = Variable(torch.cuda.FloatTensor(target_box_with_obj[i,:4]) if use_gpu else torch.FloatTensor(target_box_with_obj[i,:4]))

            # iou = self.compute_iou(box1_xyxy[:, :4], box2_xyxy[:, :4])  # [2,1]
            # max_iou, max_index = iou.max(0)
            # max_index = max_index.data.cuda()
            iou1,iou2=calc_iou(box1,boxt),calc_iou(box2,boxt)
            higher_iou_ind=0 if iou1>iou2 else 1
            # target conf = iou
            target_box_with_obj[i + higher_iou_ind,4]=iou1 if iou1>iou2 else iou2
            coo_response_mask[i + higher_iou_ind] = True
        pred_respond_box = pred_box_with_obj[coo_response_mask].view(-1, 5)
        target_respond_box = target_box_with_obj[coo_response_mask].view(-1, 5)
        # contain_loss = F.mse_loss(box_pred_response[:, 4], box_target_response[:, 4], size_average=False)
            #loc_loss
        loc_loss = F.mse_loss(pred_respond_box[:, :2], target_respond_box[:, :2], size_average=False) + F.mse_loss(
            torch.sqrt(pred_respond_box[:, 2:4]), torch.sqrt(target_respond_box[:, 2:4]), size_average=False)

            #conf_loss
        conf_loss_obj=F.mse_loss(pred_respond_box[:,4],target_respond_box[:,4],size_average=False)
            # cls loss
        # cls_loss = F.mse_loss(pred_grid_with_obj[:, 10:30], target_grid_with_obj[:, 10:30],size_average=False)
        # class_pred = pred_grid_with_obj[:, 10:30].softmax(dim=1)
        # cls_loss = F.mse_loss(class_pred, target_grid_with_obj[:, 10:30], size_average=False)
        class_target = target_grid_with_obj[:, 10:30].argmax(axis=1)
        CrossEntropyLoss = nn.CrossEntropyLoss()
        cls_loss = CrossEntropyLoss(pred_grid_with_obj[:, 10:30], class_target)

        #no_obj loss
            #conf_loss
        conf_loss_no_obj=F.mse_loss(pred_grid_without_obj[:,[4,9]],target_grid_without_obj[:,[4,9]],size_average=False)

        return (self.l_coord*loc_loss+conf_loss_obj+self.l_noobj*conf_loss_no_obj+cls_loss)/batch_size,loc_loss,conf_loss_obj,conf_loss_no_obj,cls_loss



    def forward_loop(self, pred, labels):
        coord = 5
        noobj = .5
        loss_xy = torch.Tensor([0]).to(device)
        loss_wh = torch.Tensor([0]).to(device)
        loss_class = torch.Tensor([0]).to(device)
        loss_confidence = torch.Tensor([0]).to(device)
        batch_size = len(labels)

        for i in range(batch_size):
            for m in range(7):
                for n in range(7):
                    pred_boxes = pred[i, m, n].clone().to(device)
                    b1 = pred_boxes[:5].to(device)
                    b2 = pred_boxes[5:10].to(device)
                    pred_class = pred_boxes[10:30].to(device)
                    label = labels[i, m, n].clone().to(device)
                    if label[4] == 1:
                        # 包含目标
                        iou1 = calc_iou(b1[:4], label[:4])
                        iou2 = calc_iou(b2[:4], label[:4])
                        (winner_box, loser_box) = (b1, b2) if iou1 > iou2 else (b2, b1)
                        winner_iou = iou1 if iou1 > iou2 else iou2
                        loss_xy = loss_xy + (winner_box[0] - label[0]) ** 2 + (winner_box[1] - label[1]) ** 2
                        tmp_loss_wh = (winner_box[2] ** 0.5 - label[2] ** 0.5) ** 2 + (
                                    winner_box[3] ** 0.5 - label[3] ** 0.5) ** 2
                        tmp_loss_wh = torch.where(torch.isnan(tmp_loss_wh), torch.full_like(tmp_loss_wh, 0),
                                                  tmp_loss_wh)
                        loss_wh = loss_wh + tmp_loss_wh
                        loss_confidence = loss_confidence + ((winner_box[4] - winner_iou) ** 2)

                        loss_class = loss_class + torch.sum((pred_class - label[5:25]) ** 2)
                    else:
                        # 不包含目标
                        loss_confidence = loss_confidence + (b1[4] - label[4]) ** 2 * noobj
                        loss_confidence = loss_confidence + (b2[4] - label[4]) ** 2 * noobj

        return (loss_xy * coord + loss_wh * coord + loss_confidence + loss_class) / batch_size

if __name__ == '__main__':
    target=torch.zeros(2,7,7,30)
    for i in range(2):
        target[i, 2, 3,0:4] = F.softmax(torch.randn(4))
        target[i,2,3,4]=1
        target[i,2,3,5:10]=target[i,2,3,:5]
        target[i,2,3,10+i]=1

    pred=F.softmax(torch.randn((2,7,7,30)))
    loss_func=Loss_yolov1()
    loss=loss_func(target,target)
    print(loss)