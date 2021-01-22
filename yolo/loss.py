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


    def forward1(self,pred,target):
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
        cls_loss = F.mse_loss(pred_grid_with_obj[:, 10:30], target_grid_with_obj[:, 10:30],size_average=False)
        # class_pred = pred_grid_with_obj[:, 10:30].softmax(dim=1)
        # cls_loss = F.mse_loss(class_pred, target_grid_with_obj[:, 10:30], size_average=False)
        # class_target = target_grid_with_obj[:, 10:30].argmax(axis=1)
        # CrossEntropyLoss = nn.CrossEntropyLoss()
        # cls_loss = CrossEntropyLoss(pred_grid_with_obj[:, 10:30], class_target)

        #no_obj loss
            #conf_loss
        conf_loss_no_obj=F.mse_loss(pred_grid_without_obj[:,[4,9]],target_grid_without_obj[:,[4,9]],size_average=False)

        return (self.l_coord*loc_loss+conf_loss_obj+self.l_noobj*conf_loss_no_obj+cls_loss)/batch_size,loc_loss,conf_loss_obj,conf_loss_no_obj,cls_loss

    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] * wh[:, :, 1] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1]) # [N, ]
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1]) # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou

    def forward(self, pred_tensor, target_tensor):
        """ Compute loss for YOLO training.
        Args:
            pred_tensor: (Tensor) predictions, sized [n_batch, S, S, Bx5+C], 5=len([x, y, w, h, conf]).
            target_tensor: (Tensor) targets, sized [n_batch, S, S, Bx5+C].
        Returns:
            (Tensor): loss, sized [1, ].
        """
        # TODO: Romove redundant dimensions for some Tensors.

        S, B, C = 7, 2, 20
        N = 5 * B + C    # 5=len([x, y, w, h, conf]

        batch_size = pred_tensor.size(0)
        coord_mask = target_tensor[:, :, :, 4] > 0  # mask for the cells which contain objects. [n_batch, S, S]
        noobj_mask = target_tensor[:, :, :, 4] == 0 # mask for the cells which do not contain objects. [n_batch, S, S]
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor) # [n_batch, S, S] -> [n_batch, S, S, N]

        coord_pred = pred_tensor[coord_mask].view(-1, N)            # pred tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_pred = coord_pred[:, :5*B].contiguous().view(-1, 5)    # [n_coord x B, 5=len([x, y, w, h, conf])]
        class_pred = coord_pred[:, 5*B:]                            # [n_coord, C]

        coord_target = target_tensor[coord_mask].view(-1, N)        # target tensor on the cells which contain objects. [n_coord, N]
                                                                    # n_coord: number of the cells which contain objects.
        bbox_target = coord_target[:, :5*B].contiguous().view(-1, 5)# [n_coord x B, 5=len([x, y, w, h, conf])]
        class_target = coord_target[:, 5*B:]                        # [n_coord, C]

        # Compute loss for the cells with no object bbox.
        noobj_pred = pred_tensor[noobj_mask].view(-1, N)        # pred tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_target = target_tensor[noobj_mask].view(-1, N)    # target tensor on the cells which do not contain objects. [n_noobj, N]
                                                                # n_noobj: number of the cells which do not contain objects.
        noobj_conf_mask = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0) # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask[:, 4 + b*5] = 1 # noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        noobj_pred_conf = noobj_pred[noobj_conf_mask]       # [n_noobj, 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask]   # [n_noobj, 2=len([conf1, conf2])]
        loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')

        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_not_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(1)# [n_coord x B, 5]
        bbox_target_iou = torch.zeros(bbox_target.size()).cuda()                    # [n_coord x B, 5], only the last 1=(conf,) is used

        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):
            pred = bbox_pred[i:i+B] # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
            pred_xyxy = Variable(torch.FloatTensor(pred.size())) # [B, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            pred_xyxy[:,  :2] = pred[:, :2]/float(S) - 0.5 * pred[:, 2:4]
            pred_xyxy[:, 2:4] = pred[:, :2]/float(S) + 0.5 * pred[:, 2:4]

            target = bbox_target[i] # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
            target = bbox_target[i].view(-1, 5) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
            target_xyxy = Variable(torch.FloatTensor(target.size())) # [1, 5=len([x1, y1, x2, y2, conf])]
            # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
            # rescale (center_x,center_y) for the image-size to compute IoU correctly.
            target_xyxy[:,  :2] = target[:, :2]/float(S) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2]/float(S) + 0.5 * target[:, 2:4]

            iou = self.compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4]) # [B, 1]
            max_iou, max_index = iou.max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_not_response_mask[i+max_index] = 0

            # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
            # from the original paper of YOLO.
            bbox_target_iou[i+max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
        bbox_target_iou = Variable(bbox_target_iou).cuda()

        # BBox location/size and objectness loss for the response bboxes.
        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)        # [n_response, 5], only the last 1=(conf,) is used
        loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
        loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]), reduction='sum')
        loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

        # Class probability loss for the cells which contain objects.
        loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

        # Total loss
        loss = self.l_coord * (loss_xy + loss_wh) + loss_obj + self.l_noobj * loss_noobj + loss_class
        loss = loss / float(batch_size)

        return loss,loss_xy + loss_wh,loss_obj,loss_noobj,loss_class

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