import torch
import torch.nn as nn
import numpy as np
import os, sys

from utils.tools import *


class Yololoss(nn.Module): # class Trainer에서 사용(self.yololoss)
    def __init__(self, device, num_class):
        super(Yololoss, self).__init__()
        self.device = device # self.device를 받음 
        self.num_class = num_class #self.model.n_classes를 받음 
        self.mseloss = nn.MSELoss().to(device)
        self.bceloss = nn.BCELoss().to(device)
        self.bcelogloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device)).to(device)

    def compute_loss(self, pred, targets, yololayer): # class Trainer의 def run_iter에서 받는다 ( pred = self.model(input_img), targets = batch[1], yololayer = self.model.yolo_layers(Sequential에 Yololayer가 포함되어 있다면, class Yololayer()를 할당)
        lcls, lbox, lobj = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)

        # pout.shape : [batch, anchors, grid_y, grid_x, box_attrib]
        # the number of boxes in each yolo layer : anchors * grid_h * grid_w
        # yolo0 -> 3 * 19 * 19, yolo1 -> 3 * 38 * 38, yolo2 -> 3 * 76 * 76
        # total boxes : 22743

        # positive prediction vs negative prediction
        # pos : neg = 0.01 : 0.99

        # Only in positive prediction, we can get box_loss and class_loss
        # in negative prediction, only obj_loss

        # get positive targets
        tcls, tbox, tindices, tanchors = self.get_targets(pred, targets, yololayer) # return tcls, tboxes, indices, anch

        # 3 yolo layers
        for pidx, pout in enumerate(pred):
            batch_id, anchor_id, gy, gx = tindices[pidx]

            tobj = torch.zeros_like(pout[...,0], device=self.device)

            num_targets = batch_id.shape[0]

            if num_targets:
                ps = pout[batch_id, anchor_id, gy, gx] # [batch, ancor, grid_w, grid_h, box_attrib]
                pxy = torch.sigmoid(ps[...,0:2])
                pwh = torch.exp(ps[..., 2:4]) * tanchors[pidx]
                pbox = torch.cat((pxy, pwh), 1)

                # assignment
                iou = bbox_iou(pbox.T, tbox[pidx], xyxy = False)


                # box loss
                # MSE
                # loss_wh = self.mseloss(pbox[..., 2:4], tbox[pidx][...,2:4])
                # loss_xy = self.mseloss(pbox[..., 0:2], tbox[pidx][..., 0:2])
                # print(f"loss_xy : {loss_xy}")
                # print(f"loss_wh : {loss_wh}")

                lbox += (1 - iou).mean()

                # objectness
                # gt box and predicted box -> positive : 1, or 0 using IOU
                tobj[batch_id, anchor_id, gy, gx] = iou.detach().clamp(0).type(tobj.dtype)

                # class loss
                if ps.size(1) - 5 > 1:
                    t = torch.zeros_like(ps[...,5:], device=self.device)
                    t[range(num_targets), tcls[pidx]] = 1

                    lcls += self.bcelogloss(ps[:, 5:], t)


            lobj += self.bcelogloss(pout[...,4], tobj)

        # loss weight
        lcls *= 0.05
        lobj *= 1.0
        lbox *= 0.5

        # total loss
        loss = lcls+ lbox + lobj
        loss_list = [loss.item(), lobj.item(), lcls.item(), lbox.item()]

        return loss, loss_list


    # prediction 값과 shape을 비슷하게 만들어 loss 계산을 쉽게 하기 위함
    # class Trainer의 def run_iter에서 받는다 ( pred = self.model(input_img), targets = batch[1], yololayer = self.model.yolo_layers(Sequential에 Yololayer가 포함되어 있다면, 그 Sequential에서의 첫 레이어)

    def get_targets(self, preds, targets, yololayer):
        # preds = [[batch, anchor, row, col, box_attrib] * 3]
        # 즉, 3개의 앵커박스 
        # yololayer = [Yololayer(), Yololayer(), Yololayer()]

        num_anc = 3 # 앵커의 수 
        num_targets = targets.shape[0] # target은 bounding box에 대한 2차원 리스트이다. bounding 박수의 갯수 
        tcls, tboxes, indices, anch = [], [], [], []

        targets = targets.to(self.device) # 바운딩 박스의 정보를 device로
        gain = torch.ones(7, device=self.device) # ????

        # anchor_index
        # repeat는 반복을 해주지만, 새로운 차원을 만들 수도 있다.
        # 예를 들어, a = [[1,2],[3,4]] 라는 텐서가 있다면, 
        # a.repeat(row, col) 을 의미한다.
        # a.repeat(2, 1)은 row 방향으로 a의 텐서를 한번 반복하여 concat한다는 것이고
        # a.repeat(1, 2)는 col 방향으로 a의 텐서를 한번 반복하여 concat한다는 것이다.
        # a.repeat(1,1,1)은 channel을 만든다는 것이며
        # a.repeat(2,1,1)은 channel 방향으로 a의 텐서를 한번 반복하여 concat한다는 것이다.
        ai = torch.arange(num_anc, device=self.device).float().view(num_anc, 1).repeat(1, num_targets) # 앵커박스 하나 당 각 바운딩 박스에 대한 열을 추가

        # [batch_id, class_id, box_cs, box_cy, box_w, box_h] + [anchor_id]
        # ai[:, :, None]은 [3,5] 를 [3,5,1]로 만든다는 것
        targets = torch.cat((targets.repeat(num_anc,1,1), ai[:, :, None]), 2) # concat은 dim이 맞아야 하기 때문에, [:, :, None]으로 3차원으로 reshape
        
        # channel : 3개의 앵커 박스 
        # row : bbox
        # col : [batch_id, cls, center_x, center_y, width, height]

        # channel : 3개의 앵커 박스 
        # row : bbox의 수 
        # col : anchor_idx?

        # anchor를 grid cell에 맞게 normalize
        for yi, yl in enumerate(yololayer):
            anchors = yl.anchor / yl.stride
            # 3개의 앵커인 (x/stride, y/stride)로 만들어 줌 


            print("---------------")
            print("---------------")
            print(torch.tensor(preds[yi].shape))
            print(torch.tensor(preds[yi].shape)[[3,2,3,2]])
            print("---------------")
            print("---------------")

            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]]  #????
            # grid_w, grid_h
            # preds = [[batch, anchor, row, col, box_attrib] * 3]

            print(f"gain : {gain}")
            sys.exit(1)

            t = targets * gain

            # targets =  [batch_id, class_id, box_cs, box_cy, box_w, box_h] + [anchor_id]
            # gain = [1, 1, grid_y(col), grid_x(row), grid_y, grid_x, 1]

            # choose best anchor
            if num_targets: # bbox의 개수가 1 이상이라면 

                r = t[:,:,4:6] / anchors[:, None] # target value 의 w, h값을 왜 나누어주냐. b_w = p_w * e^(t_w)에서 e^(t_w)를 구해주기 위함

                # select thre raios less than 4
                j = torch.max(r, 1. / r).max(2)[0] < 4

                t = t[j]

            else: # bbox의 개수가 0이라면 
                t = targets[0]

            # batch_id, class_id
            b, c = t[:, :2].long().T

            gxy = t[:, 2:4]
            gwh = t[:, 4:6]

            gij = gxy.long() # 해당 부분의 그리드 인덱스
            gi, gj = gij.T

            # anchor index
            a = t[:, 6].long()

            # add indiecs
            indices.append((b, a, gj.clamp_(0,gain[3]-1), gi.clamp_(0,gain[2]-1)))

            # add target_box
            tboxes.append(torch.cat((gxy-gij, gwh), dim=1))

            # add anchor
            anch.append(anchors[a])

            # add class
            tcls.append(c)

        return tcls, tboxes, indices, anch



