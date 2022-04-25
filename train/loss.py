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

    def compute_loss(self, pred, targets, yololayer): 
        # class Trainer의 def run_iter에서 받는다 ( pred = self.model(input_img), targets = batch[1], yololayer = self.model.yolo_layers(Sequential에 Yololayer가 포함되어 있다면, class Yololayer()를 할당)
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
        # preds = [[batch, anchor, lh, lw, box_attrib] * 3]
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
            # 앵커박스의 크기를 stride에 맞춰 선형적으로 스케일링

            gain[2:6] = torch.tensor(preds[yi].shape)[[3,2,3,2]] # grid_w, grid_h
            # preds = [[batch, anchor, lh, lw, box_attrib] * 3]

            t = targets * gain

            # targets =  [batch_id, class_id, box_cx, box_cy, box_w, box_h] + [anchor_id]
            # gain = [1, 1, lw, lh, lw, lh, 1]
            # t = [batch_id, class_id, cx*lw, cy*lh, width*lw, height*lh, anchor_id]
            # 즉, grid의 width와 height에 맞추어 lh, lw를 스케일링

            # choose best anchor
            if num_targets: # bbox의 개수가 1 이상이라면 
                r = t[:,:,4:6] / anchors[:, None] 
                # anchors[:, None] 은, [3,2] 행렬을 [3, 1, 2]로 만들어준다
                # t[:,:,4:6]은 width*lw, height*lh를 의미한다. [3, bbox수, 2]
                # width*lw, height*lh를 anchors로 나눈다 
                # target value 의 w, h값을 왜 나누어주냐. b_w = p_w * e^(t_w)에서 e^(t_w)를 구해주기 위함
                # 논문에서 b_w = p_w(우선순위앵커박스의 width) * e^(t_w)에서 b_w / p_w를 구해주는 것이라고 보면 된다. r은 e^(t_w)이다. 로그를 씌워주면 t_w가 나오겠지.

                # select thre raios less than 4
                # 너무 큰 앵커들은 거른다는 건데, 왜 max(r, 1/r)을 하는거지?

                # print("---------------------")
                # print("---------------------")
                # print(f"this is max : \n {torch.max(r, 1./r)}")
                # print("\n")
                # print(f"this is dim2 : \n {torch.max(r, 1./r).max(dim=2)}")
                # print("\n")
                # print(f"this is dim2[0] : \n {torch.max(r, 1./r).max(dim=2)[0]}")
                # print("\n")
                # print(f"this is dim2[0]<4 : \n {torch.max(r, 1./r).max(dim=2)[0]<4}")
                # print("---------------------")
                # print("---------------------")
                # sys.exit(1)
                j = torch.max(r, 1. / r).max(dim=2)[0] < 4
                # 3개의 앵커박스에서의 prior anchor를 스케일링 한 값들 중,
                # r과 1. /r 을 한 각 요소 중 큰것만을 뽑아 [3, num_bbox, 2]를 만들어주고 
                # 스케일링 된 (x,y) 중(dim=2) 큰 것만 모아 각 앵커 박스에, num_bbox만큼 tensor를 만들어 준 후
                # 결과가 [value, indice]가 나오는데, [0]을 이용하여 value tensor를 불러오고,
                # < 4를 이용하여, 4보다 작은 값들을 True, 아닌 값을 False로 만든다
                # 즉, 각 앵커박스에, gt에 해당하는  [width*lw / anchor_x, height*lh / anchor_h]중, (x,y) (1/x, 1/y)중 가장 큰 값이 4가 안 넘는 gt만 True로 선택한다는 것이다.
                
                # print("----------------")
                # print("----------------")
                # print(f"this is j : \n {j}")
                # print(f"this is t : \n {t}")
                # print(f"this is t[j] : \n {t[j]}")
                # print("----------------")
                # print("----------------")
                # sys.exit(1)
                t = t[j]
                # True값인 녀석만 t로 가져온다
                # 즉, True로 선택된 각 앵커박스에 대한 box만 뽑아온다. prior anchor box인 듯 하다.

            else: # bbox의 개수가 0이라면 
                t = targets[0] # 맨 위에 있는 Annotation을 가져온다

            # batch_id, class_id
            b, c = t[:, :2].long().T
            # 선택된 box의 batch_id는 batch_id끼리, class는 class끼리 묶어 준다

            gxy = t[:, 2:4] # 선택된 box의 [cx*lw, cy*lh] 
            gwh = t[:, 4:6] # 선택된 box의 [width*lw / anchor_x, height*lh / anchor_y]
            gij = gxy.long() # 해당 부분의 그리드 인덱스
            gi, gj = gij.T

            # anchor index
            a = t[:, 6].long()

            # x.clamp는 x의 값은 변경되지 않지만, x.clamp_는 x의 값도 output의 값으로 변경된다
            # add indiecs
            indices.append((b, a, gj.clamp_(0,gain[3]-1), gi.clamp_(0,gain[2]-1)))
            # gain = [1, 1, lw, lh, lw, lh, 1]

            # add target_box
            tboxes.append(torch.cat((gxy-gij, gwh), dim=1))

            # add anchor
            anch.append(anchors[a])

            # add class
            tcls.append(c)

            print(tcls)
            print("--")
            print(tboxes)
            print("--")
            print(indices)
            print("--")
            print(anch)
            sys.exit(1)

        return tcls, tboxes, indices, anch



