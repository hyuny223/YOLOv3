import os, sys
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.tools import *
from train.loss import *


class Trainer:

    def __init__(self, model, train_loader, eval_loader, hparam, device, torch_writer):
        self.model = model # 초기화된 Darknet53 모델
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam["max_batch"]
        self.device = device
        self.torch_writer = torch_writer
        self.epoch = 0
        self.iter = 0
        self.yololoss = Yololoss(self.device, self.model.n_classes)
        self.optimizer = optim.SGD(model.parameters(), lr=hparam["lr"], momentum=hparam["momentum"])

        self.scheduler_multistep = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[20,40,60],
                                                            gamma = 0.5)

    def run_iter(self):
        for i, batch in enumerate(self.train_loader):

            # drop the batch when invalid values
            if batch is None:
                continue

            input_img, targets, anno_path = batch
            # targets = [batch_idx, cls, center_x, center_y, width, height]
            # targets의 row는 바운딩 박스의 갯수이다. 
            # input_img는 jpg or png 파일을 RBG로 uchar 형식으로 np.array로 만든 데이터이다
            
            input_img = input_img.to(self.device, non_blocking=True)

            output = self.model(input_img) # prediction. return yolo_result
            # 즉, 초기화된 Darknet53 모델에 input_img를 넣어 feature map을 담은 list를 출력한다 

            # get loss between output and target
            loss, loss_list = self.yololoss.compute_loss(output, targets, self.model.yolo_layers) # pred, target, yololayer. compute의 return은 loss, loss_list
            # yololayer = [Yololayer(), Yololayer(), Yololayer()]

            # get gradients
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler_multistep.step(self.iter)
            self.iter += 1

            # [loss.item(), lobj.item(), lcls.item(), lbox.item()]
            loss_name = ["total_loss", "obj_loss", "cls_loss", "box_loss"]

            if i%10 == 0:
                print(f"epoch {self.epoch} / iter {self.iter} / lr {get_lr(self.optimizer)} / loss {loss.item()}")
                self.torch_writer.add_scalar("lr", get_lr(self.optimizer), self.iter)
                self.torch_writer.add_scalar("total_loss", loss, self.iter)
                for ln, lv in zip(loss_name, loss_list):
                    self.torch_writer.add_scalar(ln, lv, self.iter)
        return loss

    def run(self):
        while True:
            self.model.train()

            #loss calculation

            loss = self.run_iter()
            self.epoch += 1

            # save model (checkpoint)

            checkpoint_path = os.path.join("./output", "model_epoch" + str(self.epoch) + ".pth")
            torch.save({"epoch ": self.epoch,
                        "iteration" : self.iter,
                        "model_state_dict" : self.model.state_dict(),
                        "optimizer_state_dict" : self.optimizer.state_dict(),
                        "loss" : loss}, checkpoint_path)

            # evaluation


