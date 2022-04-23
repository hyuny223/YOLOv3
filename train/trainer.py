import os, sys
from sched import scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils.tools import *

class Trainer:

    def __init__(self, model, train_loader, eval_loader, hparam):
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.max_batch = hparam["max_batch"]
        self.epoch = 0
        self.iter = 0
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
            print(input_img.shape, targets.shape)

    def run(self):
        while True:
            self.model.train()

            #loss calculation

            self.run_iter()
            self.epoch += 1





