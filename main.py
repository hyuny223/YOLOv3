import numpy as np
import argparse
import os, sys
import gc

import torch
import torch.nn as nn
from utils.tools import *
from dataloader.yolodata import *
from torch.utils.data.dataloader import DataLoader
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *

from tensorboardX import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOV3_PYTORCH arguments")
    parser.add_argument("--gpus", type=int, nargs="+", default=[], help="List of GPU device ID")
    parser.add_argument("--mode", type=str, help="mode : train / eval / demo", default=None)
    parser.add_argument("--cfg", type=str, help="model config path", default=None)
    parser.add_argument("--checkpoint", type=str, help="model checkpoint path", default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args=parser.parse_args()
    return args

def collate_fn(batch):
    batch = [data for data in batch if data is not None]

    # skip invaild data
    if len(batch) == 0:
        return

    imgs, targets, anno_path = list(zip(*batch))

    imgs = torch.stack([img for img in imgs])

    for i, boxes in enumerate(targets):

        # insert index of batch
        boxes[:,0] = i

    targets = torch.cat(targets, 0)

    return imgs, targets, anno_path

def train(cfg_param = None, using_gpus = None): # cfg_param : net dictionary4
    print("train")
    my_transform = get_transformations(cfg_param = cfg_param, is_train=True)

    train_data = Yolodata(is_train=True,
                        transform=my_transform,
                        cfg_param=cfg_param)
    train_loader = DataLoader(train_data,
                            batch_size = cfg_param['batch'],
                            num_workers=0,
                            pin_memory = True,
                            drop_last = True,
                            shuffle = True,
                            collate_fn = collate_fn)


    model = Darknet53(args.cfg, cfg_param, training=True)
    model.train()
    model.initialize_weights()

    # set divice
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    torch_writer = SummaryWriter("./output")

    trainer = Trainer(model=model,
                    train_loader=train_loader,
                    eval_loader=None,
                    hparam=cfg_param,
                    device = device,
                    torch_writer=torch_writer)
    trainer.run()

def eval(cfg_param = None, using_gpus = None):
    print("eval")

def demo(cfg_param = None, using_gpus = None):
    print("demo")




if __name__ == "__main__":
    print("main")

    args = parse_args()

    #cfg parser
    net_data = parse_hyperparam_config(args.cfg) # net layer의 정보
    cfg_param = get_hyperparam(net_data) # 구해진 net layer의 정보를 dictionary로

    using_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        train(cfg_param=cfg_param, using_gpus=using_gpus)
    elif args.mode == "eval":
        eval(cfg_param=cfg_param)
    elif args.mode == "demo":
        demo(cfg_param=cfg_param)
    else:
        print("unknown mode")

