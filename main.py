import numpy as np
import argparse
import os, sys

import torch
import torch.nn as nn
from utils.tools import *
from dataloader.yolodata import *
from torch.utils.data.dataloader import DataLoader
from dataloader.data_transforms import *
from model.yolov3 import *
from train.trainer import *

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

def train(cfg_param = None, using_gpus = None):
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
                            shuffle = True)

    model = Darknet53(args.cfg, cfg_param, training=True)
    model.train()
    model.initialize_weights()

    trainer = Trainer(model=model, train_loader=train_loader, eval_loader=None, hparam=cfg_param)
    trainer.run()


def eval(cfg_param = None, using_gpus = None):
    print("eval")

def demo(cfg_param = None, using_gpus = None):
    print("demo")




if __name__ == "__main__":
    print("main")

    args = parse_args()

    #cfg parser
    net_data = parse_hyperparam_config(args.cfg)
    cfg_param = get_hyperparam(net_data)

    using_gpus = [int(g) for g in args.gpus]

    if args.mode == "train":
        train(cfg_param=cfg_param, using_gpus=using_gpus)
    elif args.mode == "eval":
        eval(cfg_param=cfg_param)
    elif args.mode == "demo":
        demo(cfg_param=cfg_param)
    else:
        print("unknown mode")

