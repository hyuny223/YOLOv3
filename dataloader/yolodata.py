import torch
from torch.utils.data import Dataset
import os, sys

from PIL import Image
import numpy as np
import torchvision

class Yolodata(Dataset):

    file_dir = ''
    anno_dir = ''
    file_txt = ''


    train_dir = '/workspace/yolov3/KITTI/training'
    train_txt = 'train.txt'

    valid_dir = '/workspace/yolov3/KITTI/eval'
    valid_txt = 'eval.txt'

    class_str = ["Car", "Van" , "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
    num_class = None
    img_data = []

    def __init__(self, is_train=True, transform=None, cfg_param=None):
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param["classes"]

        if self.is_train:
            self.file_dir = self.train_dir + "/JPEGImages/"
            self.anno_dir = self.train_dir + "/Annotations/"
            self.file_txt = self.train_dir + "/ImageSets/" + self.train_txt

        else:
            self.file_dir = self.valid_dir + "/JPEGImages/"
            self.anno_dir = self.valid_dir + "/Annotations/"
            self.file_txt = self.valid_dir + "/ImageSets/" + self.valid_txt

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding="UTF-8", errors="ignore") as f:
            img_names = [i.replace("\n", "") for i in f.readlines()]


        for i in img_names:
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")

        self.img_data = img_data


    # get item per one element in one batch_idx
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index]

        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)


            img_origin_h, img_origin_w = img.shape[:2] # image shape : [H,W,C]

        if os.path.isdir(self.anno_dir):
            txt_name = self.img_data[index]

            for ext in ['.png', '.PNG', '.jpg', '.JPG']:
                txt_name = txt_name.replace(ext, ".txt")

            anno_path = self.anno_dir + txt_name

            if not os.path.exists(anno_path):
                return

            bbox = [] #[class, center_x, center_y, width, height]

            with open(anno_path, 'r') as f:
                for line in f.readlines():
                    line = line.replace("\n", '')
                    gt_data = [l for l in line.split(' ')]

                   # skip when abnormal data
                    if len(gt_data) < 5:
                        continue
                    cls, cx, cy, w, h = float(gt_data[0]), float(gt_data[1]),float(gt_data[2]),float(gt_data[3]),float(gt_data[4])

                    bbox.append([cls,cx,cy,w,h])


            bbox = np.array(bbox)

            #skip emtyp target

            empty_target = False

            if bbox.shape[0] == 0:
                empty_target = True
                bbox = np.array([0,0,0,0,0])

            # data augmentation

            if self.transform is not None:
                img, bbox = self.transform((img,bbox))


            if not empty_target:
                batch_idx = torch.zeros(bbox.shape[0])
                target_data = torch.cat((batch_idx.view(-1,1),torch.tensor(bbox)), dim=1)
            else:
                return
            return img, target_data, anno_path

        else:
            bbox = np.array([0,0,0,0,0])
            if self.transform is not None:
                img, _ = self.transform((img,bbox))
            return img, None, None

    def __len__(self):
        return len(self.img_data)
