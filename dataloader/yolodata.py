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

    def __init__(self, is_train=True, transform=None, cfg_param=None): # is_train=True, transform=my_transform, cfg_param=cfg_param
        super(Yolodata, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.num_class = cfg_param["classes"] # class의 갯수. 8개

        if self.is_train:
            self.file_dir = self.train_dir + "/JPEGImages/"
            self.anno_dir = self.train_dir + "/Annotations/"
            self.file_txt = self.train_dir + "/ImageSets/" + self.train_txt # 이미지 파일명이 기록된 txt

        else:
            self.file_dir = self.valid_dir + "/JPEGImages/"
            self.anno_dir = self.valid_dir + "/Annotations/"
            self.file_txt = self.valid_dir + "/ImageSets/" + self.valid_txt

        img_names = []
        img_data = []
        with open(self.file_txt, 'r', encoding="UTF-8", errors="ignore") as f:
            img_names = [i.replace("\n", "") for i in f.readlines()] # 이미지 파일명이 기록된 txt 파일을 읽는다. 한 줄애 파일명 한 개가 작성되어 있는데, 그 공백을 없애주고, list애 순차적으로 넣는다


        for i in img_names: # 이미지 파일 디렉토리에서 파일의 형식에 따라 img_data 리스트에 파일의 풀네임을 append
            if os.path.exists(self.file_dir + i + ".jpg"):
                img_data.append(i+".jpg")
            elif os.path.exists(self.file_dir + i + ".JPG"):
                img_data.append(i+".JPG")
            elif os.path.exists(self.file_dir + i + ".png"):
                img_data.append(i+".png")
            elif os.path.exists(self.file_dir + i + ".PNG"):
                img_data.append(i+".PNG")

        self.img_data = img_data # 클래스 변수에 넣어준다


    # get item per one element in one batch_idx
    def __getitem__(self, index):
        img_path = self.file_dir + self.img_data[index] # '/workspace/yolov3/KITTI/training/000000.jpg' 와 같은 string


        with open(img_path, 'rb') as f:
            img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8) # 위에서 작성한 img_path에서 파일을 오픈한다. RGB로 convert하고, 타입은 uchar로 한다. 이미지는 np.array로 나타낸다


            img_origin_h, img_origin_w = img.shape[:2] # image shape : [H,W,C]

        if os.path.isdir(self.anno_dir): # Annotation 디렉토리가 있다면
            txt_name = self.img_data[index] # txt_name에 해당 이미지의 파일명을 할당

            for ext in ['.png', '.PNG', '.jpg', '.JPG']:
                txt_name = txt_name.replace(ext, ".txt") # 파일 형식을 txt로 변경 -> Annotation의 파일이 txt형식으로 되어 있기 때문

            anno_path = self.anno_dir + txt_name # Annotation 디렉토리에 있는 txt 파일 string을 anno_path에 저장. 예를 들어, /workspace/yolo3/KITTI/training/Annotation/000000.txt 파일로 되어 있음

            if not os.path.exists(anno_path): # 파일이 존재하지 않는다면 다음 파일로 진행
                return

            bbox = [] #[class, center_x, center_y, width, height]
            # Annotation에 저장된 bbox(실제 바운딩박스)의 정보

            with open(anno_path, 'r') as f:
                for line in f.readlines(): # annotation txt파일에 저장된 정보를 읽는다
                    line = line.replace("\n", '') # 공백 처리. anno line : 0 0.08 0.705 0.161 0.379 와 같은 형식. type은 str임
                    gt_data = [l for l in line.split(' ')] # space를 기준으로 list를 gt 데이터를 만든다 

                   # skip when abnormal data
                    if len(gt_data) < 5: # gt_data의 요소가 5개가 안 되면 다음 파일로 간다.
                        continue
                    cls, cx, cy, w, h = float(gt_data[0]), float(gt_data[1]),float(gt_data[2]),float(gt_data[3]),float(gt_data[4])

                    bbox.append([cls,cx,cy,w,h]) # 해당 이미지의 Annotation 한 줄


            bbox = np.array(bbox) # 해당 이미지의 모든 annotation list를 array로 변환

            #skip emtyp target

            empty_target = False

            if bbox.shape[0] == 0: # annotation이 하나도 없다면
                empty_target = True # 비어있는 이미지로 판단하고 
                bbox = np.array([0,0,0,0,0]) # bbox는 [0,0,0,0,0]으로 만들어준다

            # data augmentation

            if self.transform is not None: # 이미지를 augmentation 한다면
                img, bbox = self.transform((img,bbox)) # 이미지와 bbox를 augmentation 한다. img는 img_path에서 open하여 np.array로 바꿔준 변수


            if not empty_target:
                batch_idx = torch.zeros(bbox.shape[0]) # 바운딩 박수 수 만큼 행벡터를 만든다 
                target_data = torch.cat((batch_idx.view(-1,1),torch.tensor(bbox)), dim=1) # 위에서 만든 바운딩 박스에 대한 batch_idx를 열벡터로 만들어주고, bbox와 열에 concat한다. 즉, [class, center_x, center_y, width, height]의 맨 앞에 batch_idx를 concat하는 것이다. [batch_idx, class, center_x, center_y, width, height]

            else:
                return # 빈 파일이라면 concat해줄 필요가 없다
            return img, target_data, anno_path

        else: # Annotation 디렉토리가 없다면 
            bbox = np.array([0,0,0,0,0]) # bbox는 0으로 설정
            if self.transform is not None:
                img, _ = self.transform((img,bbox)) # 변환된 img를 받는다.
            return img, None, None # 이미지를 return

    def __len__(self):
        return len(self.img_data) # 이미지 파일의 길이`
