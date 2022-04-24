import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
import torch

# parse model layer configuration
def parse_model_config(path):
    file = open(path, 'r') # path의 파일을 읽는다. args.cfg
    lines = file.read().split('\n') #file에 있는 자료를 read. 공백 기준으로 나눋나.
    lines = [x for x in lines if x and not x.startswith("#")] # 해쉬(주석)인 부분을 제외한 나머지를 lines에 넣는다
    lines = [x.rstrip().lstrip() for x in lines] # 앞 뒤 공백 부분을 모두 제거한다. strip()으로 사용해도될 것 같고, lines에서 한 번에 처리해도 될 듯

    module_defs = []
    type_name = None
    for line in lines:
        if line.startswith("["): # [로 시작한다면, 즉 네트워크의 이름이라면
            type_name = line[1:-1].rstrip() # 괄호를 제외한 내용이 type_name. [net]라면, net를 말한다

            if type_name == 'net': # type_name이 net라면 continue하여 돌려보낸다.
                continue

            module_defs.append({}) # net부분이 아니라면 module_defs에 자리를 마련. dictionary로 할당.
            module_defs[-1]["type"] = type_name # 마련된 자리에 "type"에 type_name을 할당한다

            if module_defs[-1]["type"] == "convolutional": # 만약에 그 이름이 conv라면,
                module_defs[-1]["batch_normalize"] = 0 # batch_normalize에는 0을 할당한다.

        else: # 네트워크의 이름이 아니라면, 즉 네트워크의 내용이라면
            if type_name == "net":
                continue

            key, value = line.split('=') # 할당 연산자를 기준으로 key와 value를 나눈다
            value = value.strip() # value에 공백을 제거한다.
            module_defs[-1][key.rstrip()] = value.strip() # module_defs에 할당된 레이어에 대한 key의 dictionary에 value 값을 할당
    return module_defs # cfg에 해당하는 내용을 할당한 dictionary list를 return


#Parse the yolov3 configuration
def parse_hyperparam_config(path): # 위가 net가 아닌 layer를 가져오는 것이었다면, 여기는 net의 정보를 가져온다.
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x  in lines if x and not x.startswith("#")]
    lines = [x.rstrip().lstrip() for x in lines]
    # strip() : Return a copy of the sequence with specified leading and trailing bytes removed.

    # startswith : Return True if string starts with the prefix, otherwise return False.
    module_defs = []
    for line in lines:
        if line.startswith('['):
            type_name = line[1:-1].rstrip()
            if type_name != "net":
                continue
            module_defs.append({})
            module_defs[-1]["type"] = type_name
            if module_defs[-1]["type"] == "convolutional":
                module_defs[-1]["batch_normalize"] = 0
        else:
            if type_name != "net":
                continue
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def get_hyperparam(data): # 위에서 parse_hyperparam_config()를 통해 얻은 module_defs를 딕셔너리로 만든다
    for d in data:
        if d["type"] == "net":
            batch = int(d["batch"])
            subdivision = int(d["subdivisions"])
            momentum = float(d["momentum"])
            decay = float(d["decay"])
            saturation = float(d["saturation"])
            lr = float(d["learning_rate"])
            burn_in = int(d["burn_in"])
            max_batch = int(d["max_batches"])
            lr_policy = d["policy"]
            in_width = int(d["width"])
            in_height = int(d["height"])
            in_channels = int(d["channels"])
            classes = int(d["class"])
            ignore_clas = int(d["ignore_cls"])

            return {"batch":batch,
                    "subdivision":subdivision,
                    "momentum":momentum,
                    "decay":decay,
                    "saturation": saturation,
                    "lr":lr,
                    "burn_in":burn_in,
                    "max_batch":max_batch,
                    "lr_policy":lr_policy,
                    "in_width":in_width,
                    "in_height":in_height,
                    "in_channels":in_channels,
                    "classes":classes,
                    "ignore_clas":ignore_clas}
        else:
            continue



def xywh2xyxy_np(x: np.array):
    y = np.zeros_like(x)
    y[...,0] = x[...,0] - x[...,2] / 2 # minx
    y[...,1] = x[...,1] - x[...,3] / 2 # miny
    y[...,2] = x[...,0] + x[...,2] / 2 # maxx
    y[...,3] = x[...,1] + x[...,3] / 2 # maxy

    return y

def drawBox(img):
    img = img * 255

    img_data = np.array(np.transpose(img,(1,2,0)), dtype=np.uint8)
    img_data = Image.fromarray(img_data)

    # plt.imshow(img_data)
    # plt.show()

    # img_data.save("test.jpg")


# box_a, box_b IOU
def bbox_iou(box1, box2, xyxy=False, eps = 1e-9):
    box2 = box2.T

    if xyxy:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    else:
        b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] /2
        b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] /2
        b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] /2
        b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] /2

    # intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # union
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    union = b1_w * b1_h + b2_w * b2_h - inter + eps

    iou = inter / union

    return iou


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
