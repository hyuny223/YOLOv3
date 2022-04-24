import os,sys
import numpy as np
import torch
import torch.nn as nn

from utils.tools import *

def make_conv_layer(layer_idx : int, modules : nn.Module, layer_info : dict, in_channel : int): # Conv2d 모듈 add 후 activation function 모듈 add
    filters = int(layer_info["filters"]) # output channel size
    size = int(layer_info["size"]) # kernel size
    stride = int(layer_info["stride"]) # stride
    pad = (size - 1) // 2
    # modules = nn.Sequential()
    modules.add_module("leyer_" +str(layer_idx) + "_conv",
                        nn.Conv2d(in_channel, filters, size, stride, pad)) # Adds a child module to the current module.
    # add_module(name: str, module: Optional['Module']) -> None

    if layer_info["batch_normalize"] == '1':
        modules.add_module("layer_" + str(layer_idx) + "_bn",
                            nn.BatchNorm2d(filters))

    if layer_info["activation"] == "leaky":
        modules.add_module("layer_" + str(layer_idx) + "_act",
                            nn.LeakyReLU())

    elif layer_info["activation"] == "relu":
        modules.add_module("layer_" + str(layer_idx) + "_act",
                            nn.ReLU())

def make_shortcut_layer(layer_idx : int, modules : nn.Module):
    modules.add_module("layer_" + str(layer_idx) + "_shortcut",
                        nn.Identity())

def make_route_layer(layer_idx : int, modules : nn.Module):
    modules.add_module("layer_" + str(layer_idx) + "_route",
                        nn.Identity())

def make_upsample_layer(layer_idx : int, modules : nn.Module, layer_info: dict):
    stride = int(layer_info["stride"])
    modules.add_module("layer_" + str(layer_idx) + "_upsample",
                        nn.Upsample(scale_factor=stride, mode="nearest"))

    # Upsample(size: Optional[_size_any_t]=None, scale_factor: Optional[_ratio_any_t]=None, mode: str='nearest', align_corners: Optional[bool]=None, recompute_scale_factor: Optional[bool]=None)

class Yololayer(nn.Module):
    def __init__(self, layer_info:dict, in_width : int, in_height : int, is_train : bool):
        super(Yololayer, self).__init__()
        self.n_classes = int(layer_info["classes"])
        self.ignore_thresh = float(layer_info["ignore_thresh"])
        self.box_attr = self.n_classes + 5 # box[4] + objectness[1] + class_prob[n_classes]
        mask_idxes = [int(x) for x in layer_info["mask"].split(',')]
        # 앵커는 한 그리드 당 3개, 1앵커당 3개의 박스가 작용하고, 그래서 크기별로 9개의 박스가 존재한다(small / middle / large). 그 중 사용할 박스 세 가지를 선택

        anchor_all = [int(x) for x in layer_info["anchors"].split(',')] # 9개의 박스를 할당. 그런데 ',' 기준으로 split 했기에, 좌표의 형태로 다시 묶어주어야 함
        anchor_all = [(anchor_all[i], anchor_all[i+1]) for i in range(0, len(anchor_all), 2)] # 두 개씩 묶어 줌 (x,y)
        self.anchor = torch.tensor([anchor_all[x] for x in mask_idxes]) # mask_idxes에 해당하는 anchor를 anchor로 선택. tensor로 바꾸어 주었는데, 어떤 형식이려나?
        self.in_width = in_width
        self.in_height = in_height
        self.stride = None
        self.lw = None
        self.lh = None
        self.is_train = is_train

    def forward(self, x): # x는 feature amp
        # x is input. [X C H W], [B, 255, 19, 19], [B, 255, 76, 76], [B, 255, 38, 38] / B : the number of anchors : 3
        self.lw, self.lh = x.shape[3], x.shape[2] ## yolov3는 19, 38, 76
        self.anchor = self.anchor.to(x.device) # 앵커를 x와 동일한 디바이스에 올리기
        self.stride = torch.tensor([torch.div(self.in_width, self.lw, rounding_mode = "floor"), # self.in_width // self.lw (grid의 width 간격)
                                    torch.div(self.in_height, self.lh, rounding_mode = "floor")]).to(x.device) # self.in_height // self.lh (grid의 height 간격)


        # if kitti data. n_classes is 8. C = (8 + 5) * 3 = 39(channels)

        # [batch, box_attrib * anchor의 수(3개), height, lw, lh] e.g. [1,39,19,19]
        # self.box_attrib = self.n_classes + x, y, w, h + objectness. 즉, 앵커 하나당 채널 수

        # 4dim [batch, box_atrrib * anchor, lw, lh ] → 5dim [batch, anchor, box_attrib, lw, lh]
        # → [batch, anchor, lw, lh, box_attrib]

        # self.anchor.shape[0] = 앵커의 개수(3)
        x = x.view(-1, self.anchor.shape[0], self.box_attr, self.lh, self.lw).permute(0,1,3,4,2).contiguous()

        return x


class Darknet53(nn.Module):
    def __init__(self, cfg, param, training): # param : args.cfg, get_hyperparam() ← cfg에서 net의 딕셔너리를 가져온다, True
        super().__init__()
        self.batch = int(param['batch']) # 1
        self.in_channels = int(param['in_channels']) # 3
        self.in_width = int(param['in_width']) # 608
        self.in_height = int(param['in_height']) # 608
        self.n_classes = int(param['classes']) # 8
        self.module_cfg = parse_model_config(cfg) # tools.py에서 제작한 cfg 정보를 담은 layer에 대한 dictionary list를 self.module_cfg에 할당
        self.module_list = self.set_layer(self.module_cfg) # module_list
        self.yolo_layers = [layer[0] for layer in self.module_list if isinstance(layer[0], Yololayer)] # self.module_list의 요소가 Yololayer라면 할당?
        self.training = training

    def set_layer(self, layer_info): # layer_info란, self.module_cfg로서, cfg파일을 dictionary list로 만든 것이다.
        module_list = nn.ModuleList() # Holds submodules in a list.
        in_channels = [self.in_channels] # first channels of input. layer의 채널!!! layer를 통해서 출력된 Ouput channels를 append한다. 그러면 다음 layer에 input되는 채널 수를 알 수 있게 된다

        # shortcut : add
        # route : concat
        for layer_idx, info in enumerate(layer_info): #info는 self.module_cfg에서 받아온 dictionary이다.
            modules = nn.Sequential() # 레이어들을 하나로 sequential하게 묶어준다. add_module을 사용하면 nn.Sequential()에 모듈을 넣어줄 수 있다.

            if info["type"] == "convolutional": # 이번 레이어의 이름이 conv라면 # idx, module, layer_info, channel
                make_conv_layer(layer_idx, modules, info, in_channels[-1])
                in_channels.append(int(info["filters"])) # 레이어에 필요한 filters를 append

            elif info["type"] == "shortcut": # 이번 레이어의 이름이 shortcut이라면
                make_shortcut_layer(layer_idx, modules)
                in_channels.append(in_channels[-1]) # shortcut은 add이기에, 직전의 레이어와 channel수가 동일하다.

            elif info["type"] == "route": # 이번 레이어의 이름이 route라면
                make_route_layer(layer_idx, modules)
                layers = [int(y) for y in info["layers"].split(',')] # layers 리스트에 route의 요소를 할당. 1개일 수도 있고, 2개일 수도 있다.
                if len(layers) == 1: # route의 요소가 1개라면(e.g. n번째 이전 레이어) # 해당 n번째 레이어가 다음 레이어의 input으로 들어간다
                    in_channels.append(in_channels[layers[0]]) # 그 수에 해당하는 n번째 레이어의 channel을 할당한다.
                elif len(layers) == 2: # route의 요소가 2개라면(e.g. (n, m) # 해당 n번째 레이어와 m번째 레이어를 concat 한다
                    in_channels.append(in_channels[layers[0]] + in_channels[layers[1]]) # 그 수에 해당하는 번째 레이어의 channel의 합을 할당한다. concat이기 때문.

            elif info["type"] == "upsample": # 이번 레이어의 이름이 upsample이라면
                make_upsample_layer(layer_idx, modules, info)
                in_channels.append(in_channels[-1]) # 스케일을 up 하는 것이기에 채널 수는 이전과 같다.

            elif info["type"] == "yolo": # 이번 레이어의 이름이 yolo라면
                yololayer = Yololayer(info, self.in_width, self.in_height, self.training) # 최종 feature맵이다. 헤더라고 해야하나.
                modules.add_module("layer_" + str(layer_idx) +"_yolo", yololayer) # Adds a child module to the current module. add_module(name, module)
                in_channels.append(in_channels[-1]) # 이전 레이어와 동일한 채널 수를 갖는다.

            module_list.append(modules) # 모듈 리스트에, add된 modules를 할당.

        return module_list

    def initialize_weights(self):
        # track all layers
        for m in self.modules(): # ????????????
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1) # scale
                nn.init.constant_(m.bias, 0) # shift

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        yolo_result = []
        layer_result = []

        for idx, (name, layer) in enumerate(zip(self.module_cfg, self.module_list)):
            if name["type"] == "convolutional":
                x = layer(x)
                layer_result.append(x)

            elif name["type"] == "shortcut":
                x = x + layer_result[int(name["from"])]
                layer_result.append(x)

            elif name["type"] == "yolo":
                yolo_x = layer(x)
                layer_result.append(yolo_x)
                yolo_result.append(yolo_x)

            elif name["type"] == "upsample":
                x = layer(x)
                layer_result.append(x)

            elif name["type"] == "route":
                layers = [int(y) for y in name["layers"].split(',')]
                x = torch.cat([layer_result[l] for l in layers], dim=1)
                layer_result.append(x)

        return yolo_result
