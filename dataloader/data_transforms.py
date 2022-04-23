import numpy as np
import cv2
import torch
from torchvision import transforms as tf

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from utils.tools import *

def get_transformations(cfg_param = None, is_train = None):
    if is_train:
        data_transform = tf.Compose([AbsoluteLabels(),
                                    DefaultAug(),
                                    RelativeLabels(),
                                    ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                    ToTensor()])
    else:
        data_transform = tf.Compose([AbsoluteLabels(),
                                    ResizeImage(new_size = (cfg_param['in_width'], cfg_param['in_height'])),
                                    RelativeLabels(),
                                    ToTensor()])
    return data_transform


#absolute bbox
class AbsoluteLabels(object):
    def __init__(self,):
        pass

    def __call__(self,data):
        image, label = data
        h, w, _ = image.shape
        label[:,[1,3]] *= w #cx, w
        label[:,[2,4]] *= h #cy, h

        return image, label

class RelativeLabels(object):
    def __init__(self,):
        pass

    def __call__(self, data):
        image, label = data
        h, w, _ = image.shape
        label[:,[1,3]] /= w #cx, w
        label[:,[2,4]] /= h #cy, h

        return image, label

class ToTensor(object):
    def __init__(self,):
        pass

    def __call__(self, data):
        image, labels = data
        image = torch.tensor(np.transpose(np.array(image, dtype=float) /255, (2,0,1)), dtype=torch.float32) #HWC -> CHW
        labels = torch.FloatTensor(np.array(labels))

        return image, labels


class ResizeImage(object):
    def __init__(self, new_size, interpolation=cv2.INTER_LINEAR):
        self.new_size = tuple(new_size)
        self.interpolation = interpolation

    def __call__(self,data):
        image, label = data
        image = cv2.resize(image, self.new_size, interpolation = self.interpolation)

        return image, label


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        #unpack data
        img, labels = data

        #convert xywh to xyxy (minx miny maxx maxy)
        boxes = np.array(labels)
        boxes[:,1:] = xywh2xyxy_np(boxes[:,1:])

        #convert bbox to imgaug format
        bounding_boxes = BoundingBoxesOnImage(
                [BoundingBox(*box[1:], label=box[0]) for box in boxes],
                shape=img.shape)

        #apply augmentations
        img, bounding_boxes = self.augmentations(image=img,
                                                bounding_boxes = bounding_boxes)
        bounding_boxes = bounding_boxes.clip_out_of_image()

        # convert bounding_boxes to np.array()
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

            # return [x,y,w,h]
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = (x1 + x2) /2
            boxes[box_idx, 2] = (y1 + y2) /2
            boxes[box_idx, 3] = (x2 - x1)
            boxes[box_idx, 4] = (y2 - y1)

        return img, boxes


class DefaultAug(ImgAug):
    def __init__(self,):
        self.augmentations = iaa.Sequential([iaa.Sharpen(0.0, 0.1),
                                            iaa.Affine(rotate=(-0,0),
                                            translate_percent=(-0.1,0.1),
                                            scale=(0.8,1.5))])



