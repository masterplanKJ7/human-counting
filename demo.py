import cv2
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from preprocess import prep_image
from darknet import Darknet
from util import *

def prep_image(img, inp_dim):
    img = cv2.resize(img, (inp_dim, inp_dim))
    img = img[:,:,::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def write_rectangle(tensor, img):
    classes = load_classes('data/coco.names')
    c1 = tuple(tensor[1:3].int())
    c2 = tuple(tensor[3:5].int())
    cls = int(tensor[-1])
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, (0, 255, 0), 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, (0, 255, 0), -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

def arg_parse():
    parser = argparse.ArgumentParser(description = 'YOLO v3 Cam Demo')'
    parser.add_argument('--confidence', dest = 'confidence', help = 'Object Confidence to filter predictions', default = 0.25)
    parser.add_argument('--nms_thresh', dest = 'nms_thresh', help = 'NMS Threshhold', default = 0.4)
    parser.add_argument('--reso', dest = 'reso', 
                        help = 'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed',
                        default = '160', type = str)
    return parser.parse_args()

if __name__ == '__main__':

    CUDA = torch.cuda.is_available()
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)

    cfgfile = 'cfg/yolov3.cfg'
    weightsfile = 'yolov3.weights'
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    model.net_info['height'] = args.reso
    model.eval()

    num_classes = 80
    inp_dim = int(model.net_info['height'])

    cap = cv2.VideoCapture(0) #インカメを使用

    while cap.isOpened():

        ret, frame = cap.read()

        if ret:

            img = prep_image(frame, inp_dim) #画像の前処理

            output = model(Variable(img), CUDA) #画像の推論
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim)) / inp_dim
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]

            for o in output:
                frame = write_rectangle(o, frame) #枠とラベルを描画

            cv2.imshow('frame', frame) #結果を描画

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break
