import os
import json

import torch
from torch.optim import *
import numpy as np
from sklearn import metrics
import math

from datasets import inverse_normalize
import cv2


def normalize_img(value, vmax=None, vmin=None):
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if not (vmax - vmin) == 0:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    return value


def visualize(raw_image, boxes):
    import cv2
    boxes_img = np.uint8(raw_image.copy())[:, :, ::-1]

    for box in boxes:

        xmin,ymin,xmax,ymax = int(box[0]),int(box[1]),int(box[2]),int(box[3])

        cv2.rectangle(boxes_img[:, :, ::-1], (xmin, ymin), (xmax, ymax), (0,0,255), 1)

    return boxes_img[:, :, ::-1]


def build_optimizer_and_scheduler_adam_v2(model, args):
    # optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    imgnet = []
    others = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'imgnet' in name:
                imgnet.append(param)
            else:
                others.append(param)

    optimizer = Adam([{'params':imgnet}, {'params':others}], lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_adam(model, args):
    optimizer_grouped_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(optimizer_grouped_parameters, lr=args.init_lr, weight_decay=args.weight_decay)
    scheduler = None
    return optimizer, scheduler


def build_optimizer_and_scheduler_sgd(model, args):
    optimizer_grouped_parameters = model.parameters()
    optimizer = SGD(optimizer_grouped_parameters, lr=args.init_lr)
    scheduler = None
    return optimizer, scheduler


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, mode='w', encoding='utf-8') as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


def save_results(name_list, area_list, bb_list, iou_list, conf_list, piap_list, filename):
    with open(filename, "w") as file_iou:
        file_iou.write('name,area,bb,ciou,conf\n')
        for indice in np.argsort(iou_list):
            file_iou.write(f"{name_list[indice]},{area_list[indice]},{bb_list[indice]},{iou_list[indice]},{conf_list[indice]},{piap_list[indice]}\n")


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    wu = 0 if 'warmup_epochs' not in vars(args) else args.warmup_epochs
    if args.lr_schedule == 'cos':  # cosine lr schedule
        if epoch < wu:
            lr = args.init_lr * epoch / wu
        else:
            lr = args.init_lr * 0.5 * (1. + math.cos(math.pi * (epoch - wu) / (args.epochs - wu)))
    elif args.lr_schedule == 'cte':  # constant lr
        lr = args.init_lr
    else:
        raise ValueError

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr