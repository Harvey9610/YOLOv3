import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from models import Model
from datasets import *

import cv2
import random
import numpy as np

def parse_data_config(path='cfg/voc.data'):
    """Parses the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def load_classes(path='data/coco.names'):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names

# Get data configuration
data_config = parse_data_config()
train_path = data_config["train"]
valid_path = data_config["valid"]
class_names = load_classes(data_config["names"])

model = Model()
model._initialize_weights()
model.train()

# Get dataloader
dataset = ListDataset(train_path, augment=True, multiscale=False)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
)

optimizer = torch.optim.Adam(model.parameters())

metrics = [
    "grid_size",
    "loss",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "cls",
    "cls_acc",
    "recall50",
    "recall75",
    "precision",
    "conf_obj",
    "conf_noobj",
]

for batch_i, (path, imgs, targets) in enumerate(dataloader):
    batches_done = batch_i

    imgs = imgs.to('cpu')
    targets = targets.to('cpu')

    optimizer.zero_grad()
    outputs, loss = model(imgs, targets)
    print(f'batch: {batch_i}, loss: {loss}')
    loss.backward()
    optimizer.step()
