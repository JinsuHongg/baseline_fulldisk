# basic package
import os
import time
import datetime
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

# pytorch package
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

# predefined class
from .scripts.model import Alexnet, Mobilenet, ResNet18, ResNet34, ResNet50
from .scripts.train import SolarFlSets, HSS2, TSS, F1Pos, train_loop, test_loop

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
torch.backends.cudnn.benchmark = True
print('1st check cuda..')
print('Number of available device', torch.cuda.device_count())
print('Current Device:', torch.cuda.current_device())
print('Device:', device)

# model
model = 'Alexnet'
model_path = "./baseline_fulldisk/results/trained/Alexnet_202503_train2011to2013_test2024_neg-under_1.pth"

# test set and calibration set
img_dir = '/workspace/data/hmi_jpgs_512'
df_test = pd.read_csv('./baseline_fulldisk/scripts/data/24image_bin_class_test_12min.csv')
df_test['Timestamp'] = pd.to_datetime(df_test['Timestamp'], format = '%Y-%m-%d %H:%M:%S')
data_test = SolarFlSets(annotations_df = df_test, img_dir = img_dir, normalization = True)
test_dataloader = DataLoader(data_test, batch_size = 16, shuffle = False)

# define model here
if model == 'Alexnet':
    net = Alexnet().to(device)
elif model == "Mobilenet":
    net = Mobilenet().to(device)
elif model == "Resnet18":
    net = ResNet18().to(device)
elif model == "Resnet34":
    net = ResNet34().to(device)
elif model == "Resnet50":
    net = ResNet50().to(device)
else:
    print("Model Selected: ", model)
    print('Invalid Model')
    exit()

# trained model with "torch.nn.DataParallel", the keys of weights should be corrected
checkpoint = torch.load(model_path, 
                        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
new_state_dict = OrderedDict()
for k, v in checkpoint['model_state_dict'].items():
    new_key = k.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = v
net.load_state_dict(new_state_dict)

loss_fn = nn.CrossEntropyLoss() 
test_loss, test_result = test_loop(test_dataloader,  model=net, loss_fn=loss_fn)
table = confusion_matrix(test_result[:, 1], test_result[:, 0]).ravel()
HSS_score = HSS2(table)
TSS_score = TSS(table)
F1_score = f1_score(test_result[:, 1], test_result[:, 0], average='macro')

training_result = []
training_result.append([test_loss, HSS_score, TSS_score, F1_score])

df_result = pd.DataFrame(training_result, columns=['Test_loss', 'HSS', 'TSS', 'F1_macro'])
df_result.to_csv("./baseline_fulldisk/results/validation/testset_2014_12min.csv", index = False)

with open("./baseline_fulldisk/results/prediction/testset_2014_12min.npy", 'wb') as f:
    test_log = np.save(f, test_result)

print(f"HSS: {HSS_score}, TSS: {TSS_score}, F1-macro: {F1_score}")