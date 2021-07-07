import numpy as np
import os
import torch
import argparse
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

def parse_args():
  parser = argparse.ArgumentParser('DS')
  parser.add_argument('--data_root',type=str)
  parser.add_argument('--location',type=int, default=2)
  args, _ = parser.parse_known_args()
  return args

args = parse_args()
args.data_root

# Dataset

patch_cols_n = 1
patch_rows_n = 1

resize = 256
cropsize = 224

transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                         T.CenterCrop(cropsize),
                         T.ToTensor(),
                         T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std = [0.229, 0.224, 0.225])
                         ])

DATES = sorted([os.path.join(args.data_root,date) for date in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root,date))])
dataset_path = [os.path.join(date,f"H2_Fab_15L_{args.location:04d}_{os.path.basename(date).replace('-','')[2:]}.JPG") for date in DATES]
dataset = []

for path in dataset_path:
  img = Image.open(path).convert('RGB')
  dataset.append(transform_x(img))
  
# visualization

def denormalization(x):
  mean=[0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  x = (((x.transpose(1,2,0)*std)+mean)*255.).astype(np.uint8)
  return x

for data in dataset:
  plt.imshow(denormalization(data.numpy()))
  plt.show()
