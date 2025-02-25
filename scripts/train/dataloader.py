import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
# import torchvision.transforms as transforms

class SolarFlSets(Dataset):
    def __init__(self, annotations_df: pd.DataFrame, img_dir: str , num_sample = False, random_state:int=1004, transform=None, target_transform=None, normalization=False):
        
        if num_sample:
            self.df = annotations_df.sample(n=num_sample, random_state=random_state)
        else: self.df = annotations_df

        self.img_dir = img_dir
        self.transform = transform
      
        self.target_transform = target_transform
        self.norm = normalization

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
    
        # deploy channel if necessary
        timestamp = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, 
                                f'{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}/' + \
                                f'HMI.m{timestamp.year}.{timestamp.month:02d}.{timestamp.day:02d}_' + \
                                f'{timestamp.hour:02d}.{timestamp.minute:02d}.{timestamp.second:02d}.jpg')
        image = read_image(img_path).repeat(3, 1, 1).float()
        label = self.df.iloc[idx, 2] #0: timestamp 1: GOES class 2: target label
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.norm:
            image = image / 255 # zero to one normalization
        return image, label