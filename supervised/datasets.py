"""
File: datasets.py
------------------
This file holds various dataset and dataloading
functions. 
"""

import cloudpickle as cp
import torch
from torch.utils.data import Dataset
import torchplate
from torchplate import utils as tp_utils
import requests 
from urllib.request import urlopen
import rsbox
from rsbox import ml, misc
import pickle 
import pdb
import torchvision


class BaseDataset(Dataset):
    def __init__(self, data_set):
        self.data_distribution = data_set
        
    def __getitem__(self, index):
        sample = self.data_distribution[index % len(self.data_distribution)][0]
        label = self.data_distribution[index % len(self.data_distribution)][1]
        sample = torch.tensor(sample, dtype=torch.float) / 255.0
        label = torch.tensor(label)

        # resize image 
        c_fn = torchvision.transforms.CenterCrop(128)
        # r_fn = torchvision.transforms.Resize(size=(128, 128))
        sample = c_fn(sample)
        label = c_fn(label)

        # t_fn = torchvision.transforms.Pad((0, 12))
        # sample = t_fn(sample)
        # label = t_fn(label)

        # add channel dimension 
        sample = torch.unsqueeze(sample, 0)
        label = torch.unsqueeze(label, 0)

        
        return (sample, label) 
        
    def __len__(self):
        return len(self.data_distribution) * 1



def get_dataloaders(path):
    print("-"*100)
    print(path)
    in_file = open(path, "rb")
    data_distribution = pickle.load(in_file)[:3]  # scale
    torch_set = BaseDataset(data_distribution)
    train_dataset, test_dataset = tp_utils.split_dataset(torch_set, ratio=0.4)
    trainloader = torch.utils.data.DataLoader(train_dataset)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
    return trainloader, testloader
