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
import random
import torchvision
import torchvision.transforms as transforms
import PIL.Image as PILI
from PIL import Image
import pdb
import rsbox
from rsbox import misc, ml
import torch
import numpy as np
from glob import glob 
import os
import imageio
from tqdm import tqdm
from torch.utils.data import sampler
import torch.utils.data as data
from glob import glob
import torchvision 
import torchvision.transforms as T



class EpisodeDataset(data.Dataset):

    def __init__(self, task_distribution, k, num_query):
        """Args:
            task_distribution: takes the following form of paths to arrays: 
            [   
                # class 1 
                [(x, y), (x, y), ..., (x, y), (x, y), (x, y)],  # all y's are the same 
                [(x, y), (x, y), ..., (x, y), (x, y), (x, y)]
            ]
            that is, we have a standard [(x,y),...,(x,y)] distribution for each task/class. 
            task_distribution is specifically a list of PyTorch Dataset objects following this form. 
            Each element in the list represents a task/class. 
        """

        self.task_distribution = task_distribution
        self.k = k
        self.num_query = num_query

        self.task_idx_array = list(range(len(self.task_distribution))) 


    def __getitem__(self, task_idx_fake):

        # task_idx is irrelevant... we will use a sampled one from self.task_idx_array. 
        if len(self.task_idx_array) < 1:
            self.task_idx_array = list(range(len(self.task_distribution))) 
        assert len(self.task_idx_array) > 0
        random.shuffle(self.task_idx_array)
        task_idx = self.task_idx_array.pop(random.randrange(len(self.task_idx_array)))
        # print("Chosen: ", task_idx)
        # print("Remaining: ", self.task_idx_array)


        dist_ = pickle.load(open(self.task_distribution[task_idx], 'rb')) 
        ds = ClassDataset(dist_)
        

        # loop through ds 
        x_support = []
        y_support = []
        x_query = []
        y_query = []

        assert len(ds) >= self.k + self.num_query
            

        for i in range(self.k):
            x_support.append(ds[i][0])
            y_support.append(ds[i][1])
            
        for i in range(self.k, self.k + self.num_query):
            x_query.append(ds[i][0])
            y_query.append(ds[i][1])


        x_support = torch.stack(x_support)
        y_support = torch.stack(y_support)
        x_query = torch.stack(x_query)
        y_query = torch.stack(y_query)

        print("task dataset loaded!")

        return x_support, y_support, x_query, y_query

    def __len__(self):
        return len(self.task_distribution)




class ClassDataset(Dataset):
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



def task_batch_collate(batch):
    """
    Collate function for task batch data loader.
    for task in batch:
        sx, sy, qx, qy = task  # sx, sy, qx, qy are of shape (B, C, H, W)
    How to do this? Simple! Avoid collating. 
    """
    
    return batch



def get_dataloaders(path, k, num_query, batch_size):
    task_dist_path = glob(path + "/*")
    dist_ = EpisodeDataset(task_dist_path, k=k, num_query=num_query)
    dl = torch.utils.data.DataLoader(dist_, collate_fn=task_batch_collate, batch_size=batch_size)
    return dl 




def test_data_loading():
    # just a test function to ensure proper data loading 

    dd = get_dataloaders("../dists", k=5, num_query=10, batch_size=1)

    print(len(dd))

    for task_batch in dd:
        print(type(task_batch))
        print(len(task_batch))
        for task in task_batch:
            print(type(task))
            print(len(task))
            sx, sy, qx, qy = task
            print(sx.shape)
            print(sy.shape)
            print(qx.shape)
            print(qy.shape)
            ml.plot(sx[0], False)
            ml.plot(sy[0], False)
            ml.plot(qx[0], False)
            ml.plot(qy[0], False)
            print('task')
        print('-'*10)
        break


