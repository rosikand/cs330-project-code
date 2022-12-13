"""
File: data.py
------------------
Task batch data loader for multi-task learning 
"""


import torchvision.transforms as transforms
import PIL.Image as PILI
from PIL import Image
import pdb
import rsbox
import random
from rsbox import misc, ml
import torch
import numpy as np
import os
import imageio
from tqdm import tqdm
from torch.utils.data import sampler
import torch.utils.data as data
from glob import glob
import torchvision 
import torchvision.transforms as T
from torch.utils.data import IterableDataset



class ClassificationEpisodeDataset(data.Dataset):

    def __init__(self, root_path, k, n, num_query, resize=None, normalize=True, extension=None):
        """
        Episodic dataset class for classification tasks. 
        We will form a task distribution from the dataset of the form:
        task_distribution = [task_1, task_2, ..., task_n] where 
        each task_n = [(x_1, y_1), (x_2, y_2), ..., (x_k, y_k)]
        Args:
            - root_path (str): path to the root directory of the dataset containing the classes 
            - k (int): number of support examples per class
            - n (int): number of classes per task
            - num_query (int): number of query examples per class
            - resize: tuple of (h, w) for what to resize the images to. If None, won't resize. Default = None. 
            - normalize: if True, divide by 255. Default = True. 
            - extension: include only files with this extension. Default = None. 
        """

        self.root_path = root_path
        self.k = k
        self.n = n
        self.num_query = num_query
        self.resize = resize
        self.normalize = normalize
        self.extension = extension

        # form task distribution 
        dir_path = self.root_path + "/*/"
        self.class_list = glob(dir_path, recursive = True)
        self.num_tasks = len(self.class_list) // self.n
        random.shuffle(self.class_list)
        # have to eliminate the last few classes to make it divisible by n 
        last_class = self.num_tasks * self.n
        self.class_list = self.class_list[:last_class]
        self.task_distribution = list(np.split(np.array(self.class_list), self.num_tasks))
        assert len(self.task_distribution) == self.num_tasks, "Task distribution not formed correctly."
        # will be of form 
        # [task_1, task_2, ..., task_num_tasks] where
        # [task_i] = [class_1, class_2, ..., class_n], more precisely: 
        # [
        #     [class_1_path, class_2_path, ..., class_n_path],
        #     [class_1_path, class_2_path, ..., class_n_path]
        # ]




    def __getitem__(self, task_idx):

        x_support = []
        y_support = []
        x_query = []
        y_query = []

        task_set = self.task_distribution[task_idx % self.num_tasks]


        for i, class_ in enumerate(task_set):
            curr_class_imgs = ml.img_dir_to_data(class_, self.resize, self.normalize, self.extension)
            assert len(curr_class_imgs) >= self.k + self.num_query, "Not enough images in this class to form task."
            random.shuffle(curr_class_imgs)
            x_support += curr_class_imgs[:self.k]
            y_support += [i] * self.k
            x_query += curr_class_imgs[self.k:self.k + self.num_query]
            y_query += [i] * self.num_query


        x_support = np.stack(x_support)
        y_support = np.array(y_support)
        x_query = np.stack(x_query)
        y_query = np.array(y_query)

        # tensorize 
        x_support = torch.from_numpy(x_support).float()
        y_support = torch.from_numpy(y_support)
        x_query = torch.from_numpy(x_query).float()
        y_query = torch.from_numpy(y_query)


        return x_support, y_support, x_query, y_query


    def __len__(self):
        return self.num_tasks * 1




def task_batch_collate(batch):
    """
    Collate function for task batch data loader.
    for task in batch:
        sx, sy, qx, qy = task  # sx, sy, qx, qy are of shape (B, C, H, W)

    How to do this? Simple! Avoid collating. 
    """
    
    return batch



def get_loader(root_path, k, n, num_query, batch_size=1, resize=None, normalize=False, extension=None):
    ds = ClassificationEpisodeDataset(root_path, k, n, num_query, resize, normalize, extension)
    dl = torch.utils.data.DataLoader(ds, collate_fn=task_batch_collate, batch_size=batch_size)
    return dl


# ds = get_loader(root_path="mini_omniglot", k=5, n=3, num_query=10, batch_size=2, normalize=False)
# for batch in ds:
#     for task in batch:
#         sx, sy, qx, qy = task
#         print(sx.shape, sy.shape, qx.shape, qy.shape)
#         pdb.set_trace()
    
