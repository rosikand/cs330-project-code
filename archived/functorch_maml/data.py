"""
File: data.py
------------------
Task batch data loader for multi-task learning 
"""

import torch 
from torch.utils.data import Dataset
import os
import rsbox 
from rsbox import ml, misc
from glob import glob
import pdb
import os
import numpy as np
from PIL import Image
import random
import torchvision 
import torchvision.transforms as T


class ImageMetaDataset(Dataset):
    def __init__(self, root_path=None, k=None, n=None, num_query=None, image_size=(128, 128), extension='png'):
        self.root_path = root_path
        self.k = k
        self.n = n
        self.num_query = num_query
        self.extension = extension
        self.image_size = image_size

        # get all class dirs 
        self.all_classes = []
        for file in os.listdir(self.root_path):
            d = os.path.join(root_path, file)
            if os.path.isdir(d):
                self.all_classes.append(d)

        assert (len(self.all_classes) % self.n) == 0, "num available classes must be divisible by num classes (n)"
    

    @staticmethod
    def img_dir_to_data(dirpath, image_size, extension):
        data_subset = []
        sub_set = glob(dirpath + '/*.' + extension)
        for elem in sub_set:
            image = Image.open(elem)
            image_array = np.array(image)
            image_array = image_array.astype(float)
            image_array = image_array/255.0

            # transform (resize and movedim) (need to clean up!)
            if len(image_array.shape) == 3:
                image = np.array(T.Resize(size=image_size)(torch.movedim(torch.tensor(image_array), -1, 0)))
            else:
                image = np.array(T.Resize(size=image_size)(torch.tensor(image_array)))
            if len(image.shape) == 3:
                if image.shape[0] != 3 and image.shape[2] == 3:
                    image = np.moveaxis(image, -1, 0)
                elif image.shape[0] != 1 and image.shape[2] == 1:
                    image = np.moveaxis(image, -1, 0)
                else:
                    image = image
            else:
                image = image

            if not torch.is_tensor(image):
                image = torch.tensor(image)

            image = image.float()
            data_subset.append(image)
        return data_subset   
    

    def get_images(self, class_dir):
        # return a list of k + num_query images from the class
        distro = self.img_dir_to_data(class_dir, self.image_size, self.extension)
        assert len(distro) >= (self.k + self.num_query), "each class must contain atleast k + num_query samples"
        class_support = distro[:self.k]
        class_query = distro[self.k:(self.k + self.num_query)]
        return class_support, class_query
    

    def __getitem__(self, index):
        # return one task (k-shot, n-way classification)
        # sample the first n classes 
        
        if index == 0:
            lower = index
            upper = 1 * self.n
        else:
            lower = (index - 1) * self.n
            upper = index * self.n
        curr_classes = self.all_classes[int(lower):int(upper)]

        support_samples = []
        support_labels = []
        query_samples = []
        query_labels = []
        class_idx = 0
        for class_dir in curr_classes:
            class_support, class_query = self.get_images(class_dir)
            support_label = [class_idx] * len(class_support)
            query_label = [class_idx] * len(class_query)
            
            # append
            support_samples += class_support
            support_labels += support_label
            query_samples += class_query
            query_labels += query_label

            class_idx += 1


        # vectorize 
        support_samples = torch.stack(support_samples, 0)
        query_samples = torch.stack(query_samples, 0)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels) 

        return support_samples, support_labels, query_samples, query_labels
    
    def __len__(self):
        return int(len(self.all_classes) / self.n)



def get_loaders(root_path, k, n, num_query, extension):
    multiset = ImageMetaDataset(
        root_path=root_path,
        k=k, 
        n=n, 
        num_query=num_query, 
        extension=extension
    )

    # note to self: batch size is not working/unintuitive (it is batch size of images sampled... not per-task)
    task_loader = torch.utils.data.DataLoader(multiset)

    return task_loader

