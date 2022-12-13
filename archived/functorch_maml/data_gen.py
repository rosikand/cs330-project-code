"""
File: data_gen.py
------------------
Same as data.py but without using PyTorch
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


class CustomTaskSet:
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
        
        # edit dataset list 
        if len(self.all_classes) % self.n != 0:
            self.all_classes = self.all_classes[:-(len(self.all_classes) % self.n)]

        assert (len(self.all_classes) % self.n) == 0, "num available classes must be divisible by num classes (n)"

        # class-wide variables 
        self.task_idx = -1


    @staticmethod
    def img_dir_to_data(dirpath, image_size, extension):
        data_subset = []
        # sub_set = glob(dirpath + '/*.' + extension)
        # sub_set = [x for x in fruits if "a" in x]
        # sub_set = dirpath + os.listdir(dirpath)
        sub_set = [os.path.join(dirpath,  curr_file) for curr_file in os.listdir(dirpath)]
        for elem in sub_set:
            image = Image.open(elem)
            image_array = np.array(image)
            image_array = image_array.astype(float)
            image_array = image_array/255.0
            
            # custom one-time hack 
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array, image_array, image_array], axis=0)
            
            assert len(image_array.shape) == 3

            # switch dim 
            if (image_array.shape[0] != 3 and image_array.shape[2] == 3) or (image_array.shape[0] != 1 and image_array.shape[2] == 1):
                image_array = np.moveaxis(image_array, -1, 0)
            

            if image_array.shape[0] == 1:
                image_array = np.repeat(image_array, 3, 0)


            assert image_array.shape[0] == 3

            # resize 
            image = np.array(T.Resize(size=image_size)(torch.tensor(image_array)))


            # tensorize 
            if not torch.is_tensor(image):
                image = torch.tensor(image)

            image = image.float()
            data_subset.append(image)
        
        if len(data_subset) < 10:
            pdb.set_trace()

        return data_subset   
    

    def get_images(self, class_dir):
        # return a list of k + num_query images from the class
        distro = self.img_dir_to_data(class_dir, self.image_size, self.extension)
        if len(distro) < (self.k + self.num_query):
            pdb.set_trace()
        assert len(distro) >= (self.k + self.num_query), "each class must contain atleast k + num_query samples"
        class_support = distro[:self.k]
        class_query = distro[self.k:(self.k + self.num_query)]
        return class_support, class_query
    

    def __next__(self):
        # return one task (k-shot, n-way classification)
        # sample the first n classes 

        self.task_idx += 1

        index = self.task_idx
        
        if index == 0:
            lower = index
            upper = 1 * self.n
        else:
            lower = (index - 1) * self.n
            upper = index * self.n
        
        # stopper 
        if int(upper) >= len(self.all_classes):
            self.task_idx = -1  # reset
            raise StopIteration
        else:
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

    
    def __iter__(self):
        return self



def get_loader(root_path, k, n, num_query, extension):
    multiset = CustomTaskSet(
        root_path=root_path,
        k=k, 
        n=n, 
        num_query=num_query, 
        extension=extension
    )

    return multiset

