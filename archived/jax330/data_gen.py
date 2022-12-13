from glob import glob
import random
import pdb
import torch
import numpy as np
import imageio
from rsbox import ml, misc
import PIL.Image as PILI
from PIL import Image
from torch.utils.data import IterableDataset
import torchvision 
import torchvision.transforms as T


def get_xy_distribution(root_path):
    dir_path = root_path + "/*/"
    class_list = glob(dir_path, recursive = True)

    label_idx = 0
    dist_ = []
    for class_ in class_list:
        cpaths = glob(class_ + "*")
        labels = [label_idx] * len(cpaths)
        data_set = list(zip(cpaths, labels))
        dist_.append(data_set)
        label_idx += 1
    
    return dist_



def image_file_to_array(filename):
    """
    Takes an image path and returns numpy array
    Args:
        filename: Image filename
        dim_input: Flattened shape of image
    Returns:
        1 channel image
    """
    image = ml.load_image(filename)
    image = image.astype(np.float32)
    image = 1.0 - image  # invert colors? (why?) 
    return image

dist_ = get_xy_distribution("../datasets/mini_omniglot")



class DataGenerator(IterableDataset):

    def __init__(self, xy_dist, num_way, num_shot, num_query):
        self.dist_ = xy_dist
        self.num_way = num_way
        self.num_shot = num_shot
        self.num_query = num_query
        self.num_samples_per_class = num_shot + num_query

    
    def _sample(self):
        # paths 
        sampled_classes = random.sample(self.dist_, self.num_way)

        batch_labels = [i for i in range(self.num_way)]

        
        support_x = []
        support_y = []
        query_x = []
        query_y = []
        for class_, new_label in zip(sampled_classes, batch_labels):
            for idx, batch in enumerate(class_):
                if idx == self.num_samples_per_class:
                    break

                x = batch[0]
                x = image_file_to_array(x)

                if idx < self.num_shot:
                    support_x.append(x)
                    support_y.append(new_label)
                else:
                    query_x.append(x)
                    query_y.append(new_label)
                
        # shuffle (yes)? 

        support_x = np.array(support_x)
        support_y = np.array(support_y)
        query_x = np.array(query_x)
        query_y = np.array(query_y)

        return support_x, support_y, query_x, query_y



    def __iter__(self):
        while True:
            yield self._sample()



def get_gen_loader(root_path, k, n, num_query, batch_size=1):
    dist_ = get_xy_distribution(root_path)
    ts = DataGenerator(dist_, n, k, num_query)
    tl = iter(torch.utils.data.DataLoader(ts, batch_size=batch_size, pin_memory=True, collate_fn=ml.numpy_collate))
    return tl
