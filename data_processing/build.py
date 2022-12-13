"""
File: build.py
---------------- 
Builds the dataset from the raw images and annotations. 
"""


import numpy as np
import json
import pdb
from detectron2.structures import polygons_to_bitmask
from tqdm.auto import tqdm
from PIL import Image
import pickle



def poly_to_mask(polygon, height, width):
    return polygons_to_bitmask(polygon, height=520, width=704).astype(int)


def build_cell_dist(json_file, root_path, num_images):
    """
    Build a list of the form:
    [(image_array, mask_array), ...]

    for the first num_images in the json_file. paths must be relative from root_path. 
    """

    return_dict = {}
    f = open(json_file)
    ds = json.load(f)
    images = ds['images']


    for i, image in enumerate(images):
        # nested dict to store the annotations for each image
        base_annot = np.zeros((520, 704)).astype(int)
        img_file = image['file_name']
        
        # load the image 
        img = Image.open(root_path + "/" + img_file)
        img_array = np.array(img).astype(float)

        image_dict = {"image_array": img_array, "annotation": base_annot}
        return_dict[image['id']] = image_dict
        if i > num_images:
            break

    annotations = ds['annotations']


    for ann_id, ann_dict in tqdm(annotations.items()):
        # ann is a dict! 
        img_id = ann_dict['image_id']

        if img_id in return_dict:
            bitmask_ = poly_to_mask(ann_dict['segmentation'], 520, 704)
            return_dict[img_id]['annotation'] = np.bitwise_or(bitmask_, return_dict[img_id]['annotation'])

    
    rdist = []
    for img_id, img_dict in return_dict.items():
        x = img_dict['image_array']
        y = img_dict['annotation']
        rdist.append((x, y))

    return rdist



def main():
    # build the dataset

    cell_types = ['shsy5y', 'skbr3', 'a172', 'skov3', 'bt474', 'huh7', 'mcf7', 'bv2']
    num_images_per_task = 20

    for cell in tqdm(cell_types):
        data_dir = "images/" + cell
        annotation_file = "annotations/" + cell + ".json"
        rdist = build_cell_dist(annotation_file, data_dir, num_images_per_task)
        # serialize 
        save_path = "dists/" + str(cell) + "_dist" + ".pkl"
        out_file = open(save_path, "wb")
        pickle.dump(rdist, out_file)
        out_file.close()



if __name__ == "__main__":
    main()
