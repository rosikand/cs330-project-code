"""
File: configs.py
------------------
Holds the configs classes/objects.
"""


import rsbox
from rsbox import ml
import modules
import data
import maml_trainer
import torch.nn as nn
import cloudpickle as cp
import torch.optim as optim
import wandb
from rsbox import misc
import data_gen

class BaseConfig:
    experiment_name = "testbase"
    k = 2
    n = 5
    num_query = 5
    num_inner_steps = 20
    inner_lr = 0.1
    outer_lr = 0.001
    model = modules.get_network(n)
    dataset_path = "datasets/universal_omniglot"
    img_extension = "png"
    trainloader = data_gen.get_loader(root_path=dataset_path, k=k, n=n, num_query=num_query, extension=img_extension)
    epochs = 10
    meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.CrossEntropyLoss()
    experiment = maml_trainer.MAML
    logger = None
    # logger = wandb.init(
    #     project = "maml-dev", 
    #     entity = "rosikand",
    #     name = experiment_name + "-" + misc.timestamp()
    # )
    
class cubscfg(BaseConfig):
    # ugh the inheritence is failing! 
    dataset_path = "datasets/cubs"
    img_extension = "jpg"
    
    experiment_name = "testbase"
    k = 12
    n = 5
    num_query = 5
    num_inner_steps = 20
    inner_lr = 0.1
    outer_lr = 0.001
    model = modules.get_network(n)
    trainloader = data_gen.get_loader(root_path=dataset_path, k=k, n=n, num_query=num_query, extension=img_extension)
    epochs = 10
    meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
    criterion = nn.CrossEntropyLoss()
    experiment = maml_trainer.MAMLSGD
    logger = None


    
class higher(BaseConfig):
    # ugh the inheritence is failing! 
    dataset_path = "datasets/cubs"
    img_extension = "jpg"
    
    experiment_name = "testbase"
    k = 5
    n = 5
    num_query = 5
    num_inner_steps = 10
    inner_lr = 0.1
    outer_lr = 0.001
    model = modules.get_network(n)
    trainloader = data_gen.get_loader(root_path=dataset_path, k=k, n=n, num_query=num_query, extension=img_extension)
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    experiment = maml_trainer.MAMLHigher
    logger = None



class higheromni(BaseConfig):
    # ugh the inheritence is failing! 
    dataset_path = "datasets/universal_omniglot"
    img_extension = "png"
    
    experiment_name = "testbase"
    k = 5
    n = 5
    num_query = 5
    num_inner_steps = 10
    inner_lr = 0.1
    outer_lr = 0.001
    model = modules.get_network(n)
    trainloader = data_gen.get_loader(root_path=dataset_path, k=k, n=n, num_query=num_query, extension=img_extension)
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    experiment = maml_trainer.MAMLHigher
    logger = None
    