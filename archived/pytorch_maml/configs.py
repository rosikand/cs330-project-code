"""
File: configs.py
------------------
Holds config classes. 
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data 
import wandb
import maml
import rsbox 
from rsbox import ml, misc



class BaseConfig:
    experiment_name = "dev"
    trainloader = data.get_loader(root_path="datasets/universal_omniglot", k=5, n=3, num_query=10, batch_size=16, normalize=False)
    model = maml.ThreeWayNet()
    criterion = nn.CrossEntropyLoss()
    num_inner_steps = 100
    inner_lr = 0.01
    outer_lr = 0.01
    verbose = True
    num_epochs = 1
    logger = None
    #logger = wandb.init(project = "torch-maml-dev", entity = "rosikand", name = experiment_name + "-" + misc.timestamp())



class Setup330:
    # matches the original config for CS 330 
    experiment_name = "ohdev-330Config"
    trainloader = data.get_loader(root_path="datasets/universal_omniglot", k=1, n=5, num_query=15, batch_size=16, normalize=False)
    model = maml.FiveWayNet()
    criterion = nn.CrossEntropyLoss()
    num_inner_steps = 1
    inner_lr = 0.4
    outer_lr = 0.001
    verbose = True
    num_epochs = 10
    # logger = None
    logger = wandb.init(project = "torch-maml-dev", entity = "rosikand", name = experiment_name + "-" + misc.timestamp())
