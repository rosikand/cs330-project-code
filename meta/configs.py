"""
File: configs.py 
----------------------
Specifies config parameters. 
"""


import datasets 
import models
import experiments
import torchplate
import rsbox 
import wandb
from rsbox import ml, misc
import torch.optim as optim
import segmentation_models_pytorch as smp



class BaseConfig:
    exp_name = "v2-random-data"
    experiment = experiments.MetaExp
    path = "../dists"
    k = 3
    num_query = 3
    batch_size = 4
    trainloader = datasets.get_dataloaders(path, k, num_query, batch_size)    
    model = smp.Unet(
            encoder_name='vgg11', 
            encoder_weights='imagenet', 
            classes=1, 
            in_channels=1,
            activation='sigmoid'
    )
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    num_epochs = 10000
    num_inner_steps = 10
    inner_lr = 0.4
    outer_lr = 0.001
    # logger = None
    logger = wandb.init(project = "330-seg-lambda", entity = "rosikand", name = exp_name + "-" + misc.timestamp())


    