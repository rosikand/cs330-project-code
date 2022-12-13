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
    experiment = experiments.BaseExp
    path = "../dists/skov3_dist.pkl"
    trainloader, testloader = datasets.get_dataloaders(path)    
    model_class = models.SmpUnet(
        encoder_name='vgg11', 
        encoder_weights='imagenet', 
        classes=1, 
        in_channels=1,
        activation='sigmoid'
    )
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    optimizer = optim.Adam(model_class.model.parameters(), lr=0.001)
    logger = None
    # logger = wandb.init(project = "r-one-task-seg", entity = "rosikand", name = misc.timestamp())


    