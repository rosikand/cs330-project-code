"""
File: experiments.py
------------------
This file holds the experiments which are
subclasses of torchplate.experiment.Experiment. 
"""

import numpy as np
import torchplate
from torchplate import (
        experiment,
        utils
    )
import pdb
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from rsbox import ml, misc
import torch.nn.functional as F
import models
import segmentation_models_pytorch as smp



class BaseExp(experiment.Experiment):
    def __init__(self, config): 
        self.cfg = config
        self.model_class = self.cfg.model_class
        self.model = self.model_class.model
        self.trainloader = self.cfg.trainloader
        self.testloader = self.cfg.testloader
        self.criterion = self.cfg.loss_fn
        self.optimizer = self.cfg.optimizer
        self.logger = self.cfg.logger
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model.to(self.device)
        
       
        super().__init__(
            model = self.model,
            optimizer = self.optimizer,
            trainloader = self.trainloader,
            wandb_logger = self.logger,
            verbose = True
        )

    
    # provide this abstract method to calculate loss 
    def evaluate(self, batch):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        logits = self.model_class.forward_pipeline(x)
        loss_val = self.criterion(logits, y)
        return loss_val

    
    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def test(self):
        # test the model on the test set
        iou_scores = []
        with torch.no_grad():
            for batch in tqdm(self.testloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self.model_class.forward_pipeline(x)
                iou_score = self.compute_iou(logits, y)
                iou_scores.append(iou_score.cpu())

                # # visualize 
                # bin_pred_mask = self.logit_to_mask(logits, threshold=0.5)
                # plot_img = torch.hstack((x[0][0], bin_pred_mask[0][0], y[0][0]))
                # ml.plot(plot_img, color=False)


        iou_score_avg = np.mean(np.array(iou_scores))
        if self.logger is not None:
            self.logger.log({"test_iou": iou_score_avg})
        print(f"Average IoU score (testset): {iou_score_avg}")
    

    # def on_epoch_end(self):
    #     # test 
    #     self.test()
    #     self.model.train()
    


    def logit_to_mask(self, logits, threshold=0.5):
        """
        Applies binarization to go from 
        normalized logits --> per-pixel class prediction. 
        Arguments:
        -----------
        - logits: normalized logits ([n, c, h, w]), 
        - threshold: threshold to classify pixel as class or background. default = 0.5.  
        Returns:
        ----------
        - pred_mask: processed mask ([n, c, h, w]) 
        """
        pred_mask = (logits > threshold).to(torch.int32)
        return pred_mask
