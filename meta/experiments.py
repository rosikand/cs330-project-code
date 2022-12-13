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
import pprint
import higher
from tqdm.auto import tqdm
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from rsbox import ml, misc
import torch.nn.functional as F
import models
import os 
import segmentation_models_pytorch as smp



class MetaExp:
    def __init__(self, config):
        self.cfg = config
        self.model = self.cfg.model
        ml.print_model_size(self.model)
        self.trainloader = self.cfg.trainloader
        self.criterion = self.cfg.loss_fn
        self.logger = self.cfg.logger
        self.num_epochs = self.cfg.num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

        # meta 
        self.outer_lr = self.cfg.outer_lr
        self.inner_lr = self.cfg.inner_lr
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        self.num_inner_steps = self.cfg.num_inner_steps
    

    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def train(self):
        print(torch.cuda.memory_summary())
        print("Training beginning...")
        print("Device: ", self.device)
        
        # training loop 
        nidx = 0
        for outer_step in tqdm(range(self.num_epochs)):
            nidx += 1
            if nidx % 100 == 0:
                self.save_weights(step=nidx)

            epoch_maml_loss = 0.0
            print('----------------------------------')
            print(f"Epoch {outer_step} of {self.num_epochs}...")
            for i, task_batch in tqdm(enumerate(self.trainloader)):
                meta_losses = []
                pre_adapt_support_losses = []
                post_adapt_support_losses = []
                pre_adapt_query_losses = []
                    
                pre_adapt_support_iou = []
                post_adapt_support_iou = []
                pre_adapt_query_iou = []
                post_adapt_query_iou = []

                for task in task_batch:
                    sx, sy, qx, qy = task
                    
                    # inner loop 
                    with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                        # pre adapt metrics 
                        fmodel = fmodel.to(self.device)
                        sx = sx.to(self.device)
                        sy = sy.to(self.device)
                        qx = qx.to(self.device)
                        qy = qy.to(self.device)

                        pre_support_logits = fmodel(sx)
                        pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
                        pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
                        pre_query_logits = fmodel(qx)
                        pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
                        pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

                        # adapt 
                        iidx = 0
                        for inner_step in range(self.num_inner_steps):
                            iidx += 1
                            loss = self.criterion(fmodel(sx), sy)
                            # print(f"inner_los (step {iidx}): ", loss)
                            diffopt.step(loss)

                        # meta loss step 
                        query_y_hat = fmodel(qx)
                        meta_loss = self.criterion(query_y_hat, qy)
                        meta_losses.append(meta_loss)

                        # post adapt metrics 
                        post_support_logits = fmodel(sx)
                        post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
                        post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
                        post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))


                self.outer_opt.zero_grad()
                maml_loss = sum(meta_losses) / len(meta_losses)
                epoch_maml_loss += maml_loss
                maml_loss.backward()
                self.outer_opt.step()

                # logging
                metrics_dict = {
                    "maml_loss": maml_loss.item(),
                    "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                    "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                    "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                    "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
                    "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
                    "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
                    "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
                }

                if self.logger is not None:
                    self.logger.log(metrics_dict)

                print(f"Metrics (outer step {outer_step}, task batch {i}): ")
                pprint.pprint(metrics_dict)
                print('-'*100)
            
            epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)
            print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
            if self.logger is not None:
                self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})
   

    def save_weights(self, step, save_path=None):
        """
        Function to save model weights at 'save_path'. 
        Arguments:
        - save_path: path to save the weights. If not given, defaults to current timestamp. 
        """ 
        if save_path is None:
            if not os.path.exists("saved"):
                os.makedirs("saved")
            save_path = "saved/step-" + str(step) + "-" + misc.timestamp() + ".pth"
        torch.save(self.model.state_dict(), save_path)
        print("Model weights saved at: " + str(save_path))




# ---------------------------------------------------------



class MetaExpGD:
    # only update meta params after epoch instead of per batch 
    def __init__(self, config):
        self.cfg = config
        self.model = self.cfg.model
        self.trainloader = self.cfg.trainloader
        self.criterion = self.cfg.loss_fn
        self.logger = self.cfg.logger
        self.num_epochs = self.cfg.num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

        # meta 
        self.outer_lr = self.cfg.outer_lr
        self.inner_lr = self.cfg.inner_lr
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        self.num_inner_steps = self.cfg.num_inner_steps
    

    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def train(self):
        print("Training beginning...")
        print("Device: ", self.device)
        
        # training loop 
        for outer_step in tqdm(range(self.num_epochs)):
            epoch_maml_loss = 0.0
            print('----------------------------------')
            print(f"Epoch {outer_step} of {self.num_epochs}...")
            for i, task_batch in tqdm(enumerate(self.trainloader)):
                meta_losses = []
                pre_adapt_support_losses = []
                post_adapt_support_losses = []
                pre_adapt_query_losses = []
                    
                pre_adapt_support_iou = []
                post_adapt_support_iou = []
                pre_adapt_query_iou = []
                post_adapt_query_iou = []

                for task in task_batch:
                    sx, sy, qx, qy = task
                    
                    # inner loop 
                    with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                        # pre adapt metrics 
                        fmodel = fmodel.to(self.device)
                        sx = sx.to(self.device)
                        sy = sy.to(self.device)
                        qx = qx.to(self.device)
                        qy = qy.to(self.device)

                        pre_support_logits = fmodel(sx)
                        pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
                        pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
                        pre_query_logits = fmodel(qx)
                        pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
                        pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

                        # adapt 
                        iidx = 0
                        for inner_step in range(self.num_inner_steps):
                            iidx += 1
                            loss = self.criterion(fmodel(sx), sy)
                            # print(f"inner_los (step {iidx}): ", loss)
                            diffopt.step(loss)

                        # meta loss step 
                        query_y_hat = fmodel(qx)
                        meta_loss = self.criterion(query_y_hat, qy)
                        meta_losses.append(meta_loss)

                        # post adapt metrics 
                        post_support_logits = fmodel(sx)
                        post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
                        post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
                        post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))


                maml_loss = sum(meta_losses) / len(meta_losses)
                epoch_maml_loss += maml_loss

                # logging
                metrics_dict = {
                    "maml_loss": maml_loss.item(),
                    "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                    "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                    "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                    "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
                    "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
                    "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
                    "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
                }

                if self.logger is not None:
                    self.logger.log(metrics_dict)

                # print(f"Metrics (outer step {outer_step}, task batch {i}): ")
                # pprint.pprint(metrics_dict)
                # print('-'*100)
            
            epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)

            # outer grad step 
            self.outer_opt.zero_grad()
            epoch_avg_meta_loss.backward()
            self.outer_opt.step()
            
            print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
            if self.logger is not None:
                self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})


# ---------------------------------------------------------


class MetaExpSGD:
    # per task update
    def __init__(self, config):
        self.cfg = config
        self.model = self.cfg.model
        self.trainloader = self.cfg.trainloader
        self.criterion = self.cfg.loss_fn
        self.logger = self.cfg.logger
        self.num_epochs = self.cfg.num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

        # meta 
        self.outer_lr = self.cfg.outer_lr
        self.inner_lr = self.cfg.inner_lr
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        self.num_inner_steps = self.cfg.num_inner_steps
    

    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def train(self):
        print("Training beginning...")
        print("Device: ", self.device)
        
        # training loop 
        for outer_step in tqdm(range(self.num_epochs)):
            epoch_maml_loss = 0.0
            print('----------------------------------')
            print(f"Epoch {outer_step} of {self.num_epochs}...")
            for i, task_batch in tqdm(enumerate(self.trainloader)):
                meta_losses = []
                pre_adapt_support_losses = []
                post_adapt_support_losses = []
                pre_adapt_query_losses = []
                    
                pre_adapt_support_iou = []
                post_adapt_support_iou = []
                pre_adapt_query_iou = []
                post_adapt_query_iou = []

                for task in task_batch:
                    sx, sy, qx, qy = task
                    
                    # inner loop 
                    with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                        # pre adapt metrics 
                        fmodel = fmodel.to(self.device)
                        sx = sx.to(self.device)
                        sy = sy.to(self.device)
                        qx = qx.to(self.device)
                        qy = qy.to(self.device)

                        pre_support_logits = fmodel(sx)
                        pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
                        pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
                        pre_query_logits = fmodel(qx)
                        pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
                        pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

                        # adapt 
                        iidx = 0
                        for inner_step in range(self.num_inner_steps):
                            iidx += 1
                            loss = self.criterion(fmodel(sx), sy)
                            # print(f"inner_los (step {iidx}): ", loss)
                            diffopt.step(loss)

                        # meta loss step 
                        query_y_hat = fmodel(qx)
                        meta_loss = self.criterion(query_y_hat, qy)
                        meta_losses.append(meta_loss)

                        # post adapt metrics 
                        post_support_logits = fmodel(sx)
                        post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
                        post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
                        post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))


                        self.outer_opt.zero_grad()
                        meta_loss.backward()
                        self.outer_opt.step()

                maml_loss = sum(meta_losses) / len(meta_losses)
                epoch_maml_loss += maml_loss

                # logging
                metrics_dict = {
                    "maml_loss": maml_loss.item(),
                    "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                    "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                    "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                    "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
                    "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
                    "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
                    "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
                }

                if self.logger is not None:
                    self.logger.log(metrics_dict)

                print(f"Metrics (outer step {outer_step}, task batch {i}): ")
                pprint.pprint(metrics_dict)
                print('-'*100)
            
            epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)
            print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
            if self.logger is not None:
                self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})


# ---------------------------------------------------------



# class MetaExpAccum:
#     def __init__(self, config):
#         self.cfg = config
#         self.model = self.cfg.model
#         self.trainloader = self.cfg.trainloader
#         self.criterion = self.cfg.loss_fn
#         self.logger = self.cfg.logger
#         self.num_epochs = self.cfg.num_epochs
#         self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.model.to(self.device)
        

#         # meta 
#         self.outer_lr = self.cfg.outer_lr
#         self.inner_lr = self.cfg.inner_lr
#         self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
#         self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
#         self.num_inner_steps = self.cfg.num_inner_steps
    

#     @staticmethod
#     def compute_iou(logits, y):
#         # first compute statistics for true positives, false positives, false negative and
#         # true negative "pixels"
#         tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
#         iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
#         return iou_score 


#     def train(self):
#         print("Training beginning...")
#         print("Device: ", self.device)
        
#         # training loop 
#         for outer_step in tqdm(range(self.num_epochs)):
#             epoch_maml_loss = 0.0
#             print('----------------------------------')
#             print(f"Epoch {outer_step} of {self.num_epochs}...")
#             for i, task_batch in tqdm(enumerate(self.trainloader)):
#                 meta_losses = []
#                 pre_adapt_support_losses = []
#                 post_adapt_support_losses = []
#                 pre_adapt_query_losses = []
                    
#                 pre_adapt_support_iou = []
#                 post_adapt_support_iou = []
#                 pre_adapt_query_iou = []
#                 post_adapt_query_iou = []

#                 for task in task_batch:
#                     sx, sy, qx, qy = task
                    
#                     # inner loop 
#                     with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
#                         # pre adapt metrics 
#                         fmodel = fmodel.to(self.device)
#                         sx = sx.to(self.device)
#                         sy = sy.to(self.device)
#                         qx = qx.to(self.device)
#                         qy = qy.to(self.device)

#                         pre_support_logits = fmodel(sx)
#                         pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
#                         pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
#                         pre_query_logits = fmodel(qx)
#                         pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
#                         pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

#                         # adapt 
#                         iidx = 0
#                         for inner_step in range(self.num_inner_steps):
#                             iidx += 1
#                             loss = self.criterion(fmodel(sx), sy)
#                             # print(f"inner_los (step {iidx}): ", loss)
#                             diffopt.step(loss)

#                         # meta loss step 
#                         query_y_hat = fmodel(qx)
#                         meta_loss = self.criterion(query_y_hat, qy)
#                         meta_losses.append(meta_loss)

#                         # post adapt metrics 
#                         post_support_logits = fmodel(sx)
#                         post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
#                         post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
#                         post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))

            
#                 maml_loss = sum(meta_losses) / len(meta_losses)
#                 epoch_maml_loss += maml_loss


        
#                 # logging
#                 metrics_dict = {
#                     "maml_loss": maml_loss.item(),
#                     "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
#                     "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
#                     "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
#                     "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
#                     "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
#                     "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
#                     "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
#                 }

#                 if self.logger is not None:
#                     self.logger.log(metrics_dict)

#                 print(f"Metrics (outer step {outer_step}, task batch {i}): ")
#                 pprint.pprint(metrics_dict)
#                 print('-'*100)
            
#             epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)
#             print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
#             if self.logger is not None:
#                 self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})


#             self.outer_opt.zero_grad()
#             epoch_avg_meta_loss.backward()
#             self.outer_opt.step()




# ---------------------------------------------------------


class AccumExp:
    def __init__(self, config):
        self.cfg = config
        self.model = self.cfg.model
        self.trainloader = self.cfg.trainloader
        self.criterion = self.cfg.loss_fn
        self.logger = self.cfg.logger
        self.num_epochs = self.cfg.num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

        # meta 
        self.outer_lr = self.cfg.outer_lr
        self.inner_lr = self.cfg.inner_lr
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        self.num_inner_steps = self.cfg.num_inner_steps
    

    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def train(self):
        num_accum = 4
        print("Training beginning...")
        print("Device: ", self.device)
        
        # training loop 
        for outer_step in tqdm(range(self.num_epochs)):
            epoch_maml_loss = 0.0
            print('----------------------------------')
            print(f"Epoch {outer_step} of {self.num_epochs}...")
            for i, task_batch in tqdm(enumerate(self.trainloader)):
                meta_losses = []
                pre_adapt_support_losses = []
                post_adapt_support_losses = []
                pre_adapt_query_losses = []
                    
                pre_adapt_support_iou = []
                post_adapt_support_iou = []
                pre_adapt_query_iou = []
                post_adapt_query_iou = []

                for task in task_batch:
                    sx, sy, qx, qy = task
                    
                    # inner loop 
                    with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                        # pre adapt metrics 
                        fmodel = fmodel.to(self.device)
                        sx = sx.to(self.device)
                        sy = sy.to(self.device)
                        qx = qx.to(self.device)
                        qy = qy.to(self.device)

                        pre_support_logits = fmodel(sx)
                        pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
                        pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
                        pre_query_logits = fmodel(qx)
                        pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
                        pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

                        # adapt 
                        iidx = 0
                        for inner_step in range(self.num_inner_steps):
                            iidx += 1
                            loss = self.criterion(fmodel(sx), sy)
                            # print(f"inner_los (step {iidx}): ", loss)
                            diffopt.step(loss)

                        # meta loss step 
                        query_y_hat = fmodel(qx)
                        meta_loss = self.criterion(query_y_hat, qy)
                        meta_losses.append(meta_loss)

                        # post adapt metrics 
                        post_support_logits = fmodel(sx)
                        post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
                        post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
                        post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))


                
                maml_loss = sum(meta_losses) / len(meta_losses)
                maml_loss_accum = maml_loss / num_accum
                epoch_maml_loss += maml_loss_accum
                maml_loss_accum.backward()

                # gradient accumulation 
                if ((i + 1) % num_accum == 0) or (i + 1 == len(self.trainloader)):
                    self.outer_opt.zero_grad()
                    self.outer_opt.step()

                # logging
                metrics_dict = {
                    "maml_loss": maml_loss_accum.item(),
                    "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                    "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                    "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                    "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
                    "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
                    "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
                    "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
                }

                if self.logger is not None:
                    self.logger.log(metrics_dict)

                print(f"Metrics (outer step {outer_step}, task batch {i}): ")
                pprint.pprint(metrics_dict)
                print('-'*100)
            
            epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)
            print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
            if self.logger is not None:
                self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})


# ---------------------------------------------------------


class MetaExpEpoch:
    # epoch based metrics 
    def __init__(self, config):
        self.cfg = config
        self.model = self.cfg.model
        ml.print_model_size(self.model)
        self.trainloader = self.cfg.trainloader
        self.criterion = self.cfg.loss_fn
        self.logger = self.cfg.logger
        self.num_epochs = self.cfg.num_epochs
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        

        # meta 
        self.outer_lr = self.cfg.outer_lr
        self.inner_lr = self.cfg.inner_lr
        self.outer_opt = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        self.num_inner_steps = self.cfg.num_inner_steps
    

    @staticmethod
    def compute_iou(logits, y):
        # first compute statistics for true positives, false positives, false negative and
        # true negative "pixels"
        tp, fp, fn, tn = smp.metrics.get_stats(logits, y.long(), mode='binary', threshold=0.5)
        iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        return iou_score 


    def train(self):
        print(torch.cuda.memory_summary())
        print("Training beginning...")
        print("Device: ", self.device)
        
        # training loop 
        for outer_step in tqdm(range(self.num_epochs)):
            epoch_maml_loss = 0.0
            print('----------------------------------')
            print(f"Epoch {outer_step} of {self.num_epochs}...")
            pre_adapt_support_losses = []
            post_adapt_support_losses = []
            pre_adapt_query_losses = []
            post_adapt_query_losses_m = []
            pre_adapt_support_iou = []
            post_adapt_support_iou = []
            pre_adapt_query_iou = []
            post_adapt_query_iou = []

            for i, task_batch in tqdm(enumerate(self.trainloader)):
                meta_losses = []

                for task in task_batch:
                    sx, sy, qx, qy = task
                    
                    # inner loop 
                    with higher.innerloop_ctx(self.model, self.inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                        # pre adapt metrics 
                        fmodel = fmodel.to(self.device)
                        sx = sx.to(self.device)
                        sy = sy.to(self.device)
                        qx = qx.to(self.device)
                        qy = qy.to(self.device)

                        pre_support_logits = fmodel(sx)
                        pre_adapt_support_losses.append(self.criterion(pre_support_logits, sy))
                        pre_adapt_support_iou.append(self.compute_iou(pre_support_logits, sy))
                        pre_query_logits = fmodel(qx)
                        pre_adapt_query_losses.append(self.criterion(pre_query_logits, qy))
                        pre_adapt_query_iou.append(self.compute_iou(pre_query_logits, qy))

                        # adapt 
                        iidx = 0
                        for inner_step in range(self.num_inner_steps):
                            iidx += 1
                            loss = self.criterion(fmodel(sx), sy)
                            # print(f"inner_los (step {iidx}): ", loss)
                            diffopt.step(loss)

                        # meta loss step 
                        query_y_hat = fmodel(qx)
                        meta_loss = self.criterion(query_y_hat, qy)
                        meta_losses.append(meta_loss)

                        # post adapt metrics 
                        post_support_logits = fmodel(sx)
                        post_adapt_support_losses.append(self.criterion(post_support_logits, sy))
                        post_adapt_support_iou.append(self.compute_iou(post_support_logits, sy))
                        post_adapt_query_iou.append(self.compute_iou(query_y_hat, qy))


                self.outer_opt.zero_grad()
                maml_loss = sum(meta_losses) / len(meta_losses)
                epoch_maml_loss += maml_loss
                maml_loss.backward()
                self.outer_opt.step()

            
            epoch_avg_meta_loss = epoch_maml_loss / len(self.trainloader)
            print("EPOCH AVG META LOSS: ", epoch_avg_meta_loss)
            if self.logger is not None:
                self.logger.log({"EPOCH AVG META LOSS": epoch_avg_meta_loss})


            # logging
            metrics_dict = {
                "maml_loss": epoch_avg_meta_loss,
                "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                "pre_adapt_support_iou": sum(pre_adapt_support_iou) / len(pre_adapt_support_iou),
                "post_adapt_support_iou": sum(post_adapt_support_iou) / len(post_adapt_support_iou),
                "pre_adapt_query_iou": sum(pre_adapt_query_iou) / len(pre_adapt_query_iou),
                "post_adapt_query_iou": sum(post_adapt_query_iou) / len(post_adapt_query_iou),
                }

            if self.logger is not None:
                self.logger.log(metrics_dict)

            print(f"Metrics (outer step {outer_step}, task batch {i}): ")
            pprint.pprint(metrics_dict)
            print('-'*100)
