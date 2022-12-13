# self-contained script for training a regular inductive supervised learning network but with using functorch and torchopt 


import torch
import torchplate
from torchplate import experiment
from torchplate import utils
import functorch
from functorch import grad, grad_and_value
import torch.nn as nn
import wandb
import pdb 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rsbox import ml, misc
# import torchopt
import requests
from tqdm.auto import tqdm
import cloudpickle as cp
from urllib.request import urlopen



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 16)
        self.fc4 = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x



class OptExp:
    def __init__(self, metrics=None, logger=None, verbose=True, lr=0.01): 
        self.model_module = Net()
        self.criterion = nn.CrossEntropyLoss()
        dataset = cp.load(urlopen("https://stanford.edu/~rsikand/assets/datasets/mini_cifar.pkl")) 
        self.trainloader, self.testloader = torchplate.utils.get_xy_loaders(dataset)
        self.model, self.params = functorch.make_functional(self.model_module)  # init network 
        self.metrics = metrics
        self.verbose = verbose
        self.logger = None
        self.lr = lr

        if self.metrics is None:
            # at the very least, log the loss 
            self.metrics = {'loss': ml.MeanMetric()}
    

    def predict(self, x):
        """returns logits"""
        assert self.model is not None
        assert self.params is not None
        logits = self.model(self.params, x)
        return logits 

    def update_metrics(self, logits, labels, aux_vals):
        # note: aux_vals contains dict of already computed values
        # such as loss so that we can avoid computing them twice. 
        assert type(self.metrics) == dict
        assert self.metrics is not None

        for key, value in self.metrics.items():
            if key in aux_vals:
                value.update(aux_vals[key])
            else:
                value.update(logits, labels)
    

    @staticmethod
    def sgd_step(params, gradients, lr):
        updated_params = []
        for param, gradient in zip(params, gradients):
            update = param - (lr * gradient)
            updated_params.append(update)
        
        return tuple(updated_params)
    

    @staticmethod
    def stateless_loss(params, model, criterion, batch):
        """
        Need to perform forward pass and loss calculation in one function
        since we need gradients w.r.t params. 
        """
        x, y = batch
        logits = model(params, x)
        loss_val = criterion(logits, y)
        return loss_val, logits
    

    @staticmethod
    def train_step(params, model, criterion, batch, lr):
        grad_and_loss_fn = grad_and_value(OptExp.stateless_loss, has_aux=True)
        grads, aux_outputs = grad_and_loss_fn(params, model, criterion, batch)
        loss_val, logits = aux_outputs
        params = OptExp.sgd_step(params, grads, lr) 
        return params, loss_val, logits

    
    def train(self, num_epochs=10):
        print('Beginning training!')
        epoch_num = 0
        for epoch in range(num_epochs):
            epoch_num += 1
            tqdm_loader = tqdm(self.trainloader)
            for batch in tqdm_loader:
                tqdm_loader.set_description(f"Epoch {epoch_num}")
                self.params, loss_val, logits = OptExp.train_step(self.params, self.model, self.criterion, batch, self.lr)
                pdb.set_trace()
                loss_val = round(float(loss_val.item()), 3)

                # metric computation updates (per-batch basis)
                if self.metrics is not None:
                    aux_vals = {"loss": loss_val}
                    self.update_metrics(logits, batch[1], aux_vals)
            

            # metric print and reset (per-epoch basis)
            if self.metrics is not None:
                for key, value in self.metrics.items():
                    curr_val = value.get()
                    if self.verbose:
                        print(f"{key}: {curr_val}")
                    if self.logger is not None:
                        self.logger.log({str(key): curr_val})

                    value.reset()

        print('Finished training!')


    def debug(self):
        print(type(self.params))





class AccuracyMetric(ml.MeanMetric):
  """
  Scalar metric designed to use on a per-epoch basis 
  and updated on per-batch basis. For getting average across
  the epoch. 
  Inherit from rsbox.ml.MeanMetric and just change the update function. 
  Basic idea behind metrics: inherit from ml.MeanMetric and implement update(logits, labels). 
  """

  def update(self, logits, labels):
    new_val = torch.mean((torch.argmax(logits, -1) == labels).float())
    self.vals.append(new_val)


metrics_dict = {
        'loss': ml.MeanMetric(),
        'accuracy': AccuracyMetric()
    }

lr = 0.01 

exp = OptExp(
    metrics=metrics_dict,
    logger=wandb.init(
        project="functorch-dev", 
        entity="rosikand",
        name="lr-" + str(lr) + "-" + misc.timestamp()
    ),
    lr=lr
)
exp.train(num_epochs=500)

# exp.debug()
