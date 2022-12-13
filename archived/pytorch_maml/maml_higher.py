import torch 
import utils 
import numpy as np
import data
import torch.nn.functional as F
import torch.nn as nn
import pdb
import pprint
import wandb
import higher
import rsbox 
import configs
import argparse
from rsbox import ml, misc 



class FiveWayNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        # x = 1 - x  # invert the image colors to match MNIST and CS 330 setup 
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x





def train():
    # config 
    model = FiveWayNet()
    outer_opt = torch.optim.Adam(model.parameters())
    inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 10
    num_inner_steps = 10
    criterion = nn.CrossEntropyLoss()
    trainloader = data.get_loader(root_path="datasets/universal_omniglot", k=5, n=5, num_query=10, batch_size=16, normalize=False)
    logger = wandb.init(project = "higher-maml-dev", entity = "rosikand", name = misc.timestamp())

    # training loop 
    for outer_step in range(num_epochs):
        for i, task_batch in enumerate(trainloader):
            meta_losses = []
            pre_adapt_support_losses = []
            post_adapt_support_losses = []
            pre_adapt_query_losses = []
                
            pre_adapt_support_accuracies = []
            post_adapt_support_accuracies = []
            pre_adapt_query_accuracies = []
            post_adapt_query_accuracies = []

            for task in task_batch:
                sx, sy, qx, qy = task
                # inner loop 
                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    # pre adapt metrics 
                    pre_support_logits = fmodel(sx)
                    pre_adapt_support_losses.append(criterion(pre_support_logits, sy))
                    pre_adapt_support_accuracies.append(utils.score(F.softmax(pre_support_logits, dim=-1), sy))
                    pre_query_logits = fmodel(qx)
                    pre_adapt_query_losses.append(criterion(pre_query_logits, qy))
                    pre_adapt_query_accuracies.append(utils.score(F.softmax(pre_query_logits, dim=-1), qy))


                    # adapt 
                    for inner_step in range(num_inner_steps):
                        loss = criterion(fmodel(sx), sy)
                        diffopt.step(loss)
                    
                    # meta loss step 
                    query_y_hat = fmodel(qx)
                    meta_loss = criterion(query_y_hat, qy)
                    meta_losses.append(meta_loss)

                    # post adapt metrics 
                    post_support_logits = fmodel(sx)
                    post_adapt_support_losses.append(criterion(post_support_logits, sy))
                    post_adapt_support_accuracies.append(utils.score(F.softmax(post_support_logits, dim=-1), sy))
                    post_adapt_query_accuracies.append(utils.score(F.softmax(query_y_hat, dim=-1), qy))

            
            outer_opt.zero_grad()
            maml_loss = sum(meta_losses) / len(meta_losses)
            maml_loss.backward()
            outer_opt.step()

            # logging
            metrics_dict = {
                "maml_loss": maml_loss.item(),
                "pre_adapt_support_loss": sum(pre_adapt_support_losses) / len(pre_adapt_support_losses),
                "post_adapt_support_loss": sum(post_adapt_support_losses) / len(post_adapt_support_losses),
                "pre_adapt_query_loss": sum(pre_adapt_query_losses) / len(pre_adapt_query_losses),
                "pre_adapt_support_accuracy": sum(pre_adapt_support_accuracies) / len(pre_adapt_support_accuracies),
                "post_adapt_support_accuracy": sum(post_adapt_support_accuracies) / len(post_adapt_support_accuracies),
                "pre_adapt_query_accuracy": sum(pre_adapt_query_accuracies) / len(pre_adapt_query_accuracies),
                "post_adapt_query_accuracy": sum(post_adapt_query_accuracies) / len(post_adapt_query_accuracies),
            }

            logger.log(metrics_dict)

            print(f"Metrics (outer step {outer_step}, task batch {i}): ")
            pprint.pprint(metrics_dict)
            print('-'*100)



def train_gd():
    # update every epoch instead of per-batch 
    # config 
    model = FiveWayNet()
    outer_opt = torch.optim.Adam(model.parameters())
    inner_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    num_epochs = 10
    num_inner_steps = 10
    criterion = nn.CrossEntropyLoss()
    trainloader = data.get_loader(root_path="datasets/universal_omniglot", k=5, n=5, num_query=10, batch_size=1, normalize=False)
    logger = wandb.init(project = "higher-maml-dev", entity = "rosikand", name = misc.timestamp())

    # training loop 
    for outer_step in range(num_epochs):
        meta_losses = []
        for i, task_batch in enumerate(trainloader):
            for task in task_batch:
                sx, sy, qx, qy = task
                # inner loop 
                with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                    for inner_step in range(num_inner_steps):
                        loss = criterion(fmodel(sx), sy)
                        diffopt.step(loss)
        
                    meta_loss = criterion(fmodel(qx), qy)
                    meta_losses.append(meta_loss)
            
        outer_opt.zero_grad()
        maml_loss = sum(meta_losses) / len(meta_losses)
        maml_loss.backward()
        outer_opt.step()

        # logging
        logger.log({"maml_loss": maml_loss.item()})
        print(f"Meta Loss (outer step {outer_step}): {maml_loss}")



        
def main():
    train()




if __name__ == "__main__":
    main()
