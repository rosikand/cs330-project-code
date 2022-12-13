import copy
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
import learn2learn
from learn2learn.utils import clone_module, update_module


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

class ThreeWayNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# -----------------------------------------------------------------------------------


class MAMLExperiment:
    def __init__(self, trainloader, model, criterion, num_inner_steps, inner_lr, outer_lr, logger=None, verbose=True) -> None:
        self.net = model  # (torch nn.module)
        self.meta_optimizer = torch.optim.Adam(self.net.parameters(), lr=outer_lr)
        self.trainloader = trainloader
        self.num_inner_steps = num_inner_steps
        self.criterion = criterion
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.logger = logger
        self.verbose = verbose
        

    def inner_loop(self, x, y, train=True):

        # metrics 
        support_accuracies = []
        support_losses = []

        # 1. instantiate a copy of the current model 
        # or instead of copying, just reinstaniate the model 
        model = copy.deepcopy(self.net)
        init_parameters = {
            v: torch.clone(v)
            for v in model.parameters()
        }
        # model = clone_module(self.net)
        # model = ThreeWayNet()

        # load state dict from self.net.parameters()? 
        # model.load_state_dict(self.net.state_dict())

        # note: does the above detach the computational graph and thus eliminate 2nd derivatives? 
        # https://discuss.pytorch.org/t/are-there-any-recommended-methods-to-clone-a-model/483/11

        # clone pytorch model while keeping computational graph
        # https://discuss.pytorch.org/t/how-to-clone-a-model/483/2


        # 2. update params num inner steps times 
        for i in range(self.num_inner_steps):
            # 2.1 compute loss
            y_hat = model(x)
            loss = self.criterion(y_hat, y)
            
            # 2.2 metrics 
            support_losses.append(loss) 
            step_accuracy = utils.score(F.softmax(y_hat, dim=-1), y)
            support_accuracies.append(step_accuracy)
            
            # 2.3 compute gradients 
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # 2.4 update params
            # updates = [-self.inner_lr * g for g in grads]
            # update_module(model, updates=updates)
            for param, grad in zip(model.parameters(), grads):
                param.data -= self.inner_lr * grad

        
        # 3. last prediction 
        y_hat = model(x)
        loss = self.criterion(y_hat, y)
        support_losses.append(loss)
        step_accuracy = utils.score(F.softmax(y_hat, dim=-1), y)
        support_accuracies.append(step_accuracy)

        # 4. return model and metrics
        # adapted_parameters = model.parameters()

        # pprint.pprint(support_losses)

        # print('---------------------------------------')

     

        return model, init_parameters, support_losses, support_accuracies  
    


    def outer_step(self, task_batch, train=True):


        pre_adapt_support_losses = []
        post_adapt_support_losses = []
        pre_adapt_query_losses = []
        post_adapt_query_losses = []
            
        pre_adapt_support_accuracies = []
        post_adapt_support_accuracies = []
        pre_adapt_query_accuracies = []
        post_adapt_query_accuracies = []
    


        for task in task_batch:
            support_x, support_y, query_x, query_y = task

            # 0. pre-adaptation query metrics
            pre_adapt_query_y_hat = self.net(query_x)
            pre_adapt_query_loss = self.criterion(pre_adapt_query_y_hat, query_y)
            pre_adapt_query_accuracy = utils.score(F.softmax(pre_adapt_query_y_hat, dim=-1), query_y)
            pre_adapt_query_losses.append(pre_adapt_query_loss)
            pre_adapt_query_accuracies.append(pre_adapt_query_accuracy)

            # 1. compute adapted parameters
            adapted_model, init_parameters, support_losses, support_accuracies = self.inner_loop(support_x, support_y, train=train)
            
            
            # 2. compute query loss
            query_y_hat = adapted_model(query_x)
            query_loss = self.criterion(query_y_hat, query_y)
            query_accuracy = utils.score(F.softmax(query_y_hat, dim=-1), query_y)

            

            # 3. metrics 
            pre_adapt_support_losses.append(support_losses[0])
            post_adapt_support_losses.append(support_losses[-1])
            pre_adapt_support_accuracies.append(support_accuracies[0])
            post_adapt_support_accuracies.append(support_accuracies[-1])
            

            post_adapt_query_accuracies.append(query_accuracy)
            post_adapt_query_losses.append(query_loss)

        

        meta_loss_ = sum(post_adapt_query_losses)/len(post_adapt_query_losses)

        # 4. compute gradients
        query_grads = torch.autograd.grad(meta_loss_, init_parameters, create_graph=True)

        metrics_dict = {
            "pre_adapt_support_loss": (sum(pre_adapt_support_losses)/len(pre_adapt_support_losses)).item(), 
            "post_adapt_support_loss": (sum(post_adapt_support_losses)/len(post_adapt_support_losses)).item(), 
            "pre_adapt_query_loss": (sum(pre_adapt_query_losses)/len(pre_adapt_query_losses)).item(),
            "post_adapt_query_loss (Meta Loss)": round(meta_loss_.item(), 10),
            "pre_adapt_support_accuracies": np.mean(pre_adapt_support_accuracies),
            "post_adapt_support_accuracies": np.mean(post_adapt_support_accuracies),
            "pre_adapt_query_accuracies": np.mean(pre_adapt_query_accuracies),
            "post_adapt_query_accuracies": np.mean(post_adapt_query_accuracies)
        }

        return meta_loss_, query_grads, metrics_dict

    
    def train(self, num_epochs):
        # looper = 0  # for meta param diff checking 
        for epoch in range(num_epochs):
            for i, task_batch in enumerate(self.trainloader):
                # looper += 1


                # self.meta_optimizer.zero_grad()
                meta_loss, query_grads, metrics_dict = self.outer_step(task_batch, train=True)

                for param, grad in zip(self.net.parameters(), query_grads):
                    param.data -= self.outer_lr * grad

                # # 1. update meta params
                # # updates = [-self.meta_lr * g for g in query_grads]

                # for names in self.net.named_parameters():

                #     param.data -= self.outer_lr * grad

                # meta_loss.backward()
                # self.meta_optimizer.step()

                # print("Params at Epoch: {}, Batch: {}".format(epoch, i))
                # print(list(self.net.parameters()))
                # print('---------------------------------------')

                if self.logger is not None:
                    self.logger.log(metrics_dict)
                
                if self.verbose:
                    print(f"Epoch ({epoch}), Task ({i}): ")
                    pprint.pprint(metrics_dict)
                    print('---------------------------------------')
                # if looper == 3:
                #     break




# -------------------------------------------------------------------------------------------


# class HigherMaml:
#     def __init__(self, trainloader, model, criterion, num_inner_steps, inner_lr, outer_lr, logger=None, verbose=True) -> None:
#         self.net = model  # (torch nn.module)
#         self.meta_optimizer = torch.optim.Adam(self.net.parameters(), lr=outer_lr)
#         self.trainloader = trainloader
#         self.num_inner_steps = num_inner_steps
#         self.criterion = criterion
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.logger = logger
#         self.verbose = verbose
    

#     def train(self):
#         # implement maml training loop using higher 
#         torch.optim.Adam(self.net.parameters())



    

    






# -------------------------------------------------------------------------------------------


def main(args):
    if args.config is None:
        config_class = 'BaseConfig'
    else:
        config_class = args.config
    
    cfg = getattr(configs, config_class)
    
    maml_exp = MAMLExperiment(
            trainloader=cfg.trainloader, 
            model=cfg.model, 
            criterion=cfg.criterion, 
            num_inner_steps=cfg.num_inner_steps, 
            inner_lr=cfg.inner_lr, 
            outer_lr=cfg.outer_lr, 
            logger=cfg.logger,
            verbose = cfg.verbose
        )
    maml_exp.train(num_epochs=cfg.num_epochs)



if __name__ == "__main__":
    # configure args 
    parser = argparse.ArgumentParser(description="specify cli arguments.", allow_abbrev=True)
    parser.add_argument("-config", type=str, help='specify config.py class to use.') 
    args = parser.parse_args()
    main(args)




