"""
File: higher_maml.py
------------------
MAML with higher. 
"""


import torch 
import torch.nn.functional as F
import os
import rsbox
import utils
from rsbox import misc
from tqdm.auto import tqdm
from functorch import grad, grad_and_value
import wandb 
import utils
import pdb
import functorch
from functorch import grad, grad_and_value




# base functorch MAML 
class MAML:
    def __init__(self, trainloader, meta_optimizer, model, criterion, num_inner_steps, inner_lr, logger):
        
        # wrapper for a torch nn.module 
        self.net = model  # (torch nn.module)
        self.trainloader = trainloader
        self.meta_optimizer = meta_optimizer
        self.model_apply, self.params = functorch.make_functional(self.net)
        self.num_inner_steps = num_inner_steps
        self.criterion = criterion
        self.inner_lr = inner_lr
        self.logger = logger
        self.task_idx = 0


    def _forward(self, x, params):
        """returns logits given x, params"""
        logits = self.model_apply(params, x)
        return logits 

    
    @staticmethod
    def sgd_step(params, gradients, lr):
        """one gradient step for updating the (inner) weights"""
        updated_params = []
        for param, gradient in zip(params, gradients):
            update = param - (lr * gradient)
            updated_params.append(update)
        
        return tuple(updated_params)

    
    @staticmethod
    def stateless_inner_loss(params, model_apply, criterion, x, y):
        """
        Need to perform forward pass and loss calculation in one function
        since we need gradients w.r.t params (must be args[0]). The first
        value we return also needs to be the scalar loss value.  
        """
        logits = model_apply(params, x)
        loss_val = criterion(logits, y)
        return loss_val, logits


    def inner_loop(self, samples, labels, task_id):
        """returns updated params and (accuracy) on support set"""
        # task-specific adaptation 

        # do the gradient step here (update phi)

        accuracies = []
        losses = []
        parameters = self.params
        # check type of labels (convert to tensor?)

        grad_and_loss_fn = grad_and_value(self.stateless_inner_loss, has_aux=True) 
        for i in range(self.num_inner_steps):
            grads, aux_outputs = grad_and_loss_fn(parameters, self.model_apply, self.criterion, samples, labels)
            loss_val, logits = aux_outputs
            # print(f"inner loss step {i}, task {task_id}: ", loss_val)
            losses.append(loss_val)

            # update 
            parameters = self.sgd_step(parameters, grads, self.inner_lr) 

            # metric 
            probs = F.softmax(logits, dim=1)
            step_accuracy = utils.score(probs, labels)
            accuracies.append(step_accuracy)
              
            if self.logger is not None:
                loss_descripter = "inner_support_loss_task_" + str(task_id)
                accuracy_descripter = "inner_support_accuracy_task_" + str(task_id)
                self.logger.log({loss_descripter: loss_val, accuracy_descripter: step_accuracy})
        

        # Lth + 1 accuracy 
        final_accuracy = utils.score(F.softmax(self._forward(samples, parameters), dim=1), labels)
        accuracies.append(final_accuracy)


        support_accuracy = sum(accuracies)/len(accuracies)

        return parameters, final_accuracy 
    


    def outer_step(self, batch):
        # step for training outer parameters (theta)
        # torch.optim for theta 

        # temp solution to batching problem
        # batch = torch.unbind(batch)
        batch = [batch]

        accuracy_query_batch = []
        accuracy_pre_adapt_query_batch = []
        outer_loss_batch = []
        accuracies_support_batch = []
        for task in batch:
            self.task_idx += 1
            support_samples, support_labels, query_samples, query_labels = task
            support_samples = support_samples.squeeze()
            support_labels = support_labels.squeeze()
            query_samples = query_samples.squeeze()
            query_labels = query_labels.squeeze()


            # get query metrics pre-adaptation 
            query_logits_p = self._forward(query_samples, self.params)
            query_accuracy_p = utils.score(F.softmax(query_logits_p, dim=1), query_labels)
            accuracy_pre_adapt_query_batch.append(query_accuracy_p)


            # get adapted params 
            adapted_params, support_accuracy = self.inner_loop(support_samples, support_labels, self.task_idx)
            accuracies_support_batch.append(support_accuracy)

            # get query metrics 
            # query_samples = torch.cat(query_samples, 0)  # vectorize 
            query_logits = self._forward(query_samples, adapted_params)
            query_loss = self.criterion(query_logits, query_labels)
            outer_loss_batch.append(query_loss)
            query_accuracy = utils.score(F.softmax(query_logits, dim=1), query_labels)
            accuracy_query_batch.append(query_accuracy)
            
        # average and return 
        outer_loss = sum(outer_loss_batch)/len(outer_loss_batch)
        support_accuracy_mean = sum(accuracies_support_batch)/len(accuracies_support_batch)
        query_accuracy_mean = sum(accuracy_query_batch)/len(accuracy_query_batch) 
        accuracy_pre_adapt_query_batch_mean = sum(accuracy_pre_adapt_query_batch)/len(accuracy_pre_adapt_query_batch)

        return outer_loss, support_accuracy_mean, accuracy_pre_adapt_query_batch_mean, query_accuracy_mean



    def train(self, num_epochs=10):
        """
        Training loop. 
        """
        self.net.train()
        epoch_num = 0

        for epoch in range(num_epochs):
            epoch_num += 1
            running_meta_loss = 0.0
            running_meta_support_accuracy = 0.0
            running_meta_query_accuracy = 0.0
            running_meta_pre_adaptation_query_accuracy = 0.0 
            
            tqdm_loader = tqdm(self.trainloader)
            b_idx = 0
            for batch in tqdm_loader:
                b_idx += 1
                tqdm_loader.set_description(f"Epoch {epoch_num}")

                # update 
                self.meta_optimizer.zero_grad()
                loss, support_acc, query_pre_acc, query_post_acc = self.outer_step(batch)
                loss.backward()
                self.meta_optimizer.step()
                running_meta_loss += loss.item() 
                running_meta_support_accuracy += support_acc
                running_meta_query_accuracy += query_post_acc
                running_meta_pre_adaptation_query_accuracy += query_pre_acc


            # metrics 
            epoch_avg_loss = running_meta_loss/b_idx
            epoch_support_accuracy = running_meta_support_accuracy/b_idx
            epoch_query_accuracy = running_meta_query_accuracy/b_idx
            epoch_pre_adaptation_accuracy = running_meta_pre_adaptation_query_accuracy/b_idx
            
            log_dict = {
                    "meta_loss": epoch_avg_loss,
                    "meta_support_acc": epoch_support_accuracy,
                    "meta_pre_adaptation_query_acc": epoch_pre_adaptation_accuracy,
                    "meta_query_acc": epoch_query_accuracy
                    }


            for k, v in log_dict.items(): 
                print(k, ": ", v)

            
            if self.logger is not None:
                self.logger.log(log_dict)


    def save_checkpoint(self):
        pass


    def save_weights(self):
        pass


    def load_checkpoint(self):
        pass
    

    def test(self):
        pass


    def debug(self):
        pass
