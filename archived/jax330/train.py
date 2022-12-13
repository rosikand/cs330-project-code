import jax.numpy as jnp
from jax import random
import copy
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Dense, Relu, Flatten, Sigmoid)
from jax import grad, jit, vmap, tree_util, value_and_grad
import easyjax as ej
from easyjax import ml, metrics
import rsbox
import numpy as np
from rsbox import misc
from rsbox import ml as rml
import pdb
import optax
import data_gen
from tqdm.auto import tqdm
import pprint 


# globals
train_steps = 10
outer_lr = 0.01
num_inner_steps = 10
inner_lr = 0.01


# hyperparams 
class ConfigClass:
    data_path = "../../datasets/mini_omniglot"
    k = 5
    n = 3
    num_query = 10
    batch_size = 1    



def compute_accuracy(logits, y):
    """
    logits is of shape (b, num_classes)
    y is of shape (b,)
    """
    return jnp.mean(jnp.argmax(logits, -1) == y)


def avg_cross_entropy(logits, y):
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def stateless_loss(params, net_apply, loss_function, x, y):
    model_preds = net_apply(params, x)
    loss_value = loss_function(model_preds, y)
    return loss_value, model_preds


def inner_loop(x, y, model_params):
    support_accuracies = []
    support_losses = []
    params = model_params
    
    loss_grad_fn = value_and_grad(stateless_loss, has_aux=True)
    
    for i in tqdm(range(num_inner_steps)):
        aux_vals, grads = loss_grad_fn(params, applyfn, avg_cross_entropy, x, y)
        loss_val, logits = aux_vals
        pdb.set_trace()
        print("inner loss: ", loss_val)
        
        params = ml.update_step(params, grads, inner_lr)
        
        # metrics 
        support_losses.append(loss_val)
        support_accuracies.append(compute_accuracy(logits, y))

    # nth + 1 pred 
    # ... 
    
    support_metrics_dict = {
        "losses": support_losses, 
        "accuracies": support_accuracies
    }
    
    print("-------------------")
    
    return params, support_metrics_dict



def outer_step(model_params, batch):
    # return loss
    # loop over tasks in the batch 
    pre_adapt_support_losses = []
    post_adapt_support_losses = []
    pre_adapt_query_losses = []
    post_adapt_query_losses = []
    
    pre_adapt_support_accuracies = []
    post_adapt_support_accuracies = []
    pre_adapt_query_accuracies = []
    post_adapt_query_accuracies = []
    
    og_params = model_params
    
    
    for i in range(batch[0].shape[0]):
        sx, sy, qx, qy = batch[0][i], batch[1][i], batch[2][i], batch[3][i]
        
        # (pre adapt) query metrics 
        pre_adapt_logits = applyfn(og_params, qx)
        pre_adapt_query_losses.append(avg_cross_entropy(pre_adapt_logits, qy))
        pre_adapt_query_accuracies.append(compute_accuracy(pre_adapt_logits, qy))
        
        
        # adapt params 
        adapted_params, support_metrics_dict = inner_loop(sx, sy, og_params)
        support_losses = support_metrics_dict['losses']
        support_accuracies = support_metrics_dict['accuracies']
        pre_adapt_support_losses.append(support_losses[0])
        post_adapt_support_losses.append(support_losses[-1])
        pre_adapt_support_accuracies.append(support_accuracies[0])
        post_adapt_support_accuracies.append(support_accuracies[-1])
        
        
        # (post adapt) compute query loss and metrics 
        post_adapt_logits = applyfn(adapted_params, qx)
        post_adapt_query_losses.append(avg_cross_entropy(post_adapt_logits, qy))
        post_adapt_query_accuracies.append(compute_accuracy(post_adapt_logits, qy))
        
    
    
    # note: all metrics averaged over meta task batch 
    meta_loss_ = np.mean(post_adapt_query_losses)
    metrics_dict = {
        "pre_adapt_support_loss": np.mean(pre_adapt_support_losses), 
        "post_adapt_support_loss": np.mean(post_adapt_support_losses), 
        "pre_adapt_query_loss": np.mean(pre_adapt_query_losses),
        "post_adapt_query_loss": meta_loss_,
        "pre_adapt_support_accuracies": np.mean(pre_adapt_support_accuracies),
        "post_adapt_support_accuracies": np.mean(post_adapt_support_accuracies),
        "pre_adapt_query_accuracies": np.mean(pre_adapt_query_accuracies),
        "post_adapt_query_accuracies": np.mean(post_adapt_query_accuracies)
    }
    
    return meta_loss_, metrics_dict
        



def main():
    cfg = ConfigClass()
    loader = data_gen.get_gen_loader(cfg.data_path, cfg.k, cfg.n, cfg.num_query, cfg.batch_size)

    net_init, applyfn = stax.serial(
        Flatten,
        Dense(1*28*28), Relu,
        Dense(128), Relu,
        Dense(64), Relu,
        Dense(cfg.n)
    )
    rng = random.PRNGKey(0)
    in_shape = (-1, 1, 28, 28)
    out_shape, net_params = net_init(rng, in_shape)


    # train 
    model_params = net_params  # what we'll update 

    for epoch in range(train_steps):
        task_batch = next(loader)
        
        outer_fn_and_grad = value_and_grad(outer_step, has_aux=True)
        
        aux_data, meta_grads = outer_fn_and_grad(model_params, task_batch)
        meta_loss, metrics_dict = aux_data
        model_params = ml.update_step(model_params, meta_grads, outer_lr)
        
        # log 
        print(f"Epoch ({epoch}): ")
        pprint.pprint(metrics_dict)
        print('---------------------------------------')
            





if __name__ == '__main__':
    main()

